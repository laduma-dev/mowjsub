import glob
import os
import time

import click
import dask
import dask.array as da
import numpy as np
import xarray as xr
from daskms import xds_from_ms, xds_to_table
from omegaconf import OmegaConf
from scabha import init_logger
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader
from tqdm.dask import TqdmCallback

import mowjsub
from mowjsub import BIN
from mowjsub.fitfuncs import (
    FitBSpline,
    FitGCVSpline,
    FitMedFilter,
    FitMedFilterFast,
    FitPolynomial,
)
from mowjsub.utils import (
    copy_ms_subtables,
    doppler_regrid_dataset,
    finalise_regridded_ms,
    get_ds_from_msdsl,
    ms_to_xarray_dataset,
    output_ms_dataset,
)
from mowjsub.visibility_plane import VisContSub

log = init_logger(BIN.vis_plane)

command = BIN.vis_plane
thisdir = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/vis_mowjsub.yaml")
config = paramfile_loader(parserfile, sources)["vis_mowjsub"]


@click.command("vis-mowjsub")
@click.version_option(str(mowjsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    start_time = time.time()

    opts = OmegaConf.create(kwargs)
    ms = opts.ms
    spwid = opts.spwid
    fieldid = opts.field_id
    chunksize = opts.row_chunks
    velwidth = opts.vel_width or opts.segments
    method = opts.fit_model
    order = opts.order
    nworkers = opts.nworkers

    if method in ("spline", "b-spline", "polynomial") and order is None:
        raise RuntimeError(f"The parameter 'order' is required for fit-model={method}.")

    if method in ("spline", "b-spline", "median-filter", "scipy-median-filter") and velwidth is None:
        raise RuntimeError(f"The parameter 'vel-width' is required for fit-model={method}.")

    doppler_frame = opts.doppler_frame
    # Fail before the fit rather than after it: regridding changes the channel
    # count, so the result cannot go back into a column of the input MS.
    if doppler_frame and not opts.output_ms:
        raise RuntimeError(f"--doppler-frame={doppler_frame} changes the channel grid, so it needs a separate output MS. Pass --output-ms.")
    outchunks = dict(time=opts.time_chunks, bl_chunks=opts.bl_chunks)
    input_column = opts.input_column
    output_column = opts.output_column

    # Writing back into the input MS is the one path where the output column can
    # destroy the data the fit was made from, and it cannot be undone.
    if output_column == input_column and not opts.output_ms:
        raise RuntimeError(
            f"--output-column={output_column} is the column being read, and without --output-ms the residual is written back into {ms}, "
            f"overwriting it. Pass --output-ms to write a new MS, or choose a different --output-column."
        )
    zarr_name = opts.load_from_cache
    cont_tol = opts.cont_fit_tol

    if opts.load_from_cache:
        temp_zarr = zarr_name
    else:
        temp_zarr = ms_to_xarray_dataset(ms, spwid, fieldid, chunksize, save_to_zarr=True)
        temp_zarr = "tmp.zarr"

    ds = xr.open_zarr(temp_zarr, chunks=outchunks)

    xspec = np.asarray(ds.coords["FREQ"])

    futures = []

    if method in ("spline", "b-spline"):
        fitfunc = FitBSpline(xspec, order=order, velwidth=velwidth, fit_tol=cont_tol)
    elif method == "polynomial":
        fitfunc = FitPolynomial(xspec, order=order, fit_tol=cont_tol)
    elif method == "median-filter":
        fitfunc = FitMedFilter(xspec, velwidth=velwidth, fit_tol=cont_tol)
    elif method == "scipy-median-filter":
        fitfunc = FitMedFilterFast(xspec, velwidth=velwidth, fit_tol=cont_tol)
    elif method == "gcv-spline":
        fitfunc = FitGCVSpline(xspec, fit_lam=opts.gcv_lambda, fit_tol=cont_tol)
    else:
        raise ValueError(f"Unknown fitting method: {method}.")

    fitfunc.prepare()

    base_dims = "TIME, BASELINE, FREQ, CORR"
    signature = f"({base_dims}),({base_dims}),({base_dims}) -> ({base_dims})"
    meta = (np.ndarray((), ds.VIS.dtype),)

    dask.config.set(scheduler="threads", num_workers=nworkers)

    contfit = VisContSub(fitfunc)
    get_cont = da.gufunc(
        contfit.vis_cont_sub,
        signature=signature,
        meta=meta,
        allow_rechunk=True,
    )

    for biter, dblock in enumerate(ds.VIS.data.blocks):
        flags = ds.FLAG.data.blocks[biter]
        weights = ds.WEIGHT.data.blocks[biter]

        futures.append(
            get_cont(
                dblock,
                flags,
                weights,
            ),
        )

    continuum_dask = da.concatenate(futures)

    continuum_xarray = xr.DataArray(data=continuum_dask, dims=ds.VIS.dims, coords=ds.VIS.coords)

    continuum = continuum_xarray.stack(row=("time", "baseline"))
    continuum = continuum.transpose("row", ...).chunk({"row": chunksize})

    ms_dsl = xds_from_ms(
        ms,
        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": chunksize},
    )

    msds = get_ds_from_msdsl(ms_dsl, field_id=fieldid, data_desc_id=spwid)

    line_data = getattr(msds, input_column).data - continuum.data

    if doppler_frame:
        # The continuum was fitted on the native topocentric grid, where the
        # bandpass structure it models is stationary; only the residual is moved
        # onto the Doppler-corrected grid.
        source_vel = opts.doppler_source_vel
        regrid_ds, freqs_out, chanwidth = doppler_regrid_dataset(
            msds,
            ms,
            line_data,
            output_column,
            spwid,
            fieldid,
            frame=doppler_frame,
            chan_grid=opts.doppler_chan_grid,
            interpolation=opts.doppler_interpolation,
            source_vel=None if source_vel is None else source_vel * 1e3,
        )

        ms_name = str(opts.output_ms)
        writes = [xds_to_table([regrid_ds], ms_name, columns="ALL")]
        with TqdmCallback(desc="Writing Doppler-corrected line data"):
            da.compute(writes)

        finalise_regridded_ms(ms, ms_name, spwid, freqs_out, chanwidth, doppler_frame)
        log.info(f"UV plane continuum subtraction completed. Doppler-corrected line data written to column '{output_column}' in {ms_name}.")
    elif opts.output_ms:
        # A new MS needs every row column plus the subtables, not just the line
        # data: anything left out here is simply absent from the result.
        ms_name = str(opts.output_ms)
        dims = ("row", "chan", "corr")
        out_ds = output_ms_dataset(
            msds,
            {
                output_column: (dims, line_data),
                "FLAG": (dims, msds.FLAG.data),
                "WEIGHT_SPECTRUM": (dims, msds.WEIGHT_SPECTRUM.data),
            },
            spwid,
            fieldid,
        )

        writes = [xds_to_table([out_ds], ms_name, columns="ALL")]
        with TqdmCallback(desc="Writing line data to new MS"):
            da.compute(writes)

        copy_ms_subtables(ms, ms_name)
        log.info(f"UV plane continuum subtraction completed. Line data written to column '{output_column}' in {ms_name}.")
    else:
        ms_ds = msds.assign(
            **{
                output_column: (
                    ("row", "chan", "corr"),
                    line_data,
                ),
            }
        )

        writes = [xds_to_table(ms_ds, ms, [output_column])]
        with TqdmCallback(desc="Writing line data to MS"):
            da.compute(writes)
        log.info(f"UV plane continuum subtraction completed. Line data written to column '{output_column}' in {ms}.")

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime / 3600)
    mins = dtime / 60 - hours * 60
    secs = (mins % 1) * 60
    log.info(f"Runtime {hours}:{int(mins)}:{secs:.1f}")
