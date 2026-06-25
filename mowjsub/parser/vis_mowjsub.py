import glob
import os
import time

import click
import dask.array as da
import dask.multiprocessing
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
from mowjsub.fitfuncs import FitBSpline, FitPolynomial
from mowjsub.utils import get_ds_from_msdsl, ms_to_xarray_dataset
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
    order = opts.order[0]
    nworkers = opts.nworkers
    outchunks = dict(time=opts.time_chunks, bl_chunks=opts.bl_chunks)
    input_column = opts.input_column
    output_column = opts.output_column
    zarr_name = opts.load_from_cache
    cont_tol = opts.cont_fit_tol

    if opts.load_from_cache:
        temp_zarr = zarr_name
    else:
        temp_zarr = ms_to_xarray_dataset(ms, spwid, fieldid, chunksize, save_to_zarr=True)
        temp_zarr = "tmp.zarr"

    ds = xr.open_zarr(temp_zarr, chunks=outchunks)

    xspec = ds.coords["FREQ"]

    futures = []

    if method == "spline":
        fitfunc = FitBSpline(xspec, order=order, velwidth=velwidth, fit_tol=cont_tol)
    elif method == "polynomial":
        fitfunc = FitPolynomial(xspec, order=order, fit_tol=cont_tol)
    else:
        raise ValueError(f"Unknown fitting method: {method}. Supported methods: 'spline', 'polynomial'.")

    base_dims = "TIME, BASELINE, FREQ, CORR"
    signature = f"(FREQ),({base_dims}),({base_dims}),({base_dims}) -> ({base_dims})"
    meta = (np.ndarray((), ds.VIS.dtype),)

    dask.config.set(scheduler="threads", num_workers=nworkers)

    for biter, dblock in enumerate(ds.VIS.data.blocks):
        # if biter > 0:
        # continue
        flags = ds.FLAG.data.blocks[biter]
        weights = ds.WEIGHT.data.blocks[biter]

        contfit = VisContSub(fitfunc)
        get_cont = da.gufunc(
            contfit.vis_cont_sub,
            signature=signature,
            meta=meta,
            allow_rechunk=True,
        )
        futures.append(
            get_cont(
                xspec,
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

    msds = get_ds_from_msdsl(ms_dsl, spwid, fieldid)

    ms_ds = msds.assign(
        **{
            output_column: (
                ("row", "chan", "corr"),
                getattr(msds, input_column).data - continuum.data,
            ),
        }
    )

    if opts.output_ms:
        ms_name = opts.output_ms
        writes = [xds_to_table(ms_ds, ms_name, columns=["FLAG", "WEIGHT", output_column])]
        print(f"Writing new MS with FLAG, WEIGHT, and {output_column}")

    else:
        writes = [xds_to_table(ms_ds, ms, [output_column])]
        print(f"Writing line data to column '{output_column}' in {ms}...")

    with TqdmCallback(desc="Writing line data to MS"):
        da.compute(writes)
    print(f"UV plane continuum subtraction completed. Data written to column '{output_column}' in {ms}.")

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime / 3600)
    mins = dtime / 60 - hours * 60
    secs = (mins % 1) * 60
    log.info(f"Runtime {hours}:{int(mins)}:{secs:.1f}")
