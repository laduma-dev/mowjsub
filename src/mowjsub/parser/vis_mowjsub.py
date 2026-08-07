from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import dask
import dask.array as da
import numpy as np
import shinobi
import xarray as xr
from daskms import xds_from_ms, xds_to_table
from pydantic import Field
from tqdm.dask import TqdmCallback

from mowjsub import BIN, set_logger
from mowjsub.fitfuncs import (
    ORDER_MODELS,
    WIDTH_MODELS,
    build_fitfunc,
)
from mowjsub.parser._cli import make_command
from mowjsub.utils import (
    copy_ms_subtables,
    doppler_regrid_dataset,
    finalise_regridded_ms,
    get_ds_from_msdsl,
    ms_to_xarray_dataset,
    output_ms_dataset,
    require_distinct_ms,
)
from mowjsub.visibility_plane import VisContSub

app = BIN.vis_plane

FIT_MODELS = Literal["b-spline", "spline", "polynomial", "median-filter", "scipy-median-filter", "gcv-spline"]
FRAMES = Literal["topo", "geo", "bary", "lsrk", "lsrd", "galacto", "lgroup", "cmb", "source"]
INTERPOLATIONS = Literal["nearest", "linear"]


def _contsub_block(vis, flags, weights, spec, seed):
    """Subtract the continuum from one chunk, on a fitter of the chunk's own."""
    return VisContSub(build_fitfunc(spec, seed)).vis_cont_sub(vis, flags, weights)


def runit(opts):
    start_time = time.time()

    log = set_logger()
    ms = opts.ms
    spwid = opts.spwid
    fieldid = opts.field_id
    chunksize = opts.row_chunks
    velwidth = opts.vel_width or opts.segments
    chanwidth = opts.chan_width
    method = opts.fit_model
    order = opts.order
    nworkers = opts.nworkers

    if method in ORDER_MODELS and order is None:
        raise RuntimeError(f"The parameter 'order' is required for fit-model={method}.")

    if velwidth is not None and chanwidth is not None:
        raise RuntimeError("--vel-width and --chan-width both set the width of the fit window, in different units. Give one, not both.")

    if chanwidth is not None and chanwidth < 1:
        raise RuntimeError(f"--chan-width={chanwidth} is not a channel count. It must be at least 1.")

    if method in WIDTH_MODELS and velwidth is None and chanwidth is None:
        raise RuntimeError(f"The parameter 'vel-width' (or legacy 'segments') or 'chan-width' is required for fit-model={method}.")

    doppler_frame = opts.doppler_frame
    # Fail before the fit rather than after it: regridding changes the channel
    # count, so the result cannot go back into a column of the input MS.
    if doppler_frame and not opts.output_ms:
        raise RuntimeError(f"--doppler-frame={doppler_frame} changes the channel grid, so it needs a separate output MS. Pass --output-ms.")

    require_distinct_ms(ms, opts.output_ms)

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

    spec = dict(
        fit_model=method,
        xspec=xspec,
        order=order,
        velwidth=velwidth,
        chanwidth=chanwidth,
        fit_tol=cont_tol,
        gcv_lambda=opts.gcv_lambda,
    )
    # Validate the configuration once, here, rather than letting a bad
    # combination surface from inside a worker with the MS already being read.
    # Discarded -- the fitters the tasks use are their own.
    build_fitfunc(spec)

    base_dims = "TIME, BASELINE, FREQ, CORR"
    signature = f"({base_dims}),({base_dims}),({base_dims}) -> ({base_dims})"
    meta = (np.ndarray((), ds.VIS.dtype),)

    dask.config.set(scheduler="threads", num_workers=nworkers)

    # One fitter per chunk, constructed inside the task, for the reason given at
    # the same place in the image plane: `FitBSpline` draws its knot offsets
    # from `self.rng`, a numpy Generator is not thread-safe, and these tasks run
    # as threads. One shared `VisContSub` used to serve every chunk.
    nblocks = ds.VIS.data.numblocks[0]
    seeds = [int(seed) for seed in np.random.SeedSequence().generate_state(nblocks)]

    for biter, dblock in enumerate(ds.VIS.data.blocks):
        flags = ds.FLAG.data.blocks[biter]
        weights = ds.WEIGHT.data.blocks[biter]

        get_cont = da.gufunc(
            partial(_contsub_block, spec=spec, seed=seeds[biter]),
            signature=signature,
            meta=meta,
            allow_rechunk=True,
        )

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
        # `grid_chanwidth`, not `chanwidth`: this is the output grid's channel
        # width in Hz, a different quantity from the fit-window width in
        # channels above, which is still needed nowhere below but would be
        # silently clobbered by the shorter name.
        regrid_ds, freqs_out, grid_chanwidth = doppler_regrid_dataset(
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

        finalise_regridded_ms(ms, ms_name, spwid, freqs_out, grid_chanwidth, doppler_frame)
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


@shinobi.pystep(name=app, info="Perform visibility plane continuum subtraction on an input Measurement Set (MS)")
def vis_mowjsub(
    ms: Path = Field(..., description="Input MS file"),
    input_column: str = Field("DATA", description="Column which contains the data to be continuum subtracted."),
    output_column: str = Field(
        ...,
        description=(
            "Column name to write the continuum subtracted data to. Required: there is no standard MS column for continuum-subtracted "
            "visibilities, so naming one is left to you. Pass DATA if the imager you feed the result to expects that column -- but note "
            "that without --output-ms this writes back into the input MS, so it must differ from --input-column."
        ),
    ),
    fit_model: FIT_MODELS = Field(
        "b-spline",
        description=(
            "Fit function to model the continuum. The 'scipy-median-filter' model is much faster than 'median-filter', but treats band "
            "edges and masked channels differently, so the two do not give identical continuum models. WARNING: A median-filter continuum "
            "model may subsume low SNR line emission, use it with great care."
        ),
    ),
    order: int | None = Field(None, description="Order of spline/polynomial or number of top coefficients to use for DCT reconstruction"),
    vel_width: float | None = Field(None, description="Width of spline segments or median filter window in km/s. Give this or --chan-width, not both."),
    chan_width: int | None = Field(
        None,
        description=(
            "Width of spline segments or median filter window in number of channels. The alternative to --vel-width, for a caller that "
            "wants the window fixed in channels rather than converted from a velocity against the band centre. Give one or the other, not both."
        ),
    ),
    gcv_lambda: float | None = Field(
        None,
        description=(
            "GCV spline penalty. Zero is equivalent to an interpolating spline, high values lead to a flatter curve. If unset the parameter "
            "will be estimated using the GCV criterion; this can be very slow. Experience suggests that values chanwidth/nchan work best."
        ),
    ),
    segments: float | None = Field(
        None,
        description=(
            "## This has been replaced by --vel-width. It will be removed in future releases ## Width of spline segments or median filter "
            "window in km/s. If given as a list, then it must have same size as --order."
        ),
    ),
    spwid: int = Field(0, description="Spectral Window ID"),
    field_id: int = Field(0, description="Field ID"),
    row_chunks: int = Field(10000, description="Chunking strategy (Done along the time axis)"),
    time_chunks: int = Field(64, description="Chunk size for time axis"),
    bl_chunks: int = Field(10, description="Chunk size for baseline axis"),
    cont_fit_tol: float = Field(
        0,
        description=(
            "Minimum percentage of valid spectrum data points required to do a fit. If the percentage of data points is below this percentage, original data will be returned."
        ),
    ),
    nworkers: int = Field(
        1,
        description=(
            "Number of parallel worker threads, one row chunk each. RAISING THIS USUALLY MAKES THE RUN SLOWER, which is why it "
            "defaults to 1. Dask runs these as threads, and VisContSub.vis_cont_sub is a nested Python loop over time, baseline "
            "and correlation around a scipy call, so the loop holds the GIL and the workers contend for it rather than working in "
            "parallel. This follows the image plane on mechanism rather than on measurement -- the numbers are in --nworkers on "
            "im-mowjsub, where the same loop shape was timed at 4.3 min against 26.5 min for 1 worker against 24. Raise it only "
            "after measuring on your own data."
        ),
    ),
    output_ms: Path | None = Field(None, description="If provided, write the output to a new MS with this name. Otherwise, add new column to the input MS."),
    load_from_cache: Path | None = Field(None, description="Load the MS from a cache (give Zarr file name) if available, otherwise create it."),
    doppler_frame: FRAMES | None = Field(
        None,
        description=(
            "Spectral reference frame to Doppler-correct the output to. When set, the continuum-subtracted visibilities are resampled onto "
            "a channel grid fixed in this frame, as CASA mstransform does with regridms=True. The continuum is always fitted on the native "
            "topocentric grid first, so the fit sees the bandpass structure where it is stationary. Requires --output-ms, since the output "
            "channel grid differs from the input. Leave unset to skip Doppler correction."
        ),
    ),
    doppler_chan_grid: str = Field(
        "auto",
        description=(
            "Output channel grid for the Doppler correction. 'auto' derives the grid that every timestamp of this observation covers. "
            "Otherwise give 'nchan,chan0,chanwidth' with frequency units, e.g. '1000,1419.5MHz,26.1kHz'; use this to place several MSs on "
            "one common grid, since 'auto' only ever sees a single MS."
        ),
    ),
    doppler_interpolation: INTERPOLATIONS = Field(
        "nearest",
        description=(
            "Interpolation used when resampling onto the Doppler-corrected grid. 'nearest' is what caracal asks of CASA mstransform and "
            "leaves channel noise uncorrelated; 'linear' is smoother but correlates adjacent channels."
        ),
    ),
    doppler_source_vel: float | None = Field(
        None,
        description=(
            "Systemic radial velocity of the source in km/s, positive for recession. Only used with --doppler-frame=source; when unset it "
            "is read from the MS SOURCE::SYSVEL column."
        ),
    ),
) -> None:
    opts = SimpleNamespace(**locals())
    return runit(opts)


#: Uniform handle for this module's pystep, so the StepRef can be looked up
#: generically without knowing the function's own name.
step = vis_mowjsub

command = make_command(vis_mowjsub, positional="ms", must_exist=("ms",))
