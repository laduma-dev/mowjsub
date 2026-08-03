from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import dask
import dask.array as da
import shinobi
from daskms import xds_from_ms, xds_to_table
from pydantic import Field
from tqdm.dask import TqdmCallback

from mowjsub import BIN, set_logger
from mowjsub.parser._cli import make_command
from mowjsub.utils import (
    doppler_regrid_dataset,
    finalise_regridded_ms,
    get_ds_from_msdsl,
)

app = BIN.doppler_plane

FRAMES = Literal["topo", "geo", "bary", "lsrk", "lsrd", "galacto", "lgroup", "cmb", "source"]
INTERPOLATIONS = Literal["nearest", "linear"]


def runit(opts):
    """Doppler-correct an MS whose continuum has already been subtracted.

    This is the standalone half of what ``vis-mowjsub --doppler-frame`` does in
    one pass. Splitting it out lets a pipeline run continuum subtraction and the
    frame conversion as separate stages while keeping them in the right order:
    the continuum must be fitted on the native topocentric grid, where the
    bandpass structure it models is stationary.
    """
    start_time = time.time()

    log = set_logger()
    ms = opts.ms
    spwid = opts.spwid
    fieldid = opts.field_id
    chunksize = opts.row_chunks
    input_column = opts.input_column
    output_column = opts.output_column
    frame = opts.doppler_frame

    ms_dsl = xds_from_ms(
        ms,
        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": chunksize},
    )

    msds = get_ds_from_msdsl(ms_dsl, field_id=fieldid, data_desc_id=spwid)

    if input_column not in msds.data_vars:
        available = ", ".join(sorted(name for name, var in msds.data_vars.items() if "chan" in var.dims))
        raise RuntimeError(f"Column '{input_column}' is not in {ms}. Per-channel columns it does have: {available or 'none'}.")

    dask.config.set(scheduler="threads", num_workers=opts.nworkers)

    source_vel = opts.doppler_source_vel
    regrid_ds, freqs_out, chanwidth = doppler_regrid_dataset(
        msds,
        ms,
        getattr(msds, input_column).data,
        output_column,
        spwid,
        fieldid,
        frame=frame,
        chan_grid=opts.doppler_chan_grid,
        interpolation=opts.doppler_interpolation,
        source_vel=None if source_vel is None else source_vel * 1e3,
    )

    ms_name = str(opts.output_ms)
    writes = [xds_to_table([regrid_ds], ms_name, columns="ALL")]
    with TqdmCallback(desc="Writing Doppler-corrected data"):
        da.compute(writes)

    finalise_regridded_ms(ms, ms_name, spwid, freqs_out, chanwidth, frame)
    log.info(f"Doppler correction completed. Data from '{input_column}' written to column '{output_column}' in {ms_name}.")

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime / 3600)
    mins = dtime / 60 - hours * 60
    secs = (mins % 1) * 60
    log.info(f"Runtime {hours}:{int(mins)}:{secs:.1f}")


@shinobi.pystep(
    name=app,
    info="Doppler-correct an already continuum-subtracted Measurement Set (MS) onto a channel grid fixed in a chosen spectral frame",
)
def doppler_mowjsub(
    ms: Path = Field(..., description="Input MS file. Its continuum must already have been subtracted; this command does no fitting."),
    input_column: str = Field(
        ...,
        description=(
            "Column holding the continuum-subtracted data to Doppler-correct. Required: there is no standard MS column name for "
            "continuum-subtracted visibilities, so this is whatever the subtraction stage was told to write."
        ),
    ),
    output_column: str = Field(
        ...,
        description="Column name to write the Doppler-corrected data to in the output MS. Pass DATA if the imager you feed it to expects that column.",
    ),
    output_ms: Path = Field(
        ...,
        description=("Name of the MS to write. Required, since the Doppler correction changes the channel count and so cannot be written back into a column of the input MS."),
    ),
    spwid: int = Field(0, description="Spectral Window ID"),
    field_id: int = Field(0, description="Field ID"),
    row_chunks: int = Field(10000, description="Chunking strategy (Done along the time axis)"),
    nworkers: int = Field(4, description="Number of parallel worker threads (roughly one per CPU core)."),
    doppler_frame: FRAMES = Field(
        ...,
        description=(
            "Spectral reference frame to Doppler-correct the output to. The visibilities are resampled onto a channel grid fixed in this "
            "frame, as CASA mstransform does with regridms=True. The input channel grid must still be topocentric, i.e. as it came off the "
            "telescope."
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
step = doppler_mowjsub

command = make_command(doppler_mowjsub, positional="ms")
