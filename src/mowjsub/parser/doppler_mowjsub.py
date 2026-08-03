import glob
import os
import time

import click
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table
from omegaconf import OmegaConf
from scabha import init_logger
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader
from tqdm.dask import TqdmCallback

import mowjsub
from mowjsub import BIN
from mowjsub.utils import (
    doppler_regrid_dataset,
    finalise_regridded_ms,
    get_ds_from_msdsl,
)

log = init_logger(BIN.doppler_plane)

command = BIN.doppler_plane
thisdir = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/doppler_mowjsub.yaml")
config = paramfile_loader(parserfile, sources)["doppler_mowjsub"]


@click.command("doppler-mowjsub")
@click.version_option(str(mowjsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    """Doppler-correct an MS whose continuum has already been subtracted.

    This is the standalone half of what ``vis-mowjsub --doppler-frame`` does in
    one pass. Splitting it out lets a pipeline run continuum subtraction and the
    frame conversion as separate stages while keeping them in the right order:
    the continuum must be fitted on the native topocentric grid, where the
    bandpass structure it models is stationary.
    """
    start_time = time.time()

    opts = OmegaConf.create(kwargs)
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
