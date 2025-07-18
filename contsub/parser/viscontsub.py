import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
import dask.array as da
import time
import numpy as np
import xarray as xr
import dask.multiprocessing
from contsub.utils import ms_to_xarray_dataset
from contsub.visibility_plane import VisContSub
from contsub.fitfuncs import FitBSpline
from dask.diagnostics import ProgressBar
from tqdm.dask import TqdmCallback
from daskms import xds_from_ms, xds_to_table

log = init_logger(BIN.vis_plane)

command = BIN.vis_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

@click.command("viscontsub")
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):    
    start_time = time.time()
    
    opts = OmegaConf.create(kwargs)
    ms = opts.ms
    spwid = opts.spwid
    fieldid = opts.field_id
    chunksize = opts.row_chunks
    segments = opts.segments[0]
    method = opts.fit_model
    order = opts.order[0]
    output_prefix = opts.output
    nworkers = opts.nworkers
    outchunks = dict(time=opts.time_chunks, bl_chunks=opts.bl_chunks)
    output_column = opts.output_column

    temp_zarr = ms_to_xarray_dataset(ms, spwid, fieldid, chunksize, save_to_zarr=True)
    ds = xr.open_zarr(temp_zarr, chunks=outchunks)
    
    xspec = ds.coords['spectral']

    futures = []
    
    if method == 'spline':
        fitfunc = FitBSpline(order, segments, randomState=None, seq=None)
    else:
        raise ValueError(f"Unknown fitting method: {method}. Supported methods: 'spline'.")
    base_dims = "time,baseline,spectral,corr"
    signature = f"(spectral),({base_dims}),({base_dims}),({base_dims}) -> ({base_dims})"
    meta = (np.ndarray((), ds.vis.dtype),)

    dask.config.set(scheduler='threads', num_workers = nworkers)

    for biter, dblock in enumerate(ds.vis.data.blocks):
        #if biter > 0:
            #continue
        flags = ds.flags.data.blocks[biter]
        weights = ds.weights.data.blocks[biter]

        contfit = VisContSub(fitfunc, fit_tol=opts.cont_fit_tol)
        get_cont = da.gufunc(
                contfit.vis_cont_sub,
                signature=signature,
                meta=meta,
                allow_rechunk=True,
            )
        futures.append(get_cont(
            xspec,
            dblock,
            flags,
            weights,
            ),
        )
    
        
    continuum_dask = da.concatenate(futures)
    
    continuum_xarray = xr.DataArray(
    data=continuum_dask,
    dims=ds.vis.dims,
    coords=ds.vis.coords
    )

    continuum = continuum_xarray.stack(row = ('time','baseline'))
    continuum = continuum.transpose("row",...)

    visdata = ds.vis.stack(row=('time', 'baseline')).transpose("row", ...)

    line = visdata - continuum
    
    ms_dsl = xds_from_ms(
        ms, 
        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
        chunks={"row": chunksize } 
    )
    
    #debugging_prints
    #print(f"Total MS rows: {sum(ds_chunk.sizes['row'] for ds_chunk in ms_dsl)}")
    #print(f"Line data shape: {line.shape}")
    #print(f"Line data dims: {line.dims}")
    #print(f"Line data chunks: {line.chunks}")

    #for i, ds_chunk in enumerate(ms_dsl):
        #print(f"Chunk {i}: {ds_chunk.sizes['row']} rows")
    
    writes = []
    start_row = 0

    for i, ds_chunk in enumerate(ms_dsl):
        num_rows_in_chunk = ds_chunk.sizes['row']
        end_row = start_row + num_rows_in_chunk

        line_slice = line[start_row:end_row, :, :]
        target_row_chunks = ds_chunk.chunks['row']
        line_slice_rechunked = line_slice.chunk({'row': target_row_chunks})

        ms_dsl[i] = ds_chunk.assign(**{
        output_column: (("row", "chan", "corr"), line_slice_rechunked.data)
        })

    writes.append(xds_to_table(ms_dsl, ms, [output_column]))
    print(f"Writing line data to column '{output_column}' in {ms}...")


    with TqdmCallback(desc="Writing line data to MS"):
        da.compute(*writes)
    print(f"UV plane continuum subtraction completed. Data written to column '{output_column}' in {ms}.")

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    secs = (mins%1) * 60
    log.info(f"Runtime {hours}:{int(mins)}:{secs:.1f}")
