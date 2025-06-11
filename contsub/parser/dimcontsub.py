import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
from contsub.imcontsub import FitBSpline, ContSub
import astropy.io.fits as fitsio
from contsub.utils import zds_from_fits, get_automask
import dask.array as da
import time
import numpy as np
import dask.multiprocessing

log = init_logger(BIN.im_plane)

command = BIN.im_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]


@click.command("dimcontsub")
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    start_time = time.time()
 
    opts = OmegaConf.create(kwargs)
    infits = File(opts.input_image)
    
    if opts.output_prefix:
        prefix = opts.output_prefix
    else:
        prefix = f"{infits.BASEPATH}-contsub"
    
    outcont = File(f"{prefix}-cont.fits")
    outline = File(f"{prefix}-line.fits")
    if opts.overwrite is False and (outcont.EXISTS or outline.EXISTS):
        raise RuntimeError("At least one output file exists, but --no-overwrite has been set. Unset it to proceed.")
    
    if not infits.EXISTS:
        raise FileNotFoundError(f"Input FITS image could not be found at: {infits.PATH}")
    
    chunks = dict(ra = opts.ra_chunks or 64, dec=None, spectral=None)
    

    zds = zds_from_fits(infits, chunks=chunks)
    base_dims = ["ra", "dec", "spectral", "stokes"]
    if not hasattr(zds, "stokes"):
        base_dims.remove("stokes")
    
    dims_string = "ra,dec,spectral"
    has_stokes = "stokes" in base_dims
    stokes_idx = opts.stokes_index
    if has_stokes:
        cube = zds.DATA[...,stokes_idx]
    else:
        cube = zds.DATA
    
    niter = 1
    nomask = True 
    filemask = False
    if getattr(opts, "mask_image", None):
        mask = zds_from_fits(opts.mask_image, chunks=chunks).DATA
        filemask = True
        nomask = False
            
    
    get_mask = da.gufunc(
        get_automask,
        signature=f"(spectral),({dims_string}),(),(),() -> ({dims_string})",
        meta=(np.ndarray((), cube.dtype),),
        allow_rechunk=True,
    )

    signature = f"(spectral),({dims_string}),({dims_string}) -> (spectral,dec,ra),(spectral,dec,ra)"
    meta = np.ndarray((), cube.dtype), np.ndarray((), cube.dtype)
    xspec = zds.FREQS.data
    
    dask.config.set(scheduler='threads', num_workers = opts.nworkers)
    
    prev_sclip = opts.sigma_clip[0]
    sigma_clip = list(opts.sigma_clip)
    for iter_i in range(niter):
        futures = []
        fitfunc = FitBSpline(opts.order[iter_i], opts.segments[iter_i])
        
        if nomask: 
            try:
                sclip = sigma_clip[iter_i]
            except IndexError:
                sclip = prev_sclip
            finally:
                prev_sclip = sclip
        
        for biter,dblock in enumerate(cube.data.blocks):
            if nomask and opts.automask:
                mask_future = get_mask(xspec,
                                    dblock,
                                    sclip, 
                                    opts.order[iter_i],
                                    opts.segments[iter_i],
                )
                nomask = False
            elif nomask is False:
                mask_future = mask.data.blocks[biter]
            else:
                mask_future = da.zeros_like(dblock, dtype=bool)
                
            
            contfit = ContSub(fitfunc, nomask=False, reshape=False, fitsaxes=False)
            getfit = da.gufunc(
                contfit.fitContinuum,
                signature=signature,
                meta=meta,
                allow_rechunk=True,
            )
            
            futures.append(getfit(
                xspec,
                dblock,
                mask_future,
            ))
        
        results = da.compute(futures)
        
        continuum = np.concatenate(
            [ results[0][chunk_][0] for chunk_ in range(biter+1)],
        ).transpose((2,1,0))
        
        
        line = np.concatenate(
            [ results[0][chunk_][1] for chunk_ in range(biter+1)],
        ).transpose((2,1,0))
        
    header = zds.attrs["header"]
    
    log.info("Writing outputs") 
    
    if has_stokes:
        continuum = continuum[...,np.newaxis]
        line = line[...,np.newaxis]
    
    fitsio.writeto(outcont, continuum, header, overwrite=opts.overwrite)
    fitsio.writeto(outline, line, header, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    log.info(f"Finished. Runtime {hours} hours and {mins:.2f} minutes")
