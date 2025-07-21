import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
from contsub.image_plane import ContSub
from contsub.fitfuncs import FitBSpline
import astropy.io.fits as fitsio
from astropy.wcs import WCS
from contsub.utils import zds_from_fits, get_automask, subtract_fits
import dask.array as da
import time
import numpy as np
import dask.multiprocessing


command = BIN.im_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

log = init_logger(BIN.im_plane)

@click.command("imcontsub")
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):    
    start_time = time.time()
    
    opts = OmegaConf.create(kwargs)
    
    if opts.cont_fit_tol > 100:
        log.warning("Requested --cont-fit-tol is larger than 100 percent. Assuming it is 100.")
        opts.cont_fit_tol = 100
        
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
    
    ra_chunks = opts.ra_chunks or 64
    chan_chunks = opts.chan_chunks or 64
    
    wcs = WCS(infits)
    
    nchan = zds.shape[2]
    zds, header = zds_from_fits(opts.mask_image, stokes=opts.stokes_index,
                                ra_chunks=ra_chunks, chan_chunks=chan_chunks)
    
    dims_string = "ra,dec,spectral"
    has_stokes = zds.attrs["has_stokes"]
    
    if len(opts.order) != len(opts.segments):
        raise ValueError("If setting multiple --order and --segments, they must be of equal length. "
                        f"Got {len(opts.order)} orders and {len(opts.segments)} segments.")
    
    niter = len(opts.order)
    
    nomask = True
    automask = False 
    if getattr(opts, "mask_image", None):
        mask = zds_from_fits(opts.mask_image, stokes=opts.stokes_index,
                    ra_chunks=ra_chunks, chan_chunks=chan_chunks)
        nomask = False
    
    if getattr(opts, "sigma_clip", None):
        automask = True
        sigma_clip = list(opts.sigma_clip)
    else:
        sigma_clip = []
    
    if len(sigma_clip) < niter and automask:
        log.warning(f"Only {len(sigma_clip)} sigma-clips provided, but {niter} iterations requested."
                    " Using last value for unspecified iterations.")
        sigma_clip.extend([sigma_clip[-1]] * (niter - len(sigma_clip)))
    
    get_mask = da.gufunc(
        get_automask,
        signature=f"(spectral),({dims_string}),(),(),() -> ({dims_string})",
        meta=(np.ndarray((), zds.dtype),),
        allow_rechunk=True,
    )

    signature = f"(spectral),({dims_string}),({dims_string}) -> ({dims_string})"
    meta = (np.ndarray((), zds.dtype),)
    xspec = wcs.array_index_to_world_values(np.arange(nchan))
    
    dask.config.set(scheduler='threads', num_workers = opts.nworkers)
    dblocks = zds.blocks
    for iter_i in range(niter):
        log.info(f"Loading delayed compute for iteration {iter_i+1}/{niter} of continuum modelling.")
        futures = []
        fitfunc = FitBSpline(opts.order[iter_i], opts.segments[iter_i])
        for biter,dblock in enumerate(dblocks):
            if (nomask and automask) or (iter_i > 0 and automask):
                mask_future = get_mask(xspec,
                                    dblock,
                                    sigma_clip[iter_i], 
                                    opts.order[iter_i],
                                    opts.segments[iter_i],
                )
            elif nomask is False:
                mask_future = mask.data.blocks[biter] == False
            else:
                mask_future = da.ones_like(dblock, dtype=bool)
            
            contfit = ContSub(fitfunc, nomask=False,
                            fit_tol=opts.cont_fit_tol)
            
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
            
        dblocks = list(futures)
    
    continuum = da.concatenate(futures).transpose((2,1,0))
    if has_stokes:
        continuum = continuum[np.newaxis,...]
    
    out_ds_cont = fitsio.PrimaryHDU(continuum, header=header)
    
    out_ds_cont.writeto(outcont, overwrite=opts.overwrite)
    log.info(f"Continuum model cube written to: {outcont}")
    
    out_ds_line = subtract_fits(infits, outcont, chunks={0: opts.ra_chunks, 1:None, 2:None})
    log.info(f"Writing residual data (line cube) to: {outline}")
    out_ds_line.writeto(outline, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    secs = (mins%1) * 60
    log.info(f"Finished. Runtime {hours}:{int(mins)}:{secs:.1f}")
