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
from contsub.fitfuncs import (
    FitBSpline,
    FitPolynomial,
    FitMedFilter,
    FitGCVSpline,
)
import astropy.io.fits as fitsio
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

    if opts.fit_model in "spline polynomial dct".split() and not getattr(opts, "order", False):
        raise RuntimeError("The parameter 'order' is required for fit-model={opts.fit_model}.")
    
    if opts.fit_model in "spline median-filter dct".split() and not getattr(opts, "vel_width", False):
        raise RuntimeError("The parameter 'vel-width' is required for fit-model={opts.fit_model}.")
    
    velwidth = opts.vel_width or opts.segments
    chunks = dict(ra = opts.ra_chunks or 64, dec=None, spectral=None)
    
    rest_freq = opts.rest_freq
    zds = zds_from_fits(infits.PATH, chunks=chunks, rest_freq=rest_freq, hdu_idx=opts.hdu_index, add_freqs=True)
    header = fitsio.Header(zds.header)
    base_dims = ["ra", "dec", "spectral", "stokes"]
    if not hasattr(zds, "stokes"):
        base_dims.remove("stokes")
    
    dims_string = "ra,dec,spectral"
    has_stokes = "stokes" in base_dims
    stokes_idx = opts.stokes_index
    
    log.info(f"Input data dimensions: {zds.DATA.dims}")
    log.info(f"Input data shape: {zds.DATA.shape}")
    
    if has_stokes:
        cube = zds.DATA[...,stokes_idx]
    else:
        cube = zds.DATA
    
    
    nomask = True
    if getattr(opts, "mask_image", None):
        mask = zds_from_fits(opts.mask_image, chunks=chunks, rest_freq=rest_freq).DATA
        nomask = False
    

    signature = f"({dims_string}),({dims_string}) -> ({dims_string})"
    meta = (np.ndarray((), cube.dtype),)
    xspec = zds.FREQS.data.compute()
    
    dask.config.set(scheduler='threads', num_workers = opts.nworkers)
    dblocks = cube.data.blocks
    futures = []
    
    if opts.fit_model in ["spline", "b-spline"]:
        fitfunc = FitBSpline(xspec, order=opts.order, velwidth=velwidth, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare()
    elif opts.fit_model == "polynomial":
        fitfunc = FitPolynomial(xspec, order=opts.order, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare()
    elif opts.fit_model == "median-filter":
        fitfunc = FitMedFilter(xspec, velwidth=velwidth, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare()
    elif opts.fit_model == "gcv-spline":
        fitfunc = FitGCVSpline(xspec, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare(lam=opts.gcv_lambda)
        
    get_mask = da.gufunc(
        lambda _data: get_automask(_data, fitfunc, opts.sigma_clip),
        signature=f"({dims_string}) -> ({dims_string})",
        meta=(np.ndarray((), cube.dtype),),
        allow_rechunk=True,
    )
        
    for biter,dblock in enumerate(dblocks):
        if opts.sigma_clip:
            mask_future = get_mask(dblock)
        elif nomask is False:
            mask_future = mask.data.blocks[biter]
        else:
            mask_future = da.zeros_like(dblock, dtype=bool)
        
        contfit = ContSub(fitfunc)
        
        getfit = da.gufunc(
            contfit.fitContinuum,
            signature=signature,
            meta=meta,
            allow_rechunk=True,
        )
        
        futures.append(getfit(
            dblock,
            mask_future,
        ))
        
    continuum = da.concatenate(futures).transpose((2,1,0))
    if has_stokes:
        continuum = continuum[np.newaxis,...]
    
    out_ds_cont = fitsio.PrimaryHDU(continuum, header=header)
    
    out_ds_cont.writeto(outcont.PATH, overwrite=opts.overwrite)
    log.info(f"Continuum model cube written to: {outcont}")
    
    out_ds_line = subtract_fits(infits.PATH, outcont.PATH, chunks={0: opts.ra_chunks, 1:None, 2:None})
    log.info(f"Writing residual data (line cube) to: {outline}")
    out_ds_line.writeto(outline.PATH, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    secs = (mins%1) * 60
    log.info(f"Finished. Runtime {hours}:{int(mins)}:{secs:.1f}")
