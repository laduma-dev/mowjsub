import glob
import os
import time

import click
import dask
import dask.array as da
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from scabha import init_logger
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

import mowjsub
from mowjsub import BIN
from mowjsub.fitfuncs import (
    FitBSpline,
    FitGCVSpline,
    FitMedFilter,
    FitMedFilterFast,
    FitPolynomial,
)
from mowjsub.image_plane import ContSub
from mowjsub.utils import (
    apply_cube_doppler,
    get_automask,
    plan_cube_doppler,
    write_cubes,
    zds_from_fits,
)

command = BIN.im_plane
thisdir = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/im_mowjsub.yaml")
config = paramfile_loader(parserfile, sources)["im_mowjsub"]

log = init_logger(BIN.im_plane)


@click.command("im-mowjsub")
@click.version_option(str(mowjsub.__version__))
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

    if opts.fit_model in ("b-spline", "spline", "polynomial") and getattr(opts, "order", None) is None:
        raise RuntimeError(f"The parameter 'order' is required for fit-model={opts.fit_model}.")

    velwidth = opts.vel_width or opts.segments

    if opts.fit_model in ("b-spline", "spline", "median-filter", "scipy-median-filter") and velwidth is None:
        raise RuntimeError(f"The parameter 'vel-width' (or legacy 'segments') is required for fit-model={opts.fit_model}.")

    if opts.ra_chunks < 0:
        raise RuntimeError("The parameter 'ra-chunks' cannot be negative. Set it to zero to disable chunking.")

    ra_chunks = opts.ra_chunks
    # Zero disables chunking, i.e. the RA axis is read as a single block.
    chunks = dict(ra=ra_chunks or -1, dec=None, spectral=None)

    rest_freq = opts.rest_freq
    zds = zds_from_fits(
        infits.PATH,
        chunks=chunks,
        rest_freq=rest_freq,
        hdu_idx=opts.hdu_index,
        add_freqs=True,
    )
    base_dims = list(zds.DATA.dims)

    dims_string = "ra,dec,spectral"
    has_stokes = "stokes" in base_dims
    stokes_idx = opts.stokes_index

    log.info(f"Input data dimensions: {zds.DATA.dims}")
    log.info(f"Input data shape: {zds.DATA.shape}")

    if has_stokes:
        cube = zds.DATA[..., stokes_idx]
    else:
        cube = zds.DATA

    nomask = True
    if getattr(opts, "mask_image", None):
        mask = zds_from_fits(opts.mask_image, chunks=chunks, rest_freq=rest_freq).DATA
        nomask = False

    signature = f"({dims_string}),({dims_string}) -> ({dims_string})"
    meta = (np.ndarray((), cube.dtype),)
    # FREQS is a small in-memory grid, not a dask array: it used to be one only
    # because the whole Dataset got chunked on the way out.
    xspec = np.asarray(zds.FREQS.data)

    dask.config.set(scheduler="threads", num_workers=opts.nworkers)
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
    elif opts.fit_model == "scipy-median-filter":
        fitfunc = FitMedFilterFast(xspec, velwidth=velwidth, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare()
    elif opts.fit_model == "gcv-spline":
        fitfunc = FitGCVSpline(xspec, fit_lam=opts.gcv_lambda, fit_tol=opts.cont_fit_tol)
        fitfunc.prepare()
    else:
        raise RuntimeError(f"Unsupported fit-model: {opts.fit_model!r}.")

    get_mask = da.gufunc(
        lambda _data: get_automask(_data, fitfunc, opts.sigma_clip),
        signature=f"({dims_string}) -> ({dims_string})",
        meta=(np.ndarray((), cube.dtype),),
        allow_rechunk=True,
    )

    for biter, dblock in enumerate(dblocks):
        if opts.sigma_clip:
            mask_future = get_mask(dblock)
        elif nomask is False:
            mask_future = mask.data.blocks[biter]
        else:
            mask_future = da.zeros_like(dblock, dtype=bool)

        contfit = ContSub(fitfunc, nomask=False)

        getfit = da.gufunc(
            contfit.fitContinuum,
            signature=signature,
            meta=meta,
            allow_rechunk=True,
        )

        futures.append(
            getfit(
                dblock,
                mask_future,
            )
        )

    # Back into the order the file itself uses. This was a fixed
    # `.transpose((2, 1, 0))` plus a `[np.newaxis]`, which assumed spectral was
    # third and Stokes outermost; xarray transposes by name, so the same code
    # now writes a RA, DEC, FREQ, STOKES cube and a RA, DEC, STOKES, FREQ one
    # into their own layouts.
    continuum = xr.DataArray(da.concatenate(futures), dims=("ra", "dec", "spectral"))
    if has_stokes:
        continuum = continuum.expand_dims("stokes")
    continuum = continuum.transpose(*zds.attrs["fits_dims"]).data

    header = zds.attrs["header"]

    # The residual is an array operation now, on the cube already in hand and in
    # the file's own axis order. It used to be formed by writing the continuum
    # out and reading it straight back alongside the input -- which is also why
    # the Doppler path needed a `{prefix}-cont-topo.fits` scratch file, since the
    # continuum had to exist on disk before the residual could be taken.
    line = zds.DATA.transpose(*zds.attrs["fits_dims"]).data - continuum

    if not opts.doppler_frame:
        outputs = [(outcont.PATH, continuum, header), (outline.PATH, line, header)]
    else:
        source_vel = opts.doppler_source_vel
        plan = plan_cube_doppler(
            header,
            opts.doppler_frame,
            chan_grid=opts.doppler_chan_grid,
            obs_time=opts.doppler_time,
            obs_duration=opts.doppler_obs_duration,
            telescope=opts.doppler_telescope,
            phase_centre=opts.doppler_phase_centre,
            source_vel=None if source_vel is None else source_vel * 1e3,
        )

        # One plan for both cubes, so the pair stays on a single grid and
        # remains recombinable.
        outputs = [
            (outcont.PATH, *apply_cube_doppler(continuum, header, plan, opts.doppler_interpolation)),
            (outline.PATH, *apply_cube_doppler(line, header, plan, opts.doppler_interpolation)),
        ]

    log.info(f"Writing continuum model cube to: {outcont}")
    log.info(f"Writing residual data (line cube) to: {outline}")
    # Both cubes in one pass: they share the per-pixel fit, and a writeto each
    # would make dask walk that graph twice and refit every spectrum.
    write_cubes(outputs, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime / 3600)
    mins = dtime / 60 - hours * 60
    secs = (mins % 1) * 60
    log.info(f"Finished. Runtime {hours}:{int(mins)}:{secs:.1f}")
