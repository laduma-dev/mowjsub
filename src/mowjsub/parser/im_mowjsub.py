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
from pydantic import Field

from mowjsub import BIN, set_logger
from mowjsub.fitfuncs import (
    ORDER_MODELS,
    WIDTH_MODELS,
    build_fitfunc,
)
from mowjsub.image_plane import ContSub
from mowjsub.parser._cli import make_command
from mowjsub.sources import beam_element, footprint_from_catalogue, footprint_from_mask, read_catalogue
from mowjsub.utils import (
    apply_cube_doppler,
    get_automask,
    plan_cube_doppler,
    write_cubes,
    zds_from_fits,
)

app = BIN.im_plane

FIT_MODELS = Literal["b-spline", "spline", "polynomial", "median-filter", "scipy-median-filter", "gcv-spline"]
FRAMES = Literal["topo", "geo", "bary", "lsrk", "lsrd", "galacto", "lgroup", "cmb", "source"]
INTERPOLATIONS = Literal["nearest", "linear"]
LOG_LEVELS = Literal["info", "debug", "trace", "error", "critical"]


def _automask_block(
    data,
    spec,
    seed,
    sigma_clip,
    source_footprint=None,
    source_sigma_clip=None,
    max_masked_fraction=None,
    min_channels=1,
    sky_element=None,
    source_min_channels=None,
    source_sky_element=None,
):
    """Build the automask for one chunk, on a fitter of the chunk's own.

    ``source_footprint`` is this chunk's slice of the sky footprint, already cut
    to the block's RA extent by the caller -- chunking is along RA only, so the
    slice is exact and no overlap between chunks is needed. ``sky_element``
    arrives already resolved from beams, since a task cannot see the header.
    """
    return get_automask(
        data,
        build_fitfunc(spec, seed),
        sigma_clip,
        source_footprint=source_footprint,
        source_sigma_clip=source_sigma_clip,
        max_masked_fraction=max_masked_fraction,
        min_channels=min_channels,
        sky_element=sky_element,
        source_min_channels=source_min_channels,
        source_sky_element=source_sky_element,
    )


def _fit_block(data, mask, spec, seed):
    """Fit the continuum of one chunk, on a fitter of the chunk's own."""
    return ContSub(build_fitfunc(spec, seed), nomask=False).fitContinuum(data, mask)


def _resolve_footprint(opts, cube, header, log):
    """Turn --source-catalogue / --source-mask into a sky footprint, or None.

    Returns None when the deeper threshold would be inert, so the caller can
    skip the second clip entirely rather than compute a mask it will not use.
    """
    if not opts.source_catalogue and not opts.source_mask:
        return None

    if opts.source_catalogue and opts.source_mask:
        raise RuntimeError("--source-catalogue and --source-mask both name the absorption search footprint. Give one, not both.")

    if opts.source_sigma_clip is None:
        raise RuntimeError("--source-catalogue/--source-mask set where a deeper threshold applies, but --source-sigma-clip does not say what it is.")

    if not opts.sigma_clip:
        raise RuntimeError("The source footprint refines the automatic mask, so it needs --sigma-clip. With --mask-image the mask is yours already.")

    if opts.source_sigma_clip >= opts.sigma_clip:
        raise RuntimeError(
            f"--source-sigma-clip={opts.source_sigma_clip} is not below --sigma-clip={opts.sigma_clip}, so the footprint could only "
            "mask less than the blind clip already does. It exists to go deeper where a background source makes absorption possible."
        )

    shape = cube.shape[:2]
    if opts.source_mask:
        footprint = footprint_from_mask(opts.source_mask, shape)
    else:
        ra, dec, maj = read_catalogue(opts.source_catalogue)
        footprint = footprint_from_catalogue(header, ra, dec, maj, radius_pix=opts.source_radius, shape=shape)

    if not footprint.any():
        log.warning("The source footprint is empty; the deeper threshold will not be applied anywhere.")
        return None

    log.info(
        f"Absorption search footprint covers {footprint.mean() * 100:.2f}% of the sky area, clipped at {opts.source_sigma_clip} sigma there against {opts.sigma_clip} elsewhere"
    )
    return footprint


def runit(opts):
    start_time = time.time()

    log = set_logger(opts.loglevel)

    if opts.cont_fit_tol > 100:
        log.warning("Requested --cont-fit-tol is larger than 100 percent. Assuming it is 100.")
        opts.cont_fit_tol = 100

    infits = Path(opts.input_image)

    if opts.output_prefix:
        prefix = str(opts.output_prefix)
    else:
        prefix = f"{infits.with_suffix('')}-contsub"

    outcont = Path(f"{prefix}-cont.fits")
    outline = Path(f"{prefix}-line.fits")

    if opts.overwrite is False and (outcont.exists() or outline.exists()):
        raise RuntimeError("At least one output file exists, but --no-overwrite has been set. Unset it to proceed.")

    if opts.fit_model in ORDER_MODELS and opts.order is None:
        raise RuntimeError(f"The parameter 'order' is required for fit-model={opts.fit_model}.")

    velwidth = opts.vel_width or opts.segments
    chanwidth = opts.chan_width

    if velwidth is not None and chanwidth is not None:
        raise RuntimeError("--vel-width and --chan-width both set the width of the fit window, in different units. Give one, not both.")

    if chanwidth is not None and chanwidth < 1:
        raise RuntimeError(f"--chan-width={chanwidth} is not a channel count. It must be at least 1.")

    if opts.fit_model in WIDTH_MODELS and velwidth is None and chanwidth is None:
        raise RuntimeError(f"The parameter 'vel-width' (or legacy 'segments') or 'chan-width' is required for fit-model={opts.fit_model}.")

    if opts.ra_chunks < 0:
        raise RuntimeError("The parameter 'ra-chunks' cannot be negative. Set it to zero to disable chunking.")

    ra_chunks = opts.ra_chunks
    # Zero disables chunking, i.e. the RA axis is read as a single block.
    chunks = dict(ra=ra_chunks or -1, dec=None, spectral=None)

    rest_freq = opts.rest_freq
    zds = zds_from_fits(
        str(infits),
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
    if opts.mask_image:
        mask = zds_from_fits(str(opts.mask_image), chunks=chunks, rest_freq=rest_freq).DATA
        nomask = False

    signature = f"({dims_string}),({dims_string}) -> ({dims_string})"
    meta = (np.ndarray((), cube.dtype),)
    # FREQS is a small in-memory grid, not a dask array: it used to be one only
    # because the whole Dataset got chunked on the way out.
    xspec = np.asarray(zds.FREQS.data)

    dask.config.set(scheduler="threads", num_workers=opts.nworkers)
    dblocks = cube.data.blocks
    futures = []

    spec = dict(
        fit_model=opts.fit_model,
        xspec=xspec,
        order=opts.order,
        velwidth=velwidth,
        chanwidth=chanwidth,
        fit_tol=opts.cont_fit_tol,
        gcv_lambda=opts.gcv_lambda,
    )
    # Validate the configuration once, here, rather than letting a bad
    # combination surface from inside a worker once the cube is already being
    # read. Discarded -- the fitters the tasks use are their own.
    build_fitfunc(spec)

    # One fitter per chunk, constructed inside the task. A single instance used
    # to be shared by every worker, and `FitBSpline` draws its knot offsets from
    # `self.rng` -- a numpy Generator, which is explicitly not thread-safe --
    # while the scheduler below runs those workers as threads. Independent child
    # seeds also stop the chunks drawing from one correlated stream.
    #
    # Two seeds per chunk, not one. The shared fitter served both passes, so the
    # automask fit advanced the RNG and the final fit placed *different* knots.
    # Giving both passes one seed would make them place the same knots, which is
    # arguably more coherent, but it is a change in results (measured at 0.47
    # sigma RMS, 2.9 sigma peak on a real cube) and does not belong in a fix
    # that is otherwise result-preserving.
    nblocks = cube.data.numblocks[0]
    seeds = np.random.SeedSequence().generate_state(2 * nblocks)
    mask_seeds = [int(seed) for seed in seeds[:nblocks]]
    fit_seeds = [int(seed) for seed in seeds[nblocks:]]

    footprint = _resolve_footprint(opts, cube, zds.attrs["header"], log)

    # Resolved here, not in the task: the sky-extent cut is stated in beams and
    # only the header knows how many pixels that is.
    sky_element = beam_element(zds.attrs["header"], opts.min_sky_beams)
    source_sky_element = sky_element if opts.source_min_sky_beams is None else beam_element(zds.attrs["header"], opts.source_min_sky_beams)
    # Chunking is along RA only, so each block's slice of the sky footprint is
    # exactly the rows that block holds -- no overlap between chunks is needed,
    # unlike a spatial filter, which is one reason to prefer a per-sightline
    # threshold to a smoothing kernel here.
    ra_bounds = np.cumsum((0,) + cube.data.chunks[0])

    for biter, dblock in enumerate(dblocks):
        if opts.sigma_clip:
            block_footprint = None if footprint is None else footprint[ra_bounds[biter] : ra_bounds[biter + 1]]
            get_mask = da.gufunc(
                partial(
                    _automask_block,
                    spec=spec,
                    seed=mask_seeds[biter],
                    sigma_clip=opts.sigma_clip,
                    source_footprint=block_footprint,
                    source_sigma_clip=opts.source_sigma_clip,
                    max_masked_fraction=opts.max_masked_fraction,
                    min_channels=opts.min_channels,
                    sky_element=sky_element,
                    source_min_channels=opts.source_min_channels,
                    source_sky_element=source_sky_element,
                ),
                signature=f"({dims_string}) -> ({dims_string})",
                meta=(np.ndarray((), cube.dtype),),
                allow_rechunk=True,
            )
            mask_future = get_mask(dblock)
        elif nomask is False:
            mask_future = mask.data.blocks[biter]
        else:
            mask_future = da.zeros_like(dblock, dtype=bool)

        getfit = da.gufunc(
            partial(_fit_block, spec=spec, seed=fit_seeds[biter]),
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
        outputs = [(str(outcont), continuum, header), (str(outline), line, header)]
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
            (str(outcont), *apply_cube_doppler(continuum, header, plan, opts.doppler_interpolation)),
            (str(outline), *apply_cube_doppler(line, header, plan, opts.doppler_interpolation)),
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


@shinobi.pystep(name=app, info="Perform image plane continuum subtraction on an input FITS file")
def im_mowjsub(
    input_image: Path = Field(..., description="Input image"),
    output_prefix: str | None = Field(None, description="Name of ouput image"),
    mask_image: Path | None = Field(None, description="Mask image"),
    sigma_clip: float | None = Field(
        None,
        description=(
            "Sigma-clipping level for the automatic mask, which is built by fitting an unmasked continuum first and clipping that residual -- "
            "so it works on a cube that still holds its continuum. Only used if --mask-image is not given."
        ),
    ),
    source_catalogue: Path | None = Field(
        None,
        description=(
            "Catalogue of continuum sources, marking the sightlines where absorption is physically possible -- it cannot occur "
            "without something behind it to absorb. Those sightlines then carry the deeper --source-sigma-clip while the rest of "
            "the cube keeps --sigma-clip, which is affordable because the search volume there is a small fraction of the cube and "
            "so is the number of trials the threshold has to survive. Reads PyBDSF and SoFiA catalogues, or a plain 'ra dec "
            "[maj_deg]' list in degrees. Run the finder over the full-band continuum image, not this cube."
        ),
    ),
    source_mask: Path | None = Field(
        None,
        description="2D FITS mask naming the same footprint as --source-catalogue, for a footprint built by something whose catalogue format is not readable here.",
    ),
    source_sigma_clip: float | None = Field(
        None,
        description=(
            "Sigma-clipping level applied on the sightlines --source-catalogue/--source-mask name. Must be below --sigma-clip: the "
            "footprint exists to look DEEPER where a background source makes absorption possible. Measured on synthetic cubes, 2.0 "
            "against a blind 3.0 recovered 68% of a shallow absorber's depth where the blind clip alone got 33%, and gained 5-10 "
            "points at higher SNR. Below about 1.5 the fit starts to fail outright, so pair it with --max-masked-fraction."
        ),
    ),
    source_radius: float | None = Field(
        None,
        description="Footprint radius in pixels for catalogue entries carrying no major axis. Defaults to 3. A source's own catalogued extent is used where there is one.",
    ),
    min_channels: int = Field(
        1,
        description=(
            "Channels a detection must span to survive into the automatic mask. Detections are grown by two 18-neighbour passes to "
            "catch the line wings that sit below any threshold their peak clears -- worth up to 22 points of recovered line flux -- "
            "but the growth reaches noise as readily as signal. A cube's noise is correlated across the beam and independent "
            "between channels, so noise components are spatially extended but spectrally thin: measured on beam-correlated noise "
            "they span a median of 1 channel and never more than 2, while a 3-sigma resolved line spans 3. This is the axis that "
            "separates them, which a voxel count cannot, since both occupy about a beam: at 3 sigma the cut takes the excluded "
            "fraction of a beam-correlated noise residual from 8.11% to 0.14%. "
            "IT IS AN ENABLER FOR A LOWER THRESHOLD, NOT A FREE IMPROVEMENT. A feature only just clearing the threshold crosses it "
            "in one or two channels, so the cut removes it along with the noise -- on a synthetic 3-sigma absorber, --min-channels 2 "
            "at --sigma-clip 3 alone recovered 68% of the depth where no cut recovered 87%. Paired with a deeper threshold that "
            "gives a real feature the extent to survive, the same cube went to 102%. So use it with --source-sigma-clip, or with a "
            "--sigma-clip you have lowered deliberately. 1, the default, leaves the mask as it has always behaved."
        ),
    ),
    min_sky_beams: float = Field(
        0.0,
        description=(
            "Sky extent a detection must cover, as a linear size in beam major axes -- so 0.5 means half a beam across. Fractions "
            "are allowed and the element is rounded UP, so any positive value still selects something. The test is whether the "
            "detection CONTAINS a region of that shape, not whether it has that many pixels: a count is orientation-blind and "
            "would pass an elongated streak with a beam's worth of pixels in it. All three of BMAJ, BMIN and BPA are used, so "
            "the element is an ellipse carrying the beam's axis ratio and position angle. Stated in beams rather than pixels "
            "because a pixel means something different on every cube, while the beam is the scale below which nothing on the "
            "sky can be real. Needs a beam in the header. 0 disables it."
        ),
    ),
    source_min_channels: int | None = Field(
        None,
        description=(
            "--min-channels for the deeper --source-sigma-clip inside the source footprint. Defaults to --min-channels. Worth "
            "setting higher: a lower threshold produces far more noise components for the growth to inflate, and with no cut at "
            "all 2 sigma excluded 96.9% of the band and the fit returned NaN."
        ),
    ),
    source_min_sky_beams: float | None = Field(
        None,
        description="--min-sky-beams for the deeper clip inside the source footprint. Defaults to --min-sky-beams.",
    ),
    max_masked_fraction: float | None = Field(
        None,
        description=(
            "Largest fraction of a spectrum that may be masked. Where a sightline exceeds it only its strongest detections are "
            "kept, ranked by residual. This is the guard on a deep --source-sigma-clip: mask most of a band and the spline fit has "
            "too little left to constrain it, which does not fail loudly -- it returns a NaN or wildly wrong continuum. Unset "
            "means no cap."
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
    cont_fit_tol: float = Field(
        0,
        description=(
            "Minimum percentage of valid spectrum data points required to do a fit. Spectra below this tolerance will be set to NaN. "
            "Leaving this unset may result in poor or NaN spectra in the output cubes"
        ),
    ),
    overwrite: bool = Field(True, description="Overwrite output image if it already exists"),
    stokes_index: int = Field(0, description="Index of stokes channel (zero-based) to use."),
    rest_freq: float | None = Field(None, description="Cube rest frequency in MHz. Will ignore the one in the FITS header if it exists."),
    stokes_axis: bool = Field(True, description="### DEPRECATED #### Set this flag if the input image has a stokes dimension. (Default is True)."),
    hdu_index: int = Field(0, description="FITS primary HDU index", json_schema_extra={"abbreviation": "hi"}),
    ra_chunks: int = Field(64, description="Chunking along RA-axis. If set to zero, no Chunking is perfomed."),
    nworkers: int = Field(
        1,
        description=(
            "Number of parallel worker threads, one RA chunk each -- so it is also capped by the chunk count, 1440/--ra-chunks for "
            "a typical cube. RAISING THIS USUALLY MAKES THE RUN SLOWER, which is why it defaults to 1. Dask runs these as threads, "
            "and continuum fitting is a per-spectrum Python loop (ContSub.fitContinuum) around a scipy call, so the loop holds the "
            "GIL and the workers contend for it rather than working in parallel. Measured on a 1440x1440x256 cube with the default "
            "b-spline model, one fit pass took 4.3 min with a single worker and 26.5 min with 24. Raise it only for a model whose "
            "per-spectrum work is dominated by a C routine that drops the GIL, and only after measuring."
        ),
    ),
    loglevel: LOG_LEVELS = Field("info", description="Log level for the output"),
    doppler_frame: FRAMES | None = Field(
        None,
        description=(
            "Spectral reference frame to Doppler-correct the output cubes to, applied after the continuum fit. A cube has already been "
            "integrated over time, so only a single correction factor can be applied: this is safe only when the line-of-sight velocity "
            "drifts by much less than one channel over the observation, and cannot undo smearing from a larger drift. Pass "
            "--doppler-obs-duration to have that checked. For long tracks, correct in the visibility plane instead (doppler-mowjsub). "
            "Leave unset to skip Doppler correction."
        ),
    ),
    doppler_chan_grid: str = Field(
        "auto",
        description=(
            "Output channel grid for the Doppler correction. 'auto' derives it from this cube, dropping one guard channel at each end. "
            "Otherwise give 'nchan,chan0,chanwidth' with frequency units, e.g. '1000,1419.5MHz,26.1kHz'; use this to place several "
            "observations on one common grid for stacking."
        ),
    ),
    doppler_interpolation: INTERPOLATIONS = Field(
        "nearest",
        description=(
            "Interpolation used when resampling onto the Doppler-corrected grid. 'nearest' leaves channel noise uncorrelated; 'linear' is "
            "smoother but correlates adjacent channels."
        ),
    ),
    doppler_source_vel: float | None = Field(
        None,
        description=(
            "Systemic radial velocity of the source in km/s, positive for recession. Required with --doppler-frame=source, since a cube carries no SOURCE::SYSVEL to fall back on."
        ),
    ),
    doppler_time: str | None = Field(None, description="Observation epoch for the Doppler correction, as an ISO date or an MJD. Overrides the cube's MJD-OBS or DATE-OBS."),
    doppler_obs_duration: float | None = Field(
        None,
        description=(
            "Length of the observation in hours. Only used to check how far the line-of-sight velocity drifted over the track, and to take "
            "the correction at the midpoint rather than the start. Nothing standard records this in a FITS header, so it must be given; "
            "without it the drift check is skipped."
        ),
    ),
    doppler_telescope: str | None = Field(
        None,
        description=(
            "Observatory name, resolved through astropy's site registry. Overrides the cube's OBSGEO-X/Y/Z or TELESCOP keywords, and is needed when the header carries neither."
        ),
    ),
    doppler_phase_centre: str | None = Field(
        None,
        description=("Field centre used to project the observatory velocity, as 'RA,Dec' in degrees or sexagesimal. Overrides the cube's WCS reference coordinate."),
    ),
) -> None:
    opts = SimpleNamespace(**locals())
    return runit(opts)


#: Uniform handle for this module's pystep, so the StepRef can be looked up
#: generically without knowing the function's own name.
step = im_mowjsub

command = make_command(im_mowjsub, positional="input_image", must_exist=("input_image", "mask_image", "source_catalogue", "source_mask"))
