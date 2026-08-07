"""Continuum source positions, and the sky footprint they define.

Absorption cannot exist without a background continuum source, so the sightlines
where it is physically possible are known before the cube is searched. Naming
them lets the automask carry a lower threshold exactly there, which is what
``utils.get_automask`` does with the footprint this module builds.

Two ways to say where the sources are:

* a catalogue -- PyBDSF's or SoFiA's output, or a plain ``ra dec`` list typed by
  hand;
* a 2D FITS mask, for a footprint built by something this module cannot parse.

Finding the sources is deliberately *not* done here. A pipeline runs a finder as
its own step -- caracal reaches PyBDSF and SoFiA through their dosho cabs -- and
points this at the result. That keeps a compiled source-finder out of mowjsub's
dependencies, and it means the finder sees the right image: the full-band
continuum image the imaging worker already made, whose sensitivity is far better
than anything collapsed out of a single line cube.
"""

import logging

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from . import LOGGER

log = logging.getLogger(LOGGER)

#: Column names PyBDSF and common catalogue formats use for sky position.
_RA_NAMES = ("RA", "ra", "RA_deg", "RAJ2000", "ra_deg", "_RAJ2000")
_DEC_NAMES = ("DEC", "dec", "DEC_deg", "DEJ2000", "dec_deg", "_DEJ2000", "Dec")
#: Column names for a source's major axis, used to size its footprint.
_MAJ_NAMES = ("Maj", "MAJ", "maj", "Maj_deg", "DC_Maj")


def read_catalogue(path):
    """Read source positions from a catalogue.

    Accepts anything ``astropy.table`` can read -- PyBDSF's FITS and ASCII
    outputs among them -- and falls back to parsing a bare whitespace-separated
    ``ra dec [maj_deg]`` list, which is the form a user types by hand.

    Args:
        path (str|Path): Catalogue file.

    Returns:
        tuple: ``(ra_deg, dec_deg, maj_deg)`` as float arrays. ``maj_deg`` is
        NaN for sources whose extent the catalogue does not give.

    Raises:
        ValueError: If no position columns can be identified.
    """
    from astropy.table import Table

    try:
        table = Table.read(path)
        columns = list(table.colnames)

        def pick(names):
            for name in names:
                if name in columns:
                    return np.asarray(table[name], dtype=float)
            return None
    except Exception:
        # Not a format astropy recognises. PyBDSF's own ASCII outputs name their
        # columns in a comment line -- a bare list for `.srl`, a `#format:` line
        # for the sky model -- and those are what a pipeline hands us, so read
        # the names from there rather than making the caller convert.
        names = _column_names(path)
        if names:
            columns = names

            def index_of(wanted):
                for name in wanted:
                    if name in columns:
                        return columns.index(name)
                return None

            ra_at, dec_at, maj_at = (index_of(n) for n in (_RA_NAMES, _DEC_NAMES, _MAJ_NAMES))
            if ra_at is None or dec_at is None:
                raise ValueError(f"{path} has columns {columns}; none of {_RA_NAMES} and {_DEC_NAMES} to read a position from.") from None

            # Only the columns we need. A PyBDSF source list ends with a
            # non-numeric S_Code, so reading every column fails outright.
            wanted = [ra_at, dec_at] + ([maj_at] if maj_at is not None else [])
            rows = np.atleast_2d(np.loadtxt(str(path), comments="#", ndmin=2, usecols=wanted))
            maj = rows[:, 2] if maj_at is not None else np.full(rows.shape[0], np.nan)
            return _wrap_ra(rows[:, 0]), rows[:, 1], maj

        # A hand-written list: 'ra dec [maj_deg]', by position.
        rows = np.atleast_2d(np.loadtxt(str(path), comments="#", ndmin=2))
        if rows.shape[1] < 2:
            raise ValueError(f"{path} has fewer than two columns; expected at least 'ra dec' in degrees.") from None
        maj = rows[:, 2] if rows.shape[1] > 2 else np.full(rows.shape[0], np.nan)
        return _wrap_ra(rows[:, 0]), rows[:, 1], maj

    ra, dec = pick(_RA_NAMES), pick(_DEC_NAMES)
    if ra is None or dec is None:
        raise ValueError(f"{path} has columns {columns}; none of {_RA_NAMES} and {_DEC_NAMES} to read a position from.")

    maj = pick(_MAJ_NAMES)
    if maj is None:
        maj = np.full(ra.size, np.nan)

    return _wrap_ra(ra), dec, maj


def _wrap_ra(ra):
    """Right ascension into [0, 360). PyBDSF writes some of it negative."""
    return np.asarray(ra, dtype=float) % 360.0


def _column_names(path):
    """Column names from a comment header, for PyBDSF's ASCII outputs.

    Its source list puts them on the last comment line before the data, and its
    sky model on a ``#format:`` line. Returns None when no comment line looks
    like a column header.
    """
    header = None
    with open(path) as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            tokens = line.lstrip("#").strip()
            if tokens.lower().startswith("format:"):
                tokens = tokens.split(":", 1)[1].strip()
            fields = tokens.split()
            # A header names a position; prose about the file does not.
            if any(f in _RA_NAMES for f in fields) and any(f in _DEC_NAMES for f in fields):
                header = fields
    return header


def footprint_from_catalogue(header, ra_deg, dec_deg, maj_deg=None, radius_pix=None, shape=None):
    """Mark the pixels a catalogue's sources cover.

    A source's footprint is a disc: its catalogued major axis where there is
    one, otherwise ``radius_pix``. The disc is what the deeper threshold is
    applied over, so it wants to be the region the source actually illuminates
    rather than a single pixel -- an unresolved source still spreads over the
    restoring beam.

    Args:
        header: FITS header of the cube, for its celestial WCS and pixel scale.
        ra_deg, dec_deg (array): Source positions, degrees.
        maj_deg (array|None): Per-source major axis in degrees; NaN entries fall
            back to ``radius_pix``.
        radius_pix (float|None): Default radius, pixels. Defaults to 3.
        shape (tuple|None): ``(nra, ndec)``. Read from the header when omitted.

    Returns:
        np.ndarray: Boolean ``(nra, ndec)``, True inside a source.
    """
    wcs = WCS(header).celestial
    if shape is None:
        shape = (int(header["NAXIS1"]), int(header["NAXIS2"]))
    nra, ndec = shape

    # Degrees per pixel, from the WCS rather than CDELT so a CD/PC matrix works.
    # `wcs.utils`' function returns degrees as an array; the WCS *method* of the
    # same name returns a list of Quantities, which is not what this wants.
    scale = float(np.mean(np.abs(proj_plane_pixel_scales(wcs))))

    x, y = wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)

    if maj_deg is None:
        maj_deg = np.full(np.size(ra_deg), np.nan)
    maj_deg = np.asarray(maj_deg, float)

    # A catalogued major axis is a FWHM; take the footprint out to that radius.
    # NaN means the catalogue stated no extent, and stays NaN so the fallback
    # below can tell that apart from a stated one.
    radii = np.where(np.isfinite(maj_deg) & (maj_deg > 0), maj_deg / scale, np.nan)

    # An unresolved source still covers the restoring beam, so where the
    # catalogue gives no extent the footprint is the beam itself -- shape and
    # orientation included, since a disc of the same area points the wrong way
    # on any elongated beam. An explicit --source-radius overrides that, and a
    # cube with no beam at all falls back to a circle of 3 pixels.
    try:
        beam_a, beam_b, beam_theta = beam_pixel_ellipse(header)
    except RuntimeError:
        beam_a = beam_b = 3.0
        beam_theta = 0.0
    if radius_pix is not None:
        beam_a = beam_b = float(radius_pix)
        beam_theta = 0.0

    footprint = np.zeros((nra, ndec), dtype=bool)
    inside = 0
    for xi, yi, ri in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(radii)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        if np.isfinite(ri) and ri > 0:
            # A catalogued extent is a major axis; keep the beam's axis ratio
            # and orientation, which is the shape the source is smeared by.
            semi_major = max(float(ri), beam_a)
            semi_minor = semi_major * (beam_b / beam_a if beam_a else 1.0)
        else:
            semi_major, semi_minor = beam_a, beam_b

        reach = max(semi_major, semi_minor)
        if xi < -reach or yi < -reach or xi > nra + reach or yi > ndec + reach:
            continue  # outside the image
        footprint |= ellipse_mask((nra, ndec), xi, yi, semi_major, semi_minor, beam_theta)
        inside += 1

    log.info(f"Source footprint: {inside} of {np.size(ra_deg)} catalogued sources fall in the image, covering {footprint.mean() * 100:.2f}% of it")
    if inside == 0:
        log.warning("No catalogued source falls inside the image. The deeper source threshold will reach nothing.")

    return footprint


def footprint_from_mask(path, shape):
    """Read a 2D footprint from a FITS mask, collapsing any degenerate axes."""
    data = np.squeeze(fits.getdata(str(path)))
    if data.ndim != 2:
        raise ValueError(f"{path} has shape {data.shape} after squeezing; a source mask must be a 2D image.")
    # FITS is (dec, ra); the cube this masks is (ra, dec, spectral).
    data = data.T
    if data.shape != tuple(shape):
        raise ValueError(f"{path} is {data.shape} after transposing to (ra, dec), but the cube is {tuple(shape)}.")
    return data.astype(bool)


def beam_pixel_ellipse(header):
    """The restoring beam as a pixel-space ellipse.

    Returns semi-axes in pixels and the position angle rotated into *pixel*
    coordinates. ``BPA`` is measured North through East, which is not the pixel
    frame: with the usual negative ``CDELT1``, East runs towards *decreasing* x,
    so the major axis at position angle ``t`` points along ``(-sin t, cos t)``.
    A cube with a positive ``CDELT1`` flips that, and the sign is taken from the
    WCS rather than assumed -- an unnoticed sign here mirrors every ellipse
    about the Dec axis, which is invisible on a round beam and wrong on any
    other.

    Args:
        header: FITS header with ``BMAJ``/``BMIN`` in degrees, optionally
            ``BPA`` in degrees, and a celestial WCS.

    Returns:
        tuple: ``(semi_major_pix, semi_minor_pix, theta_pix_radians)``.

    Raises:
        RuntimeError: If the header states no beam.
    """
    bmaj, bmin = header.get("BMAJ"), header.get("BMIN")
    if not bmaj:
        raise RuntimeError(
            "A sky-extent cut in beams needs BMAJ (and BMIN, BPA) in the header, and this cube states none. Set the beam on the cube, or leave --min-sky-beams at 0."
        )
    bmin = bmin or bmaj
    bpa = float(header.get("BPA") or 0.0)

    wcs = WCS(header).celestial
    scale = float(np.mean(np.abs(proj_plane_pixel_scales(wcs))))

    # Does x increase with RA or against it?
    cdelt = wcs.wcs.cdelt[0] if wcs.wcs.has_cd() is False else wcs.wcs.cd[0, 0]
    east_is_minus_x = cdelt < 0

    theta = np.deg2rad(bpa)
    if not east_is_minus_x:
        theta = -theta

    return float(bmaj) / scale / 2.0, float(bmin) / scale / 2.0, theta


def ellipse_mask(shape, x0, y0, semi_major, semi_minor, theta):
    """Boolean ``(nx, ny)`` marking an ellipse, in pixel coordinates.

    ``theta`` is the major-axis angle already rotated into the pixel frame by
    :func:`beam_pixel_ellipse`, so the major axis points along
    ``(-sin theta, cos theta)``.
    """
    nx, ny = shape
    gx, gy = np.mgrid[0:nx, 0:ny]
    dx, dy = gx - x0, gy - y0

    # Project onto the major and minor axes.
    along = -dx * np.sin(theta) + dy * np.cos(theta)
    across = -dx * np.cos(theta) - dy * np.sin(theta)

    return (along / max(semi_major, 0.5)) ** 2 + (across / max(semi_minor, 0.5)) ** 2 <= 1.0


def beam_element(header, beams):
    """A beam-shaped structuring element ``beams`` across, for the sky cut.

    Testing sky extent by *area* is orientation-blind: an elongated streak with
    as many pixels as a beam passes a count it should not. Eroding a component's
    sky projection by this element instead asks whether it actually contains a
    beam-shaped region, which is the thing that makes a detection credible --
    nothing on the sky is smaller than the beam.

    ``beams`` is a linear size in beam major axes, so 0.5 is half a beam across.
    Both semi-axes are scaled by it and each is rounded **up** to at least half
    a pixel, so any positive request selects a non-empty element.

    Args:
        header: FITS header carrying the beam.
        beams (float): Linear extent in beam major axes. ``0`` disables.

    Returns:
        np.ndarray|None: Boolean structuring element, or None when disabled.
    """
    if not beams:
        return None

    semi_major, semi_minor, theta = beam_pixel_ellipse(header)
    a, b = semi_major * beams, semi_minor * beams

    # Big enough to hold the ellipse at any orientation.
    half = int(np.ceil(max(a, b)))
    size = 2 * half + 1
    element = ellipse_mask((size, size), half, half, a, b, theta)

    log.info(
        f"Sky-extent cut: {beams} x BMAJ is a {element.sum()}-pixel beam-shaped element "
        f"(beam {2 * semi_major:.1f} x {2 * semi_minor:.1f} pixels at BPA {np.rad2deg(theta):.1f} deg in pixel frame)"
    )
    return element
