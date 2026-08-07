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
    except Exception:  # not a format astropy recognises -- try a plain list
        rows = np.atleast_2d(np.loadtxt(str(path), comments="#", ndmin=2))
        if rows.shape[1] < 2:
            raise ValueError(f"{path} has fewer than two columns; expected at least 'ra dec' in degrees.") from None
        maj = rows[:, 2] if rows.shape[1] > 2 else np.full(rows.shape[0], np.nan)
        return rows[:, 0], rows[:, 1], maj

    def pick(names):
        for name in names:
            if name in table.colnames:
                return np.asarray(table[name], dtype=float)
        return None

    ra, dec = pick(_RA_NAMES), pick(_DEC_NAMES)
    if ra is None or dec is None:
        raise ValueError(f"{path} has columns {list(table.colnames)}; none of {_RA_NAMES} and {_DEC_NAMES} to read a position from.")

    maj = pick(_MAJ_NAMES)
    if maj is None:
        maj = np.full(ra.size, np.nan)

    return ra, dec, maj


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

    radius_pix = 3.0 if radius_pix is None else float(radius_pix)
    # Degrees per pixel, from the WCS rather than CDELT so a CD/PC matrix works.
    # `wcs.utils`' function returns degrees as an array; the WCS *method* of the
    # same name returns a list of Quantities, which is not what this wants.
    scale = float(np.mean(np.abs(proj_plane_pixel_scales(wcs))))

    x, y = wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)

    if maj_deg is None:
        maj_deg = np.full(np.size(ra_deg), np.nan)
    maj_deg = np.asarray(maj_deg, float)

    # A catalogued major axis is a FWHM; take the footprint out to that radius.
    radii = np.where(np.isfinite(maj_deg) & (maj_deg > 0), maj_deg / scale, radius_pix)

    footprint = np.zeros((nra, ndec), dtype=bool)
    grid_x, grid_y = np.mgrid[0:nra, 0:ndec]
    inside = 0
    for xi, yi, ri in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(radii)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ri = max(float(ri), 1.0)
        if xi < -ri or yi < -ri or xi > nra + ri or yi > ndec + ri:
            continue  # outside the image
        footprint |= ((grid_x - xi) ** 2 + (grid_y - yi) ** 2) <= ri**2
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
