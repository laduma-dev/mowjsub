"""Doppler-tracking corrections for visibility data.

This reproduces the spectral-frame conversion that CASA ``mstransform`` performs
with ``regridms=True``: the topocentric channel frequencies recorded in the MS
are expressed in a target spectral reference frame, which changes from timestamp
to timestamp as the observatory moves, and every spectrum is then resampled onto
a single fixed output channel grid.

Frame velocities and the composition of conversion steps follow casacore's
``MeasTable`` and ``MCFrequency``, so the grids produced here agree with those
CASA would produce. Each conversion step scales frequency by

    sqrt((1 - beta) / (1 + beta))

where ``beta`` is the line-of-sight velocity of the observer relative to the
target frame over ``c``, positive when the observer approaches the source.
Steps compose multiplicatively, exactly as casacore applies them in sequence.
"""

import numpy as np
from astropy import units
from astropy.coordinates import ICRS, SkyCoord, get_body_barycentric_posvel
from astropy.time import Time

#: Speed of light in m/s (casacore ``C::c``).
C = 299792458.0

#: Spectral frames that may be requested as a Doppler-correction target.
FRAMES = (
    "topo",
    "geo",
    "bary",
    "lsrk",
    "lsrd",
    "galacto",
    "lgroup",
    "cmb",
    "source",
)

#: MS ``SPECTRAL_WINDOW::MEAS_FREQ_REF`` codes (casacore ``MFrequency::Types``).
#: ``source`` has no code of its own; a source-frame grid is a rest-frame grid.
FRAME_CODES = {
    "source": 0,
    "lsrk": 1,
    "lsrd": 2,
    "bary": 3,
    "geo": 4,
    "topo": 5,
    "galacto": 6,
    "lgroup": 7,
    "cmb": 8,
}

# Apex velocities from casacore MeasTable, as (speed in m/s, J2000 unit vector).
# Each vector points along the motion of the inner frame relative to the outer
# one, so every step below is applied with the same sign convention.
_APEX = {
    # 20 km/s toward RA 18h, Dec +30 (B1900) -- the classical HI solar motion.
    "lsrk": (20.0e3, (0.0145021, -0.865863, 0.500071)),
    # (9, 12, 7) km/s in galactic coordinates, i.e. the IAU dynamical LSR.
    "lsrd": (np.sqrt(81.0 + 144.0 + 49.0) * 1e3, (-0.0385568, -0.881138, 0.471285)),
    # Rotation of the LSR about the galactic centre, 220 km/s toward l=90.
    "lsrgal": (220.0e3, (0.494109, -0.44483, 0.746982)),
    # 308 km/s toward l,b = 105,-7.
    "lgroup": (308.0e3, (0.593553979227, -0.177954636914, 0.784873124106)),
    # 369.5 km/s toward l,b = 264.4,+48.4.
    "cmb": (369.5e3, (-0.97176985257, 0.202393953108, -0.121243727187)),
}

# Steps applied after the topocentric-to-barycentric correction. casacore
# reaches GALACTO via LSRD, so that route is reproduced here.
_CHAIN = {
    "bary": (),
    "lsrk": ("lsrk",),
    "lsrd": ("lsrd",),
    "galacto": ("lsrd", "lsrgal"),
    "lgroup": ("lgroup",),
    "cmb": ("cmb",),
    "source": (),
}

#: FITS ``SPECSYS`` names for each frame (FITS WCS Paper III, table 29).
#: Distinct from :data:`FRAME_CODES`, which is the MS ``MEAS_FREQ_REF`` integer
#: convention -- the two describe the same frames in different file formats and
#: must not be substituted for one another.
FITS_SPECSYS = {
    "topo": "TOPOCENT",
    "geo": "GEOCENTR",
    "bary": "BARYCENT",
    "lsrk": "LSRK",
    "lsrd": "LSRD",
    "galacto": "GALACTOC",
    "lgroup": "LOCALGRP",
    "cmb": "CMBDIPOL",
    # A source-frame grid is a rest-frame grid; FITS spells that SOURCE.
    "source": "SOURCE",
}

#: Frequency units accepted in an explicit channel-grid specification.
_FREQ_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def _shift(beta):
    """Relativistic frequency scaling for a single frame-conversion step.

    Args:
        beta (float|np.ndarray): Line-of-sight velocity of the observer relative
            to the target frame, over c, positive when approaching the source.

    Returns:
        np.ndarray: Multiplicative frequency factor.
    """
    return np.sqrt((1.0 - beta) / (1.0 + beta))


def doppler_factors(times, phase_centre, location, frame, source_vel=None):
    """Per-timestamp factors converting topocentric frequency to ``frame``.

    The frequency of a channel in the output frame is its topocentric frequency
    multiplied by the factor returned here.

    Args:
        times (np.ndarray): MS ``TIME`` values, i.e. seconds since MJD 0, UTC.
        phase_centre (tuple): Field (RA, Dec) in radians.
        location (EarthLocation): Position of the observatory.
        frame (str): Target frame, one of :data:`FRAMES`.
        source_vel (float, optional): Systemic radial velocity of the source in
            m/s, positive for recession. Required for ``frame="source"``.

    Returns:
        np.ndarray: One factor per entry in ``times``.

    Raises:
        ValueError: The frame is unknown, or ``source_vel`` is missing for
            ``frame="source"``.
    """
    frame = str(frame).lower()
    if frame not in FRAMES:
        raise ValueError(f"Unknown Doppler frame '{frame}'. Choose one of: {', '.join(FRAMES)}.")

    times = np.atleast_1d(np.asarray(times, dtype=float))

    # Topocentric is what the MS already holds, so nothing moves.
    if frame == "topo":
        return np.ones(times.size)

    ra, dec = phase_centre
    target = SkyCoord(ra=ra * units.rad, dec=dec * units.rad, frame=ICRS)
    obstime = Time(times / 86400.0, format="mjd", scale="utc")

    # Direction cosines of the field centre, for projecting the frame velocities.
    lineofsight = target.cartesian.xyz.value

    # Diurnal term: the observer's motion about the Earth's centre.
    _, vdiurnal = location.get_gcrs_posvel(obstime)
    vdiurnal = np.atleast_2d(vdiurnal.xyz.to_value(units.m / units.s).T)

    if frame == "geo":
        return _shift(vdiurnal.dot(lineofsight) / C)

    # Every remaining frame is reached through the barycentre. Project the
    # observer's velocity relative to the solar-system barycentre rather than
    # using astropy's radial_velocity_correction: the latter carries relativistic
    # terms casacore omits, which would offset us from CASA by ~4.7 m/s. The
    # plain projection tracks casacore to well under 0.1 m/s.
    _, vorbital = get_body_barycentric_posvel("earth", obstime)
    vorbital = np.atleast_2d(vorbital.xyz.to_value(units.m / units.s).T)

    factors = _shift((vdiurnal + vorbital).dot(lineofsight) / C)

    for step in _CHAIN[frame]:
        speed, direction = _APEX[step]
        factors = factors * _shift(speed * float(np.dot(direction, lineofsight)) / C)

    if frame == "source":
        if source_vel is None:
            raise ValueError("frame='source' needs the source systemic velocity; pass source_vel or set --doppler-source-vel.")
        # A receding source means the observer recedes from it, hence the sign.
        factors = factors * _shift(-float(source_vel) / C)

    return factors


def common_channel_grid(freqs, factors):
    """Output channel grid covered by every timestamp.

    Mirrors the 'auto' grid caracal computes for CASA ``mstransform``: take the
    intersection of the Doppler-shifted bands over all timestamps, widen the
    channels to the coarsest shifted width, then drop one channel at each end so
    no output channel can fall outside the input band.

    Args:
        freqs (np.ndarray): Topocentric channel frequencies in Hz. May ascend or
            descend, but must be regularly spaced.
        factors (np.ndarray): Doppler factors, as returned by
            :func:`doppler_factors`.

    Returns:
        tuple: ``(nchan, chan0, chanwidth)``, the latter two in Hz.

    Raises:
        ValueError: The intersection is empty, i.e. the band drifts by more than
            its own width over the observation.
    """
    freqs = np.asarray(freqs, dtype=float)
    factors = np.atleast_1d(np.asarray(factors, dtype=float))

    chanw = (freqs[-1] - freqs[0]) / (freqs.size - 1)
    fmin, fmax = factors.min(), factors.max()

    # The band slides with the Doppler factor, so the common coverage runs from
    # the least-favourable shift of the first channel to that of the last.
    if chanw < 0:
        chan0, chanl = freqs[0] * fmin, freqs[-1] * fmax
    else:
        chan0, chanl = freqs[0] * fmax, freqs[-1] * fmin
    width = chanw * fmax

    chan0 += width
    chanl -= width

    nchan = int(np.floor((chanl - chan0) / width)) + 1
    if nchan < 1:
        raise ValueError("Doppler shift over this observation exceeds the bandwidth, leaving no common channel grid. Pass an explicit grid via --doppler-chan-grid.")

    return nchan, chan0, width


def parse_channel_grid(spec):
    """Parse an explicit ``'nchan,chan0,chanwidth'`` grid specification.

    Args:
        spec (str): For example ``'1000,1419.5MHz,26.1kHz'``. The two frequencies
            must carry a unit; ``chanwidth`` may be negative for a descending grid.

    Returns:
        tuple: ``(nchan, chan0, chanwidth)``, the latter two in Hz.

    Raises:
        ValueError: The specification is malformed or carries unusable units.
    """
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Could not parse Doppler channel grid '{spec}'. Expected 'nchan,chan0,chanwidth', e.g. '1000,1419.5MHz,26.1kHz'.")

    try:
        nchan = int(parts[0])
    except ValueError:
        raise ValueError(f"Doppler channel grid: nchan must be an integer, got '{parts[0]}'.") from None
    if nchan < 1:
        raise ValueError(f"Doppler channel grid: nchan must be positive, got {nchan}.")

    return nchan, _to_hz(parts[1]), _to_hz(parts[2])


def _to_hz(value):
    """Convert a unit-bearing frequency string to Hz."""
    text = str(value).strip()
    digits = text.rstrip("".join(set(text) - set("0123456789.+-eE")))
    unit = text[len(digits) :].strip().lower()

    if not unit:
        raise ValueError(f"Doppler channel grid: '{text}' needs a frequency unit, one of {', '.join(sorted(_FREQ_UNITS))}.")
    if unit not in _FREQ_UNITS:
        raise ValueError(f"Doppler channel grid: '{text}' does not carry a frequency unit. Only frequency-mode grids are supported; use one of {', '.join(sorted(_FREQ_UNITS))}.")
    try:
        return float(digits) * _FREQ_UNITS[unit]
    except ValueError:
        raise ValueError(f"Doppler channel grid: could not read a number from '{text}'.") from None


def grid_frequencies(nchan, chan0, chanwidth):
    """Expand a ``(nchan, chan0, chanwidth)`` triple into channel frequencies."""
    return chan0 + chanwidth * np.arange(nchan, dtype=float)


def resample_map(freqs_in, factor, freqs_out, interpolation="nearest"):
    """Map output channels onto the input channels that feed them.

    Args:
        freqs_in (np.ndarray): Topocentric input channel frequencies in Hz.
        factor (float): Doppler factor for this timestamp.
        freqs_out (np.ndarray): Output channel frequencies in Hz, in the target
            frame.
        interpolation (str): ``'nearest'`` or ``'linear'``.

    Returns:
        tuple: ``(indices, weights, valid)``. ``indices`` and ``weights`` have
        shape ``(nchan_out, k)`` where ``k`` is 1 for nearest and 2 for linear;
        ``valid`` is a boolean mask of output channels the input band covers.

    Raises:
        ValueError: Unknown interpolation scheme.
    """
    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"Unknown interpolation '{interpolation}'. Choose 'nearest' or 'linear'.")

    shifted = np.asarray(freqs_in, dtype=float) * factor
    freqs_out = np.asarray(freqs_out, dtype=float)

    # searchsorted needs ascending input; descending bands are common in MS data.
    order = np.argsort(shifted)
    ascending = shifted[order]

    upper = np.clip(np.searchsorted(ascending, freqs_out), 1, ascending.size - 1)
    lower = upper - 1
    flo, fhi = ascending[lower], ascending[upper]

    # Fractional position within the bracketing pair; outside [0, 1] means the
    # output channel lies beyond the input band.
    frac = (freqs_out - flo) / (fhi - flo)

    if interpolation == "nearest":
        pick = np.where(frac <= 0.5, lower, upper)
        indices = order[pick][:, None]
        weights = np.ones((freqs_out.size, 1))
        # A channel is covered if it sits within half a channel of the band.
        valid = (frac >= -0.5) & (frac <= 1.5)
    else:
        indices = np.stack([order[lower], order[upper]], axis=1)
        clipped = np.clip(frac, 0.0, 1.0)
        weights = np.stack([1.0 - clipped, clipped], axis=1)
        valid = (frac >= 0.0) & (frac <= 1.0)

    return indices, weights, valid


def resample_cube(data, factor, freqs_in, freqs_out, axis, interpolation="nearest"):
    """Resample a cube along its spectral axis onto a fixed output grid.

    The image-plane counterpart of :func:`regrid_rows`. A cube has already been
    integrated over time, so a single Doppler factor applies to all of it rather
    than one per row. Output channels the input band does not cover are set to
    NaN, which is the image-plane equivalent of raising a flag.

    Note what this cannot do: the observatory's line-of-sight velocity drifts
    during a track, and imaging has already summed over that drift. Shifting the
    axis afterwards re-centres the spectrum but cannot undo the smearing, so this
    is only equivalent to :func:`regrid_rows` when the drift over the observation
    is small compared to a channel.

    Args:
        data (np.ndarray|dask.array): Cube of any dimensionality.
        factor (float): Doppler factor for the observation.
        freqs_in (np.ndarray): Topocentric input channel frequencies in Hz.
        freqs_out (np.ndarray): Output channel frequencies in Hz, in the target
            frame.
        axis (int): Index of the spectral axis of ``data``.
        interpolation (str): ``'nearest'`` or ``'linear'``.

    Returns:
        np.ndarray|dask.array: ``data`` with its spectral axis replaced by
        ``freqs_out``.
    """
    indices, weights, valid = resample_map(freqs_in, factor, freqs_out, interpolation)

    # Broadcast per-channel vectors against the spectral axis alone.
    shape = [1] * np.ndim(data)
    shape[axis] = -1

    # Nearest weights are all 1, so skipping the multiply there keeps a float32
    # cube float32 instead of promoting it against float64 coefficients.
    dtype = np.result_type(data.dtype, np.float32)

    resampled = None
    for k in range(indices.shape[1]):
        contribution = np.take(data, indices[:, k], axis=axis)
        if interpolation != "nearest":
            contribution = contribution * weights[:, k].reshape(shape).astype(dtype)
        resampled = contribution if resampled is None else resampled + contribution

    # np.where rather than masked assignment, so this stays lazy under dask.
    return np.where(valid.reshape(shape), resampled, np.nan)


def regrid_rows(data, flags, weights, factors, freqs_in, freqs_out, interpolation="nearest"):
    """Resample row-ordered spectra onto a fixed output channel grid.

    Rows sharing a timestamp share a Doppler factor and therefore a channel map,
    so the work is done once per distinct factor.

    An output channel is flagged if it falls outside the input band at that
    timestamp, or if any input channel contributing to it is flagged. Output
    weights follow from propagating ``1/sigma**2`` through the interpolation.

    Args:
        data (np.ndarray): Visibilities, shape ``(nrow, nchan_in, ncorr)``.
        flags (np.ndarray): Boolean flags, same shape as ``data``.
        weights (np.ndarray): Weights, same shape as ``data``.
        factors (np.ndarray): One Doppler factor per row.
        freqs_in (np.ndarray): Topocentric input channel frequencies in Hz.
        freqs_out (np.ndarray): Output channel frequencies in Hz.
        interpolation (str): ``'nearest'`` or ``'linear'``.

    Returns:
        tuple: Regridded ``(data, flags, weights)``, each with ``nchan_out``
        channels.
    """
    nrow, _, ncorr = data.shape
    nchan_out = np.asarray(freqs_out).size

    out_data = np.zeros((nrow, nchan_out, ncorr), dtype=data.dtype)
    out_flags = np.ones((nrow, nchan_out, ncorr), dtype=bool)
    out_weights = np.zeros((nrow, nchan_out, ncorr), dtype=weights.dtype)

    for factor in np.unique(factors):
        rows = np.flatnonzero(factors == factor)
        indices, coeffs, valid = resample_map(freqs_in, factor, freqs_out, interpolation)

        block = data[rows]
        block_flags = flags[rows]
        block_weights = weights[rows]

        accum = np.zeros((rows.size, nchan_out, ncorr), dtype=data.dtype)
        # Flag an output channel as soon as any contributor to it is flagged.
        bad = ~valid[None, :, None] | np.zeros((rows.size, nchan_out, ncorr), dtype=bool)
        variance = np.zeros((rows.size, nchan_out, ncorr), dtype=np.float64)

        for k in range(indices.shape[1]):
            take = indices[:, k]
            coeff = coeffs[:, k][None, :, None]
            accum += block[:, take, :] * coeff
            bad |= block_flags[:, take, :]

            src_weight = block_weights[:, take, :]
            # 1/w is the variance of that channel; a zero weight carries no
            # information, so it only matters via the flag already raised above.
            with np.errstate(divide="ignore", invalid="ignore"):
                contribution = np.where(src_weight > 0, coeff**2 / src_weight, 0.0)
            variance += contribution

        with np.errstate(divide="ignore", invalid="ignore"):
            resolved = np.where(variance > 0, 1.0 / variance, 0.0)

        out_data[rows] = np.where(bad, 0, accum)
        out_flags[rows] = bad
        out_weights[rows] = np.where(bad, 0, resolved).astype(weights.dtype)

    return out_data, out_flags, out_weights
