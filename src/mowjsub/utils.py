import datetime
import warnings
from collections import namedtuple
from typing import Dict

import astropy.io.fits as fitsio
import dask.array as da
import numpy as np
import xarray as xr
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from casacore.tables import table
from daskms import Dataset, xds_from_ms, xds_from_table
from scabha import init_logger
from scabha.basetypes import File
from tqdm.dask import TqdmCallback
from xarrayfits import xds_from_fits

from mowjsub import BIN
from mowjsub.doppler import (
    FITS_SPECSYS,
    FRAME_CODES,
    C,
    common_channel_grid,
    doppler_factors,
    grid_frequencies,
    parse_channel_grid,
    regrid_rows,
    resample_cube,
)
from mowjsub.image_plane import ContSub
from mowjsub.masking import Mask, PixSigmaClip

warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")

log = init_logger(BIN.im_plane)


def get_automask(cube, fitfunc, sigma_clip):
    """
    Generate a binary mask by sigma-thresholding the input cube

    Args:
        xspec (Array): Spectral coordinates
        cube (Array): Data cdube
        sigma_clip(float): Sigma clip level

    Returns:
        Array : Binary mask (False is masked, True is not)
    """

    log.info("Creating binary mask as requested")
    contsub = ContSub(fitfunc)
    cont_model = contsub.fitContinuum(cube, mask=None)

    clip = PixSigmaClip(sigma_clip)

    mask = Mask(clip).getMask(cube - cont_model)
    log.info("Mask created sucessfully")

    return ~mask


def chans_in_velwidth(freqs: np.ndarray, velwidth: float):
    """
    Calculates the number of channels in given velocity width

    A channel's velocity width is dv = c * df / f, so it varies as 1/f across
    the band -- 0.4% end to end over a typical L-band cube. The reference
    frequency is therefore stated explicitly, as the band centre, rather than
    left implied by whichever channels happen to get sampled.

    Args:
        freqs (np.ndarray): Frequency grid in Hz. Need not be sorted.
        velwidth (float): Velocity width in m/s

    Returns:
        int: Channel count, at least 1.
    """
    freqs = np.sort(np.asarray(freqs, dtype=float))

    # Mean spacing rather than a single pair: tolerates the slightly uneven
    # grids that come out of regridding, and does not depend on which two
    # channels are picked.
    df = np.abs(np.diff(freqs)).mean()
    f_ref = 0.5 * (freqs[0] + freqs[-1])
    dv = df / f_ref * C

    # Round, don't truncate. Truncating biases every conversion downwards by up
    # to a full channel -- 300 km/s on the test grid lands on 209.82 channels,
    # which int() turned into 209.
    return max(1, int(round(velwidth / dv)))


def zds_from_fits(fname, chunks=None, rest_freq=None, hdu_idx=0, add_freqs=False):
    """Creates Zarr store from a FITS file. The resulting array has
    dimensions = RA, DEC, SPECTRAL[, STOKES]

    Args:
        fname (str|path): FITS file_
        chunks (dict, optional): xarray chunk object. Defaults to {1: 25, 2:25}.

    Raises:
        RuntimeError: Input FITS file doesn't have a spectral axis
        FileNotFoundError: Input FITS file not found

    Returns:
        Zarr: Zarr array (persistant store, mode=w)
    """
    chunks = chunks or dict(ra=64, dec=None, spectral=None)
    fds = xds_from_fits(fname, hdus=hdu_idx)[0]

    header = fds.hdu.header
    if rest_freq:
        header["RESTFREQ"] = rest_freq * 1e6  # set it to Hz
    wcs = WCS(header, naxis="spectral stokes".split())

    axis_names = [header["CTYPE1"], header["CTYPE2"]] + wcs.axis_type_names
    if not wcs.has_spectral:
        raise RuntimeError("Input FITS file does not have a spectral axis")

    fds_xyz = fds.hdu.transpose(*axis_names)

    new_names = ["ra", "dec", "spectral"]
    if len(axis_names) == 4:
        new_names.append("stokes")

    coords = dict([(a, fds.hdu[b].values) for a, b in zip(new_names, axis_names)])
    if add_freqs:
        data_vars = {
            "DATA": (new_names, fds_xyz.data),
            "FREQS": (("spectral",), FitsHeader(header).retFreq()),
        }
    else:
        data_vars = {
            "DATA": (new_names, fds_xyz.data),
        }
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs=dict(
            info=f"Temporary copy of data from FITS file: {fname}",
            header=fitsio.Header(header),
        ),
    )

    return ds.chunk(chunks)


def subtract_fits(data_file: File, model_file: File, hdu_idx: int, ra_chunks: int = None):
    """Returns the residual of two FITS files as a FitsPrimaryHDU object

    Args:
        data_file (File): FITS file of the data
        model_file (File): FITS file of model data
        hdu_idx (int): FITS HDU index
        ra_chunks (int): Chunk size along the RA axis. Zero or unset reads the cube as a single chunk.

    Returns:
        FitsPrimaryHDU
    """

    # xarrayfits chunk keys are C-order axis indices, and RA is NAXIS1, i.e. the
    # fastest-varying FITS axis. In C order that is the last axis, whatever the
    # cube's dimensionality. Unlisted axes are read as a single chunk.
    ndim = fitsio.getheader(data_file, hdu_idx)["NAXIS"]
    chunks = {ndim - 1: ra_chunks} if ra_chunks else {}

    data_ds = xds_from_fits(data_file, hdus=hdu_idx, chunks=chunks)[0]
    model_ds = xds_from_fits(model_file, hdus=hdu_idx, chunks=chunks)[0]
    residual_ds = data_ds.hdu.data - model_ds.hdu.data

    out_ds = data_ds.assign(
        hdu=(data_ds.hdu.dims, residual_ds, data_ds.hdu.attrs),
    )
    header = fitsio.Header(data_ds.hdu.header)
    return fitsio.PrimaryHDU(out_ds.hdu.data, header=header)


def spectral_frequencies(header):
    """Channel frequencies of a cube, in Hz.

    The grid is read through the *low-level* WCS, which is what makes this
    independent of two things the previous implementation was tied to. It
    returns SI regardless of ``CUNIT``, so there is no unit to guess; and it
    converts pixels to frequencies without an observation time, where the
    high-level WCS refuses to do so at all. The spectral axis is whichever one
    the WCS says it is, rather than ``NAXIS3``.

    Args:
        header: FITS header of a spectral cube.

    Returns:
        np.ndarray: Channel frequency in Hz, one per channel.

    Raises:
        RuntimeError: The header describes no spectral axis.
    """
    wcs = WCS(header)
    axis = wcs.wcs.spec
    if axis < 0:
        raise RuntimeError("This cube has no spectral axis.")

    nchan = int(header[f"NAXIS{axis + 1}"])

    return np.asarray(wcs.spectral.array_index_to_world_values(np.arange(nchan)), dtype=float)


class FitsHeader:
    def __init__(self, header: Dict):
        self._header = header.copy()

    def retFreq(self):
        """Channel frequencies in MHz.

        MHz rather than Hz because the fitters work in it: ``FitFunc.prepare``
        converts back with ``self.freqs * 1e6`` for :func:`chans_in_velwidth`,
        and ``FitPolynomial`` fits ``numpy.polyfit`` against these values, where
        the scale decides the conditioning. Anything measuring a frequency
        rather than fitting against one wants :func:`spectral_frequencies`.

        Returns
        -------
        frequency
            a 1D numpy array of channel frequencies in MHz
        """
        return np.around(spectral_frequencies(self._header) / 1e6, decimals=7)

    def getAppendHeader(self, nchan):
        return self.spectralSplitHeader(nchan, orig="append_fits")

    def getTableHeader(self, nchan):
        self._header["NAXIS2"] = nchan
        self._header["NCHAN"] = nchan
        if "OBSERVER" in list(self._header.keys()):
            self._header.remove("OBSERVER")
        self._header["DATE"] = str(datetime.datetime.now()).replace(" ", "T")
        self._header["ORIGIN"] = "A. Kazemi-Moridani (table_header)"
        return self._header

    def getCombineHeader(self, dimx, dimy):
        self._header["NAXIS1"] = int(dimx)
        self._header["NAXIS2"] = int(dimy)
        # self._header['CRPIX1'] = xcen
        # self._header['CRPIX2'] = ycen
        if "OBSERVER" in list(self._header.keys()):
            self._header.remove("OBSERVER")
        self._header["DATE"] = str(datetime.datetime.now()).replace(" ", "T")
        self._header["ORIGIN"] = "A. Kazemi-Moridani (combine_spatial)"
        return self._header

    def getPrimeHeader(self, nchan, ydim, xdim, mask=False, orig="prime_header"):
        self._header["NAXIS1"] = int(xdim)
        self._header["NAXIS2"] = int(ydim)
        self._header["NAXIS3"] = int(nchan)
        if mask:
            self._header["BITPIX"] = 8
            if orig == "prime_header":
                orig = "mask_header"
        if "OBSERVER" in list(self._header.keys()):
            self._header.remove("OBSERVER")
        self._header["DATE"] = str(datetime.datetime.now()).replace(" ", "T")
        self._header["ORIGIN"] = f"A. Kazemi-Moridani ({orig})"
        return self._header

    def spectralSplitHeader(self, nchan, sfreq=None, orig="spectral_split"):
        self._header["NAXIS3"] = nchan
        if sfreq is not None:
            self._header["CRVAL3"] = sfreq
        if "OBSERVER" in list(self._header.keys()):
            self._header.remove("OBSERVER")
        self._header["DATE"] = str(datetime.datetime.now()).replace(" ", "T")
        self._header["ORIGIN"] = f"A. Kazemi-Moridani ({orig})"
        return self._header

    def spatialSplitHeader(self, xlims, ylims, chans=None):
        xdn, xup = xlims
        ydn, yup = ylims
        freq = self.retFreq()
        if chans is not None:
            chdn, chup = chans
            self._header["NAXIS3"] = chup - chdn  # make sure this is correct it was 'chup - chdn + 1' before
            self._header["CRVAL3"] = freq[chdn] * 1e6
        xor = self._header["CRPIX1"]
        yor = self._header["CRPIX2"]
        self._header["CRPIX1"] = xor - xdn
        self._header["CRPIX2"] = yor - ydn
        self._header["NAXIS1"] = xup - xdn
        self._header["NAXIS2"] = yup - ydn
        if "OBSERVER" in list(self._header.keys()):
            self._header.remove("OBSERVER")
        self._header["DATE"] = str(datetime.datetime.now()).replace(" ", "T")
        self._header["ORIGIN"] = "A. Kazemi-Moridani (spatial_split)"
        return self._header


def get_ds_from_msdsl(ms_dsl, field_id=0, data_desc_id=0):
    found_ds = False
    for ds in ms_dsl:
        if ds.FIELD_ID == field_id and ds.DATA_DESC_ID == data_desc_id:
            found_ds = True
            break
    if found_ds:
        return ds
    else:
        raise ValueError(f"Dataset with FIELD_ID={field_id} and DATA_DESC_ID={data_desc_id} not found in the MS.")


def observatory_location(ms_path):
    """Array reference position of an MS, as the centroid of its antennas.

    Args:
        ms_path (str|File): Measurement Set.

    Returns:
        EarthLocation: Array centre, from ``ANTENNA::POSITION`` (ITRF metres).
    """
    positions = xds_from_table(f"{ms_path}::ANTENNA")[0].POSITION.values
    centre = np.asarray(positions, dtype=float).mean(axis=0)

    return EarthLocation.from_geocentric(*centre, unit=units.m)


def source_systemic_velocity(ms_path, field_id=0):
    """Systemic radial velocity of a field, if the MS records one.

    Args:
        ms_path (str|File): Measurement Set.
        field_id (int): Field to look up.

    Returns:
        float|None: Velocity in m/s (positive for recession), or None when the
        MS has no SOURCE table, no SYSVEL column, or no matching source.
    """
    try:
        field = xds_from_table(f"{ms_path}::FIELD")[0]
        source_id = int(np.asarray(field.SOURCE_ID.values)[field_id])
        source = xds_from_table(f"{ms_path}::SOURCE")[0]
    except Exception:
        return None

    if "SYSVEL" not in source.data_vars:
        return None

    ids = np.asarray(source.SOURCE_ID.values)
    matches = np.flatnonzero(ids == source_id)
    if matches.size == 0:
        return None

    sysvel = np.atleast_1d(np.asarray(source.SYSVEL.values)[matches[0]])
    if sysvel.size == 0 or not np.isfinite(sysvel[0]):
        return None

    return float(sysvel[0])


#: Main-table columns that are per-channel and so cannot be carried across a
#: regrid unchanged. Everything else is copied to the output MS as it stands.
CHANNEL_COLUMNS = frozenset(
    {
        "DATA",
        "CORRECTED_DATA",
        "MODEL_DATA",
        "FLAG",
        "FLAG_CATEGORY",
        "WEIGHT_SPECTRUM",
        "SIGMA_SPECTRUM",
    }
)


def output_ms_dataset(msds, columns, spw_id, field_id):
    """Assemble a dataset describing a complete new MS main table.

    ``xds_to_table`` writes exactly the columns it is given, so anything omitted
    here is simply absent from the output MS. This carries every row-shaped
    column of the input across unchanged and restores the two grouping columns,
    leaving the caller to supply the per-channel ones.

    Args:
        msds (Dataset): dask-ms dataset for the field and spectral window being
            processed.
        columns (dict): Per-channel columns to write, as
            ``{name: (dims, dask_array)}``. These take precedence over anything
            of the same name in ``msds``.
        spw_id (int): Spectral window being processed.
        field_id (int): Field being processed.

    Returns:
        Dataset: Ready for ``xds_to_table(..., columns="ALL")``.
    """
    nrow = msds.TIME.shape[0]
    row_chunks = msds.TIME.data.chunks[0]

    variables = dict(columns)
    # dask-ms groups on these, so they arrive as attrs rather than columns and
    # would otherwise be missing from the output.
    variables.setdefault("FIELD_ID", (("row",), da.full(nrow, field_id, chunks=row_chunks, dtype=np.int32)))
    variables.setdefault("DATA_DESC_ID", (("row",), da.full(nrow, spw_id, chunks=row_chunks, dtype=np.int32)))

    for name, var in msds.data_vars.items():
        if name in CHANNEL_COLUMNS or name in variables or "chan" in var.dims:
            continue
        variables[name] = (var.dims, var.data)

    return Dataset(variables)


def copy_ms_subtables(input_ms, output_ms):
    """Copy an MS's subtables onto a freshly written main table.

    ``xds_to_table`` produces the main table columns but none of the subtables
    that make the result a readable MS, so they are copied over wholesale.

    Args:
        input_ms (str|File): MS the subtables are copied from.
        output_ms (str|File): MS to attach them to.
    """
    input_ms, output_ms = str(input_ms), str(output_ms)

    # Every open below asks for automatic locking. The casacore default is user
    # locking, under which these tables arrive unlocked -- via dask-ms's table
    # cache for the output, via the parent table for a subtable -- and both
    # copy() and putkeyword() then refuse to proceed.
    with table(input_ms, ack=False, lockoptions="auto") as source:
        keywords = source.getkeywords()

    with table(output_ms, readonly=False, ack=False, lockoptions="auto") as dest:
        for name, value in keywords.items():
            # Subtables appear as keywords of the form "Table: /path/to/sub".
            if not (isinstance(value, str) and value.startswith("Table:")):
                continue
            subtable = value.split("Table:", 1)[1].strip()
            copied = f"{output_ms}/{name}"
            with table(subtable, ack=False, lockoptions="auto") as sub:
                # copy() hands back an open handle to the new table. Left
                # dangling it stays in casacore's cache read-only, and any later
                # rewrite of that subtable would be refused.
                sub.copy(copied, deep=True, valuecopy=True).close()
            dest.putkeyword(name, f"Table: {copied}")

        dest.putkeyword("MS_VERSION", keywords.get("MS_VERSION", 2.0))


def require_topocentric(spw, spw_id, ms_path):
    """Refuse a spectral window that has already been moved off the topocentric grid.

    ``doppler_factors`` converts *from* topocentric frequency, so feeding it a
    window that CASA ``mstransform`` (or an earlier mowjsub run) has already
    regridded applies the correction twice. Fused into a continuum-subtraction
    run that cannot happen; reached through ``doppler-mowjsub`` it easily can,
    since the whole point there is to consume an MS someone else produced.

    Args:
        spw (Dataset): The MS ``SPECTRAL_WINDOW`` subtable.
        spw_id (int): Spectral window being processed.
        ms_path (str|File): MS the window came from, for the error message.

    Raises:
        RuntimeError: The window's ``MEAS_FREQ_REF`` is not topocentric.
    """
    # Not every MS carries the column; absent it, take the grid on trust.
    if "MEAS_FREQ_REF" not in spw.data_vars:
        return

    code = int(np.asarray(spw.MEAS_FREQ_REF.values)[spw_id])
    if code == FRAME_CODES["topo"]:
        return

    names = {value: key for key, value in FRAME_CODES.items()}
    found = names.get(code, f"code {code}")
    raise RuntimeError(
        f"Spectral window {spw_id} of {ms_path} is already on a {found.upper()} grid, not a topocentric one. "
        f"Doppler-correcting it would apply the frame conversion twice. Use the MS as it came off the telescope."
    )


def _regrid_block(data, flags, weights, times, utimes, ufactors, freqs_in, freqs_out, interpolation):
    """Regrid one row block, looking each row's Doppler factor up by timestamp."""
    factors = ufactors[np.clip(np.searchsorted(utimes, times), 0, ufactors.size - 1)]

    return regrid_rows(data, flags, weights, factors, freqs_in, freqs_out, interpolation)


def doppler_regrid_dataset(
    msds,
    ms_path,
    line_data,
    output_column,
    spw_id,
    field_id,
    frame,
    chan_grid="auto",
    interpolation="nearest",
    source_vel=None,
):
    """Resample continuum-subtracted visibilities onto a Doppler-corrected grid.

    Args:
        msds (Dataset): dask-ms dataset for the field and spectral window being
            processed, supplying the row metadata carried into the output.
        ms_path (str|File): Input MS, read for its channel frequencies, phase
            centre and antenna positions.
        line_data (dask.array): Continuum-subtracted visibilities on the input
            channel grid, shape ``(row, chan, corr)``.
        output_column (str): Column the regridded line data is written to.
        spw_id (int): Spectral window being processed.
        field_id (int): Field being processed.
        frame (str): Target spectral frame, one of :data:`mowjsub.doppler.FRAMES`.
        chan_grid (str): ``'auto'`` or an explicit ``'nchan,chan0,chanwidth'``.
        interpolation (str): ``'nearest'`` or ``'linear'``.
        source_vel (float, optional): Source systemic velocity in m/s, for
            ``frame='source'``. Falls back to ``SOURCE::SYSVEL``.

    Returns:
        tuple: ``(dataset, freqs_out, chanwidth)`` ready for ``xds_to_table``.
    """
    spw = xds_from_table(f"{ms_path}::SPECTRAL_WINDOW")[0]
    field = xds_from_table(f"{ms_path}::FIELD")[0]

    require_topocentric(spw, spw_id, ms_path)

    freqs_in = np.asarray(spw.CHAN_FREQ.values[spw_id], dtype=float)
    phase_centre = np.asarray(field.PHASE_DIR.values[field_id], dtype=float)[0]
    location = observatory_location(ms_path)

    if frame == "source" and source_vel is None:
        source_vel = source_systemic_velocity(ms_path, field_id)
        if source_vel is None:
            raise RuntimeError("--doppler-frame=source needs a systemic velocity, but this MS has no usable SOURCE::SYSVEL. Pass --doppler-source-vel.")
        log.info(f"Using SOURCE::SYSVEL = {source_vel / 1e3:.3f} km/s for the source-frame correction")

    # One factor per timestamp; every baseline and correlation at that timestamp
    # shares it, so the channel map is built once per distinct time.
    times = np.asarray(msds.TIME.data)
    utimes = np.unique(times)
    ufactors = doppler_factors(utimes, tuple(phase_centre), location, frame, source_vel=source_vel)

    drift = (ufactors.max() - ufactors.min()) * 299792458.0
    log.info(f"Doppler correction to {frame.upper()}: line-of-sight velocity drifts by {drift:.3f} m/s over the observation")

    if str(chan_grid).lower() == "auto":
        nchan, chan0, chanwidth = common_channel_grid(freqs_in, ufactors)
        log.info(f"Derived a common {frame.upper()} channel grid: {nchan} channels from {chan0 / 1e6:.4f} MHz in steps of {chanwidth / 1e3:.4f} kHz")
    else:
        nchan, chan0, chanwidth = parse_channel_grid(chan_grid)
        log.info(f"Using the requested {frame.upper()} channel grid: {nchan} channels from {chan0 / 1e6:.4f} MHz in steps of {chanwidth / 1e3:.4f} kHz")

    freqs_out = grid_frequencies(nchan, chan0, chanwidth)

    # blockwise yields one tuple per block; map_blocks then unpacks the three
    # regridded arrays without recomputing the fit.
    regridded = da.blockwise(
        _regrid_block,
        "rco",
        line_data,
        "rio",
        msds.FLAG.data,
        "rio",
        msds.WEIGHT_SPECTRUM.data,
        "rio",
        msds.TIME.data,
        "r",
        utimes=utimes,
        ufactors=ufactors,
        freqs_in=freqs_in,
        freqs_out=freqs_out,
        interpolation=interpolation,
        dtype=object,
        concatenate=True,
        new_axes={"c": nchan},
        adjust_chunks={"c": nchan},
    )

    columns = {
        output_column: (("row", "chan", "corr"), regridded.map_blocks(lambda block: block[0], dtype=line_data.dtype)),
        "FLAG": (("row", "chan", "corr"), regridded.map_blocks(lambda block: block[1], dtype=bool)),
        "WEIGHT_SPECTRUM": (("row", "chan", "corr"), regridded.map_blocks(lambda block: block[2], dtype=msds.WEIGHT_SPECTRUM.dtype)),
    }

    return output_ms_dataset(msds, columns, spw_id, field_id), freqs_out, chanwidth


def finalise_regridded_ms(input_ms, output_ms, spw_id, freqs, chanwidth, frame):
    """Attach subtables to a freshly written MS and rewrite its spectral window.

    The main table is written by ``xds_to_table``, which produces the columns but
    none of the subtables an MS needs. This copies those across from the input
    and then overwrites the processed spectral window so it describes the
    regridded channel grid in the new reference frame.

    Args:
        input_ms (str|File): MS the subtables are copied from.
        output_ms (str|File): MS to finalise, already carrying its main table.
        spw_id (int): Row of ``SPECTRAL_WINDOW`` that was regridded.
        freqs (np.ndarray): Output channel frequencies in Hz.
        chanwidth (float): Output channel width in Hz; negative if descending.
        frame (str): Target spectral frame, used to set ``MEAS_FREQ_REF``.
    """
    output_ms = str(output_ms)
    freqs = np.asarray(freqs, dtype=float)

    copy_ms_subtables(input_ms, output_ms)

    nchan = freqs.size
    widths = np.full(nchan, chanwidth, dtype=float)

    # Address the subtable by path, not as "<ms>::SPECTRAL_WINDOW": the latter
    # reaches it through the parent table and inherits the parent's access mode,
    # which leaves the columns read-only here.
    with table(f"{output_ms}/SPECTRAL_WINDOW", readonly=False, ack=False, lockoptions="auto") as spw:
        spw.putcell("NUM_CHAN", spw_id, nchan)
        spw.putcell("CHAN_FREQ", spw_id, freqs)
        spw.putcell("CHAN_WIDTH", spw_id, widths)
        spw.putcell("RESOLUTION", spw_id, np.abs(widths))
        spw.putcell("EFFECTIVE_BW", spw_id, np.abs(widths))
        spw.putcell("REF_FREQUENCY", spw_id, float(freqs[0]))
        spw.putcell("TOTAL_BANDWIDTH", spw_id, float(abs(chanwidth) * nchan))
        spw.putcell("MEAS_FREQ_REF", spw_id, FRAME_CODES[frame])

    log.info(f"Wrote {nchan} channels to {output_ms} on a {frame.upper()} grid, from {freqs[0] / 1e6:.4f} MHz in steps of {chanwidth / 1e3:.4f} kHz")


#: Everything needed to move a cube onto a Doppler-corrected grid. Derived once
#: and applied to both output cubes, so the continuum and line necessarily land
#: on the same channels.
CubeDopplerPlan = namedtuple("CubeDopplerPlan", "frame factor freqs_in freqs_out chanwidth axis drift")


def cube_spectral_axis(header):
    """FITS axis number (0-based) of a cube's spectral axis.

    Raises:
        RuntimeError: The cube has no spectral axis, or has one that mowjsub
            cannot process.
    """
    spec = WCS(header).wcs.spec
    if spec < 0:
        raise RuntimeError("This cube has no spectral axis, so there is nothing to Doppler-correct.")

    # A stricter check than either the Doppler code or the spectral grid needs:
    # both derive the axis from the WCS and are order-agnostic. What is not is
    # the continuum write-back in im_mowjsub.runit, which transposes to a fixed
    # (2, 1, 0) and puts Stokes outermost. Until that places its axes by WCS
    # too, a cube with the spectral axis elsewhere would fail confusingly on the
    # way out rather than here. Say so up front instead.
    if spec != 2:
        raise RuntimeError(f"mowjsub needs the spectral axis on NAXIS3; this cube has it on NAXIS{spec + 1}. Reorder the axes (e.g. with CASA imtrans) and try again.")

    return spec


def _cube_phase_centre(header, override=None):
    """Field centre of a cube in radians, from the WCS or an explicit override."""
    if override:
        parts = [part.strip() for part in str(override).split(",")]
        if len(parts) != 2:
            raise ValueError(f"Could not parse phase centre '{override}'. Expected 'RA,Dec', e.g. '13h25m27.6s,-43d01m09s' or '201.365,-43.019'.")
        try:
            coord = SkyCoord(float(parts[0]) * units.deg, float(parts[1]) * units.deg)
        except ValueError:
            coord = SkyCoord(parts[0], parts[1], unit=(units.hourangle, units.deg))
        return coord.ra.rad, coord.dec.rad

    ra, dec = WCS(header).celestial.wcs.crval

    return np.radians(float(ra)), np.radians(float(dec))


def _cube_time(header, override=None):
    """Observation epoch of a cube, in seconds since MJD 0 UTC (the MS convention)."""
    if override:
        try:
            epoch = Time(float(override), format="mjd", scale="utc")
        except ValueError:
            epoch = Time(str(override), scale="utc")
        return float(epoch.mjd) * 86400.0

    # Test for a usable value, not merely a present keyword: astropy's WCS
    # handling leaves an undefined MJD-OBS card behind, and imagers write blank
    # ones too, so `"MJD-OBS" in header` is not the same question.
    mjd = header.get("MJD-OBS")
    if isinstance(mjd, (int, float)) and np.isfinite(mjd):
        return float(Time(float(mjd), format="mjd", scale="utc").mjd) * 86400.0

    date = header.get("DATE-OBS")
    if isinstance(date, str) and date.strip():
        return float(Time(date.strip(), scale="utc").mjd) * 86400.0

    raise RuntimeError("This cube's header records no usable observation time (no MJD-OBS or DATE-OBS), so the Doppler shift cannot be computed. Pass --doppler-time.")


def _cube_location(header, override=None):
    """Observatory position for a cube, from OBSGEO-*, TELESCOP or an override."""
    if override:
        try:
            return EarthLocation.of_site(str(override))
        except Exception as exc:
            raise RuntimeError(f"Could not resolve --doppler-telescope='{override}' to a position: {exc}. Give the site in the cube's OBSGEO-X/Y/Z keywords instead.") from None

    if all(f"OBSGEO-{axis}" in header for axis in "XYZ"):
        return EarthLocation.from_geocentric(
            float(header["OBSGEO-X"]),
            float(header["OBSGEO-Y"]),
            float(header["OBSGEO-Z"]),
            unit=units.m,
        )

    if "TELESCOP" in header:
        name = str(header["TELESCOP"]).strip()
        try:
            # Consults astropy's site registry, which may need to be fetched.
            return EarthLocation.of_site(name)
        except Exception:
            raise RuntimeError(
                f"The cube names TELESCOP='{name}', but that is not in astropy's site registry (or it could not be reached). "
                f"Pass --doppler-telescope, or add OBSGEO-X/Y/Z to the header."
            ) from None

    raise RuntimeError("This cube's header records no observatory position (no OBSGEO-X/Y/Z and no TELESCOP), so the Doppler shift cannot be computed. Pass --doppler-telescope.")


def plan_cube_doppler(
    header,
    frame,
    chan_grid="auto",
    obs_time=None,
    obs_duration=None,
    telescope=None,
    phase_centre=None,
    source_vel=None,
):
    """Work out the Doppler correction for a cube, without applying it.

    A cube has already been integrated over time, so unlike the visibility-plane
    path this produces a *single* factor -- taken at the midpoint of the
    observation when its length is known. That is only equivalent to the
    per-timestamp correction when the observatory's line-of-sight velocity
    drifts by much less than a channel over the track; imaging has already summed
    over any larger drift, and shifting the axis afterwards cannot undo it. When
    ``obs_duration`` is given the drift is computed and warned about; when it is
    not, that check is skipped and the caller is told so.

    Args:
        header: FITS header of the cube.
        frame (str): Target frame, one of :data:`mowjsub.doppler.FRAMES`.
        chan_grid (str): ``'auto'`` or an explicit ``'nchan,chan0,chanwidth'``.
        obs_time (str, optional): Observation epoch, ISO or MJD, overriding
            ``MJD-OBS``/``DATE-OBS``.
        obs_duration (float, optional): Length of the observation in hours. Only
            used to check the drift.
        telescope (str, optional): Site name, overriding ``OBSGEO-*``/``TELESCOP``.
        phase_centre (str, optional): ``'RA,Dec'``, overriding the WCS.
        source_vel (float, optional): Source systemic velocity in m/s, required
            for ``frame='source'``.

    Returns:
        CubeDopplerPlan
    """
    axis = cube_spectral_axis(header)

    centre = _cube_phase_centre(header, phase_centre)
    location = _cube_location(header, telescope)
    epoch = _cube_time(header, obs_time)

    # Hz throughout, which is what everything in doppler.py that is not
    # scale-free (parse_channel_grid) works in. This used to go through retFreq,
    # which meant converting from MHz and stamping the resolved epoch into a
    # header copy first -- the high-level WCS it used insists on a usable
    # observation time even to turn pixels into frequencies. Neither is needed
    # now; the epoch below is for the Doppler factor alone.
    freqs_in = spectral_frequencies(header)

    if frame == "source" and source_vel is None:
        raise RuntimeError("--doppler-frame=source needs a systemic velocity, and a cube carries none. Pass --doppler-source-vel.")

    span = None if obs_duration in (None, 0) else float(obs_duration) * 3600.0
    midpoint = epoch if span is None else epoch + 0.5 * span
    factor = float(doppler_factors([midpoint], centre, location, frame, source_vel=source_vel)[0])

    if str(chan_grid).lower() == "auto":
        nchan, chan0, chanwidth = common_channel_grid(freqs_in, [factor])
        log.info(f"Derived a {frame.upper()} channel grid: {nchan} channels from {chan0 / 1e6:.4f} MHz in steps of {chanwidth / 1e3:.4f} kHz")
    else:
        nchan, chan0, chanwidth = parse_channel_grid(chan_grid)
        log.info(f"Using the requested {frame.upper()} channel grid: {nchan} channels from {chan0 / 1e6:.4f} MHz in steps of {chanwidth / 1e3:.4f} kHz")

    freqs_out = grid_frequencies(nchan, chan0, chanwidth)

    drift = None
    if span is not None:
        edges = doppler_factors([epoch, epoch + span], centre, location, frame, source_vel=source_vel)
        drift = float(abs(edges.max() - edges.min()) * C)

        # The comparison that decides whether a cube-plane correction is
        # defensible at all. Both quantities scale with frequency, so the ratio
        # does not depend on where in the band it is evaluated.
        chan_velocity = abs(chanwidth) / (0.5 * (freqs_out[0] + freqs_out[-1])) * C
        log.info(f"Doppler correction to {frame.upper()}: line-of-sight velocity drifts by {drift:.3f} m/s over {obs_duration:g} h, against a {chan_velocity:.3f} m/s channel")
        if drift > 0.1 * chan_velocity:
            log.warning(
                f"That drift is {drift / chan_velocity:.2f} of a channel. A cube has already been integrated over time, so this correction applies one factor to all of it "
                f"and cannot undo the smearing the drift has already caused. Correct in the visibility plane instead (doppler-mowjsub, or vis-mowjsub --doppler-frame)."
            )
    else:
        log.info(f"Doppler correction to {frame.upper()}: factor {factor:.9f} at the observation epoch. Pass --doppler-obs-duration to check the drift over the track.")

    return CubeDopplerPlan(frame=frame, factor=factor, freqs_in=freqs_in, freqs_out=freqs_out, chanwidth=chanwidth, axis=axis, drift=drift)


def apply_cube_doppler(data, header, plan, interpolation="nearest"):
    """Resample a cube onto the grid in ``plan`` and rewrite its spectral WCS.

    Args:
        data (np.ndarray|dask.array): Cube in FITS axis order.
        header: FITS header to base the output header on.
        plan (CubeDopplerPlan): From :func:`plan_cube_doppler`.
        interpolation (str): ``'nearest'`` or ``'linear'``.

    Returns:
        tuple: ``(data, header)`` on the corrected grid.
    """
    # plan.axis is a FITS axis index; the array is in C order, i.e. reversed.
    numpy_axis = np.ndim(data) - 1 - plan.axis

    resampled = resample_cube(data, plan.factor, plan.freqs_in, plan.freqs_out, numpy_axis, interpolation)

    out = fitsio.Header(header)
    keyword = plan.axis + 1
    out[f"NAXIS{keyword}"] = plan.freqs_out.size
    out[f"CRPIX{keyword}"] = 1.0
    out[f"CRVAL{keyword}"] = float(plan.freqs_out[0])
    out[f"CDELT{keyword}"] = float(plan.chanwidth)
    out[f"CUNIT{keyword}"] = "Hz"
    out["SPECSYS"] = FITS_SPECSYS[plan.frame]

    # A CD matrix takes precedence over CDELT, and a PC matrix multiplies it, so
    # a stale entry on the spectral axis would silently undo the line above.
    if f"CD{keyword}_{keyword}" in out:
        out[f"CD{keyword}_{keyword}"] = float(plan.chanwidth)
    if f"PC{keyword}_{keyword}" in out:
        out[f"PC{keyword}_{keyword}"] = 1.0

    # Velocity-referenced duplicates of the spectral axis. They describe the old
    # grid, and a reader that trusts them over CRVAL3 would be silently wrong.
    for stale in ("ALTRVAL", "ALTRPIX", "VELREF"):
        out.pop(stale, None)

    return resampled, out


def ms_to_xarray_dataset(
    ms_path,
    spw_id: int,
    field_id: int,
    chunks: int,
    outchunks={"time": 64, "baseline": 64},
    save_to_zarr=False,
):
    """Creates Zarr store from a input MS. The resulting array has
    dimensions = time, basline, SPECTRAL, corr

    Args:
        ms path (str|path): MS file_
        spw_id (int): Spectral window ID
        field_id (int): Field ID
        chunks (int): How to chunk the data
        outchunks (dict, optional): xarray chunk object. Defaults to {'time': 64, 'baseline': 64}.
        save_to_zarr (bool, optional): Save the output to Zarr. Defaults to False.
    Returns:
        Zarr: Zarr array (persistant store, mode=w)
    """
    ms_dsl = xds_from_ms(
        ms_path,
        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": chunks},
    )

    spw_table = xds_from_table(f"{ms_path}::SPECTRAL_WINDOW")[0]
    field_table = xds_from_table(f"{ms_path}::FIELD")[0]
    antenna_table = xds_from_table(f"{ms_path}::ANTENNA")[0]

    frequencies = spw_table.CHAN_FREQ.data[spw_id]
    channel_width = spw_table.CHAN_WIDTH.data[spw_id][0]
    ref_frequency = spw_table.REF_FREQUENCY.data[spw_id]

    phase_center = field_table.PHASE_DIR.data[field_id].compute()[0]

    antenna_names = [name.strip() for name in antenna_table.NAME.data.compute()]
    ds = get_ds_from_msdsl(ms_dsl, field_id=field_id, data_desc_id=spw_id)

    times = ds.TIME.data
    visibilities = ds.DATA.data
    flags = ds.FLAG.data
    weights = ds.WEIGHT_SPECTRUM.data
    uvw = ds.UVW.data

    nant = antenna_table.NAME.size
    nbl = nant * (nant - 1) // 2
    nrow, nchan, ncorr = ds.DATA.shape
    ntimes = nrow // nbl

    unique_times = np.unique(times)

    reshaped_vis = da.reshape(visibilities, (ntimes, nbl, nchan, ncorr))
    reshaped_flags = da.reshape(flags, (ntimes, nbl, nchan, ncorr))
    reshaped_weights = da.reshape(weights, (ntimes, nbl, nchan, ncorr))

    if ncorr == 2:
        corr_labels = ["XX", "YY"][:ncorr]
    else:
        corr_labels = ["XX", "XY", "YX", "YY"][:ncorr]

    dataset = xr.Dataset(
        {  # TO-DO: reshape the data for all
            "VIS": (
                ["time", "baseline", "spectral", "corr"],
                reshaped_vis,
            ),  # time here will be time indicies
            "FLAG": (["time", "baseline", "spectral", "corr"], reshaped_flags),
            "WEIGHT": (["time", "baseline", "spectral", "corr"], reshaped_weights),
            "UVW": (("time", "baseline", "uvw"), uvw.reshape(ntimes, nbl, 3)),
        },
        coords={
            "FREQ": frequencies,
            "CORR": corr_labels,
            "TIME": unique_times,
            "BASELINE": np.arange(nbl),
        },
        attrs={
            "ref_freq": float(ref_frequency),
            "channel_width": float(channel_width),
            "phase_center_ra": float(phase_center[0]),  # radians
            "phase_center_dec": float(phase_center[1]),  # radians
            "phase_center_ra_deg": float(np.degrees(phase_center[0])),  # degrees
            "phase_center_dec_deg": float(np.degrees(phase_center[1])),  # degrees
            "antenna_names": antenna_names,
            "nant": nant,
            "DATA_DESC_ID": spw_id,
            "FIELD_ID": field_id,
        },
    )
    dataset = dataset.chunk(outchunks)
    if not save_to_zarr:
        return dataset
    else:
        outpath = "tmp.zarr"
        write_to_zarr = dataset.to_zarr(outpath, mode="w", compute=False)

    with TqdmCallback(desc="Writing to Zarr"):
        da.compute(write_to_zarr)

    return outpath
