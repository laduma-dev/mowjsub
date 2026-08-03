import shutil
import tempfile
import unittest
from pathlib import Path

import astropy.io.fits as fitsio
import dask
import numpy as np
from scabha import init_logger

from mowjsub import utils
from mowjsub.fitfuncs import (
    FitBSpline,
    FitGCVSpline,
    FitMedFilter,
    FitMedFilterFast,
    FitPolynomial,
    # FitDCT,
)

log = init_logger("mowjsub")


class TestFitsFunc(unittest.TestCase):
    def setUp(self):
        self.nchan = nchan = 1000
        xvals = np.linspace(0, 2 * np.pi, nchan)
        noise = np.random.randn(nchan) * 1.5

        # --- simulate line profile ---
        nterm = 3
        amps = np.random.uniform(0.1, 0.3, nterm)
        shifts = np.random.uniform(0, np.pi / 4, nterm)
        omegas = 2 * np.pi * np.random.uniform(0.2, 1, nterm)

        line = np.sum(
            [amp * np.sin(omega * xvals + shift) for amp, omega, shift in zip(amps, omegas, shifts)],
            axis=0,
        )
        line /= amps.sum()

        self.data = data = line + noise

        dfreq = 6500 * 1e-6
        freq0 = 1361
        self.freqs = freqs = freq0 + np.linspace(0, nchan * dfreq, nchan)

        nans = tuple(set(np.random.randint(20, 80, 10)))
        data[(nans,)] = np.nan
        # ---

        self.mask = mask = np.zeros(nchan, dtype=bool)
        # ---- and flags
        mask_start = np.random.randint(10, nchan - 10)
        mask_end = mask_start + 20
        mask[mask_start:mask_end] = True

        mask_start = np.random.randint(10, nchan - 10)
        mask_end = mask_start + 40
        mask[mask_start:mask_end] = True
        # ----

        self.velwidth = 300
        self.chanwidth = utils.chans_in_velwidth(freqs * 1e6, self.velwidth * 1e3)
        # Fixed knot seed, so a fit is reproducible within a test.
        self.seed = 20260726

    def test_median_filter(self):
        baseline_func = FitMedFilter(self.freqs, velwidth=self.velwidth)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)

        assert baseline.shape == self.data.shape

    def test_median_filter_fast(self):
        baseline_func = FitMedFilterFast(self.freqs, velwidth=self.velwidth)
        baseline_vel = baseline_func.fit(self.data, mask=self.mask, weights=None)
        assert baseline_vel.shape == self.data.shape

        # test if chanwidth gives same result as velwidth
        baseline_func = FitMedFilterFast(self.freqs, chanwidth=self.chanwidth)
        baseline_chan = baseline_func.fit(self.data, mask=self.mask, weights=None)

        assert np.allclose(baseline_vel, baseline_chan, atol=1e-6)

    def test_polynomial(self):
        baseline_func = FitPolynomial(self.freqs, order=3)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)

        assert baseline.shape == self.data.shape

    def test_b_spline(self):
        # Both fitters get the same seed. FitBSpline jitters its knots by up to
        # +/-25 channels, so without this the two fits differ by knot placement
        # rather than by anything the test is asking about -- which is what kept
        # pushing this tolerance up: the residual MAD ran to ~0.05 on ~8% of
        # datasets, failing whichever CI matrix entry drew one.
        baseline_func = FitBSpline(self.freqs, order=3, velwidth=self.velwidth, seed=self.seed)
        baseline_vel = baseline_func.fit(self.data, mask=self.mask, weights=None)

        # test if chanwidth gives same result as velwidth
        baseline_func = FitBSpline(self.freqs, order=3, chanwidth=self.chanwidth, seed=self.seed)
        baseline_chan = baseline_func.fit(self.data, mask=self.mask, weights=None)

        assert baseline_chan.shape == self.data.shape

        resid = baseline_chan - baseline_vel
        resid_median = np.median(resid)
        mad = np.median(np.abs(resid - resid_median))

        # With knots held fixed the two paths must agree exactly: velwidth is
        # converted by the same `chans_in_velwidth` the test used to derive
        # chanwidth, so this is an equality check, not a tolerance.
        np.testing.assert_array_equal(baseline_chan, baseline_vel)
        assert mad == 0.0

    def test_gcv_spline(self):
        baseline_func = FitGCVSpline(self.freqs)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)

        assert baseline.shape == self.data.shape


class TestZdsFromFits(unittest.TestCase):
    """The reader's contract with the rest of the image plane."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _header(self, nchan=8, npix=6, spectral_axis=3):
        header = fitsio.Header()
        header["CTYPE1"], header["CRVAL1"], header["CDELT1"], header["CRPIX1"], header["CUNIT1"] = "RA---SIN", 20.0, -1e-3, 1, "deg"
        header["CTYPE2"], header["CRVAL2"], header["CDELT2"], header["CRPIX2"], header["CUNIT2"] = "DEC--SIN", -30.0, 1e-3, 1, "deg"
        spectral = {"CTYPE": "FREQ", "CRVAL": 1.4e9, "CDELT": 1e6, "CRPIX": 1.0, "CUNIT": "Hz"}
        stokes = {"CTYPE": "STOKES", "CRVAL": 1.0, "CDELT": 1.0, "CRPIX": 1.0, "CUNIT": ""}
        trailing = [(stokes, 1), (spectral, nchan)] if spectral_axis == 4 else [(spectral, nchan), (stokes, 1)]
        for index, (axis, length) in enumerate(trailing, start=3):
            for key, value in axis.items():
                header[f"{key}{index}"] = value
            header[f"NAXIS{index}"] = length
        header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 4, npix, npix
        return header

    def _write(self, path, nchan=8, npix=6, spectral_axis=3, in_extension=False):
        header = self._header(nchan, npix, spectral_axis)
        shape = (nchan, 1, npix, npix) if spectral_axis == 4 else (1, nchan, npix, npix)
        data = np.random.default_rng(2).normal(size=shape).astype(np.float32)
        if in_extension:
            hdus = [fitsio.PrimaryHDU(), fitsio.ImageHDU(data, header=header, name="SCI")]
        else:
            hdus = [fitsio.PrimaryHDU(data, header=header)]
        fitsio.HDUList(hdus).writeto(path, overwrite=True)
        return data

    def test_the_spectral_axis_is_never_split_across_chunks(self):
        """ContSub fits whole spectra; a split spectral axis would truncate them.

        get_xds leaves unlisted axes at the file's own chunking, which is not
        necessarily one chunk, so the reader has to say so explicitly.
        """
        path = self.tmpdir / "cube.fits"
        self._write(path, nchan=64, npix=32)

        # Small enough that the file-shaped chunking would split the spectral
        # axis; at the default a test cube fits in one chunk and proves nothing.
        with dask.config.set({"array.chunk-size": "8kiB"}):
            zds = utils.zds_from_fits(path, chunks=dict(ra=8, dec=None, spectral=None))

        chunks = dict(zip(zds.DATA.dims, zds.DATA.chunks))
        assert len(chunks["spectral"]) == 1, "spectral must be a single chunk"
        assert len(chunks["dec"]) == 1
        assert max(chunks["ra"]) == 8

    def test_axes_are_matched_by_name_not_position(self):
        """A STOKES/FREQ cube reads into the same dims as a FREQ/STOKES one."""
        normal, swapped = self.tmpdir / "n.fits", self.tmpdir / "s.fits"
        self._write(normal, nchan=8, spectral_axis=3)
        self._write(swapped, nchan=8, spectral_axis=4)

        for path, native in ((normal, ("stokes", "spectral", "dec", "ra")), (swapped, ("spectral", "stokes", "dec", "ra"))):
            zds = utils.zds_from_fits(path)
            assert zds.DATA.dims == ("ra", "dec", "spectral", "stokes")
            assert zds.DATA.sizes["spectral"] == 8
            assert zds.DATA.sizes["stokes"] == 1
            # fits_dims is what the write-back places axes by.
            assert zds.attrs["fits_dims"] == native

    def test_the_cube_can_live_in_an_extension(self):
        """--hdu-index has to keep working now the reader is fitstoolz's."""
        path = self.tmpdir / "mef.fits"
        data = self._write(path, nchan=8, in_extension=True)

        zds = utils.zds_from_fits(path, hdu_idx=1)

        assert zds.DATA.sizes["spectral"] == 8
        np.testing.assert_allclose(np.asarray(zds.DATA.transpose("stokes", "spectral", "dec", "ra").data), data, rtol=1e-6)

    def test_a_cube_with_no_spectral_axis_is_refused(self):
        path = self.tmpdir / "flat.fits"
        header = self._header()
        for key in ("CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3", "NAXIS3", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4", "NAXIS4"):
            header.pop(key, None)
        header["NAXIS"] = 2
        fitsio.PrimaryHDU(np.zeros((6, 6), np.float32), header=header).writeto(path)

        with self.assertRaises(RuntimeError) as raised:
            utils.zds_from_fits(path)

        assert "spectral axis" in str(raised.exception)

    def test_rest_freq_reaches_the_header_the_outputs_inherit(self):
        path = self.tmpdir / "rf.fits"
        self._write(path)

        zds = utils.zds_from_fits(path, rest_freq=1420.40575)

        assert zds.attrs["header"]["RESTFREQ"] == 1420.40575 * 1e6
