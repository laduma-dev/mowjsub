import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import astropy.io.fits as fitsio
import dask
import dask.array as da
import numpy as np

from mowjsub import utils
from mowjsub.exceptions import BadFitError
from mowjsub.fitfuncs import (
    FitBSpline,
    FitGCVSpline,
    FitMedFilter,
    FitMedFilterFast,
    FitPolynomial,
    # FitDCT,
)


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

    def test_median_filter_fast_accepts_a_big_endian_spectrum(self):
        """FITS is big-endian, and scipy's rank filters take only native order.

        `zds_from_fits` hands back the cube in the order the file stores it, so
        on every little-endian machine this fitter was given a `>f4` spectrum
        and scipy raised a bare `RuntimeError: Unsupported array type`, naming
        neither the array nor the reason. `--fit-model scipy-median-filter`
        could therefore not run in the image plane at all. The visibility plane
        never saw it: dask-ms yields native arrays.
        """
        big_endian = self.data.astype(self.data.dtype.newbyteorder(">"))
        assert not big_endian.dtype.isnative, "this test needs a non-native array to be about anything"

        baseline_native = FitMedFilterFast(self.freqs, velwidth=self.velwidth).fit(self.data, mask=self.mask, weights=None)
        baseline_big = FitMedFilterFast(self.freqs, velwidth=self.velwidth).fit(big_endian, mask=self.mask, weights=None)

        np.testing.assert_allclose(baseline_big, baseline_native)

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


class TestBSplineKnotBounds(unittest.TestCase):
    """Knot counts and positions FITPACK will actually accept.

    ``--vel-width`` is a physical width, so the number of spline segments it
    asks for depends on how coarsely the cube is channelised. On a cube whose
    channels are already a large fraction of that width it approaches one
    segment per channel, which pushes the jittered knots past both of FITPACK's
    limits: they must lie strictly inside the data range, and there can be at
    most ``m - k - 1`` of them. Neither was enforced, and FITPACK reports the
    violation as a bare ``TypeError: An error occurred`` from inside Fortran
    that escaped ``ContSub.fitContinuum`` and killed the whole run.
    """

    def setUp(self):
        rng = np.random.default_rng(20260804)
        self.nchan = nchan = 64
        # 1 MHz channels at 1.42 GHz: ~216 km/s each, so a 250 km/s vel-width
        # is barely more than one channel. This is the case that failed.
        self.freqs = 1420.0 - np.arange(nchan) * 1.0
        self.data = 2.0 + 0.001 * np.arange(nchan) + 0.01 * rng.normal(size=nchan)
        self.mask = np.zeros(nchan, dtype=bool)

    def test_a_velwidth_of_about_one_channel_still_fits(self):
        fitter = FitBSpline(self.freqs, order=3, velwidth=250, seed=7)
        fitter.prepare()

        assert fitter.chanwidth == 1, "the parameterisation this test is about"

        baseline = fitter.fit(self.data.copy(), self.mask.copy(), None)

        assert baseline.shape == self.data.shape
        assert np.isfinite(baseline).all()

    def test_more_segments_than_channels_is_thinned_not_refused(self):
        """A chanwidth of 1 asks for one knot per channel; FITPACK allows m-k-1."""
        fitter = FitBSpline(self.freqs, order=3, chanwidth=1, seed=7)
        fitter.prepare()

        assert fitter.max_spline_order > self.nchan, "more segments requested than there are points"

        baseline = fitter.fit(self.data.copy(), self.mask.copy(), None)

        assert np.isfinite(baseline).all()

    def test_the_whole_parameter_grid_fits_or_reports_a_bad_fit(self):
        """Nothing may escape as a raw FITPACK error, whatever the parameters.

        Masking matters here: the knot bounds are set by the *valid* point
        count, not the channel count, so a heavily masked spectrum tightens them
        without changing anything the caller passed.
        """
        for order in (1, 2, 3, 5):
            for chanwidth in (1, 2, 3, 11, self.nchan // 2, self.nchan, self.nchan + 5):
                for masked in (0, 20, 40):
                    mask = self.mask.copy()
                    mask[:masked] = True

                    fitter = FitBSpline(self.freqs, order=order, chanwidth=chanwidth, seed=7)
                    fitter.prepare()
                    try:
                        baseline = fitter.fit(self.data.copy(), mask, None)
                    except BadFitError:
                        # The intended signal for a spectrum too sparse to fit;
                        # ContSub turns it into NaN for that pixel.
                        continue

                    assert np.isfinite(baseline).all(), (order, chanwidth, masked)

    def test_an_ordinary_cube_is_nowhere_near_the_bounds(self):
        """The clamps must be inert on real parameterisations, so no fit changes.

        A 26 kHz MeerKAT channel against a 250 km/s window is ~4 segments over
        1000 channels -- three orders of magnitude clear of the m-k-1 limit.
        """
        freqs = 1361.0 + np.arange(1000) * 0.026
        fitter = FitBSpline(freqs, order=3, velwidth=250, seed=7)
        fitter.prepare()

        knots = np.linspace(0, 1000, fitter.max_spline_order, dtype=int)[1:-1]

        assert knots.size < 1000 - 3 - 1
        assert knots.max() < 1000 - 2


class TestDescendingFrequencyGrid(unittest.TestCase):
    """A spectral axis that runs downwards in frequency.

    Ordinary FITS: any cube with a negative ``CDELT`` on the spectral axis, and
    unavoidable for one whose axis is a velocity or a wavelength, since both run
    opposite to frequency. scipy's spline fitters require strictly increasing x,
    so both spline models were broken on such a cube -- ``b-spline`` died with a
    bare ``ValueError: Error on input data`` that took the whole run with it, and
    ``gcv-spline`` failed per spectrum, which ``ContSub.fitContinuum`` turned
    into an all-NaN cube and no error at all.

    The same physical spectrum stored in the opposite direction must give the
    same continuum, so each fitter is checked against its own ascending result.
    """

    def setUp(self):
        rng = np.random.default_rng(20260804)
        self.nchan = nchan = 128
        self.freqs = 1400.0 + np.arange(nchan) * 1.0
        self.data = 2.0 + 0.001 * np.arange(nchan) + 0.01 * rng.normal(size=nchan)
        self.weights = 1.0 + rng.random(nchan)
        self.mask = np.zeros(nchan, dtype=bool)

    def _fitters(self, freqs):
        return {
            "b-spline": FitBSpline(freqs, order=3, velwidth=2000, seed=7),
            "gcv-spline": FitGCVSpline(freqs, fit_lam=1e-3),
            "polynomial": FitPolynomial(freqs, order=2),
            "median-filter": FitMedFilterFast(freqs, velwidth=2000),
        }

    def _fit(self, freqs, data, weights, mask):
        results = {}
        for name, fitter in self._fitters(freqs).items():
            fitter.prepare()
            results[name] = fitter.fit(data.copy(), mask=mask.copy(), weights=weights)

        return results

    def _both_directions(self, weights=None, mask=None):
        """Fit the same physical spectrum stored each way round.

        Everything channel-indexed reverses together -- data, weights and mask
        -- so the two runs differ only in which end of the band comes first.
        """
        mask = self.mask if mask is None else mask

        ascending = self._fit(self.freqs, self.data, weights, mask)
        descending = self._fit(
            self.freqs[::-1],
            self.data[::-1],
            None if weights is None else weights[::-1],
            mask[::-1],
        )

        return ascending, descending

    def test_every_fitter_matches_its_ascending_result(self):
        ascending, descending = self._both_directions()

        for name in ascending:
            np.testing.assert_allclose(descending[name][::-1], ascending[name], rtol=1e-12, atol=1e-12, err_msg=name)

    def test_weights_are_reordered_with_the_data(self):
        """The weight vector is sliced by the same mask, so it has to travel with it."""
        ascending, descending = self._both_directions(weights=self.weights)

        for name in ascending:
            np.testing.assert_allclose(descending[name][::-1], ascending[name], rtol=1e-12, atol=1e-12, err_msg=name)

    def test_a_masked_spectrum_still_fits(self):
        """Masking leaves x descending but no longer evenly spaced."""
        mask = self.mask.copy()
        mask[10:20] = True
        mask[70:75] = True

        ascending, descending = self._both_directions(mask=mask)

        for name in ("b-spline", "gcv-spline"):
            assert np.isfinite(descending[name]).all(), name
            np.testing.assert_allclose(descending[name][::-1], ascending[name], rtol=1e-12, atol=1e-12, err_msg=name)

    def test_an_ascending_grid_is_left_alone(self):
        """`ascending` must be a no-op on the grid every existing cube already has."""
        fitter = FitBSpline(self.freqs, order=3, velwidth=2000, seed=7)
        x, data, weights = fitter.ascending(self.freqs, self.data, self.weights)

        assert x is self.freqs
        assert data is self.data
        assert weights is self.weights


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

    def _unnamed_fourth_axis(self, path, ctype=None, length=1):
        """The cube from github issue #31: a fourth axis the WCS cannot type.

        Written by hand rather than through ``_write``, since the point is a
        header astropy will parse but not classify.
        """
        header = self._header()
        header["NAXIS4"] = length
        if ctype is None:
            del header["CTYPE4"]
        else:
            header["CTYPE4"] = ctype
        data = np.zeros((length, 8, 6, 6), np.float32)
        fitsio.PrimaryHDU(data, header=header).writeto(path, overwrite=True)

    def test_an_unplaceable_axis_names_the_commands_that_repair_it(self):
        """The repair is fitstoolz's, so the message has to hand it over whole.

        Both routes work today -- what nobody would guess is that the axis is
        addressed as the empty ctype, which is what astropy calls an unset
        CTYPE. Naming the FITS axis number matters for the same reason: the
        error used to report fitstoolz's dimension name ('axis0'), which
        appears in neither the file nor the repair command.
        """
        path = self.tmpdir / "unnamed.fits"
        self._unnamed_fourth_axis(path)

        with self.assertRaises(RuntimeError) as raised:
            utils.zds_from_fits(path)

        message = str(raised.exception)
        assert "axis 4" in message and "CTYPE4 is unset" in message
        assert f"fitstoolz remove-axis {path} --ctype '' --outfile" in message
        assert f"fitstoolz header {path} --add CTYPE4=STOKES --outfile" in message
        # Degenerate, so removing it discards nothing and needs no index.
        assert "--select-index" not in message

    def test_a_named_but_unknown_axis_is_addressed_by_its_own_ctype(self):
        path = self.tmpdir / "linear.fits"
        self._unnamed_fourth_axis(path, ctype="LINEAR")

        with self.assertRaises(RuntimeError) as raised:
            utils.zds_from_fits(path)

        assert "CTYPE4 is 'LINEAR'" in str(raised.exception)
        assert "--ctype LINEAR" in str(raised.exception)

    def test_removing_a_longer_axis_says_which_plane_survives(self):
        """`remove-axis` keeps index 0 by default; that is data loss, unstated."""
        path = self.tmpdir / "long.fits"
        self._unnamed_fourth_axis(path, length=5)

        with self.assertRaises(RuntimeError) as raised:
            utils.zds_from_fits(path)

        assert "--select-index 0" in str(raised.exception)
        assert "1 of 5 planes" in str(raised.exception)

    def test_rest_freq_reaches_the_header_the_outputs_inherit(self):
        path = self.tmpdir / "rf.fits"
        self._write(path)

        zds = utils.zds_from_fits(path, rest_freq=1420.40575)

        assert zds.attrs["header"]["RESTFREQ"] == 1420.40575 * 1e6


class TestRequireDistinctMs(unittest.TestCase):
    """`--output-ms` naming the MS being read.

    `output_ms_dataset` builds a fresh dataset from the input's row metadata and
    `xds_to_table` writes it, so the input would be overwritten while the fit was
    still reading from it -- and with a Doppler correction the channel count
    differs too, so re-running could not recover it. dask-ms writes what it is
    given and says nothing.

    Pure path arithmetic, so none of this needs an MS on disk.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.ms = self.tmpdir / "input.ms"
        self.ms.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_a_distinct_output_is_allowed(self):
        utils.require_distinct_ms(self.ms, self.tmpdir / "output.ms")

    def test_no_output_ms_is_allowed(self):
        """vis-mowjsub without --output-ms writes a column back, which is its own path."""
        utils.require_distinct_ms(self.ms, None)

    def test_the_same_path_is_refused(self):
        with self.assertRaises(RuntimeError) as raised:
            utils.require_distinct_ms(self.ms, self.ms)

        assert "is the MS being read" in str(raised.exception)

    def test_a_trailing_slash_does_not_disguise_it(self):
        with self.assertRaises(RuntimeError):
            utils.require_distinct_ms(self.ms, f"{self.ms}/")

    def test_a_relative_path_does_not_disguise_it(self):
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with self.assertRaises(RuntimeError):
                utils.require_distinct_ms("input.ms", "./input.ms")
        finally:
            os.chdir(cwd)

    def test_a_symlink_does_not_disguise_it(self):
        link = self.tmpdir / "link.ms"
        link.symlink_to(self.ms)

        with self.assertRaises(RuntimeError):
            utils.require_distinct_ms(link, self.ms)

    def test_an_output_that_does_not_exist_yet_still_compares(self):
        """The usual case: the output is a path, not a directory, when this runs."""
        utils.require_distinct_ms(self.ms, self.tmpdir / "not-created-yet.ms")

        with self.assertRaises(RuntimeError):
            utils.require_distinct_ms(self.tmpdir / "absent.ms", self.tmpdir / "absent.ms")


class TestWriteCubes(unittest.TestCase):
    """Writing several cubes in one pass over the graph."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.header = fitsio.Header()
        for index, (ctype, crval, cdelt, cunit) in enumerate([("RA---SIN", 20.0, -1e-3, "deg"), ("DEC--SIN", -30.0, 1e-3, "deg"), ("FREQ", 1.4e9, 1e6, "Hz")], start=1):
            self.header[f"CTYPE{index}"], self.header[f"CRVAL{index}"] = ctype, crval
            self.header[f"CDELT{index}"], self.header[f"CUNIT{index}"], self.header[f"CRPIX{index}"] = cdelt, cunit, 1
        self.header["BUNIT"], self.header["SPECSYS"] = "Jy/beam", "TOPOCENT"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_both_cubes_round_trip_with_their_headers(self):
        one = da.arange(8 * 4 * 4, chunks=(32,), dtype="f4").reshape(8, 4, 4)
        two = one * -1
        paths = [self.tmpdir / "a.fits", self.tmpdir / "b.fits"]

        utils.write_cubes([(str(paths[0]), one, self.header), (str(paths[1]), two, self.header)])

        for path, expected in zip(paths, (one, two)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a short data unit reads back as truncated
                np.testing.assert_allclose(fitsio.getdata(path), np.asarray(expected))
            header = fitsio.getheader(path)
            assert header["NAXIS3"], header["NAXIS1"] == (8, 4)
            assert header["BUNIT"] == "Jy/beam"
            assert header["SPECSYS"] == "TOPOCENT"

    def test_shared_work_is_computed_once(self):
        """Both outputs come off one fit; two writeto calls would refit."""
        calls = []

        def expensive(block):
            if block.size:  # dask calls once with an empty block to infer meta
                calls.append(block.shape)
            return block * 2

        source = da.ones((4, 4, 4), chunks=(2, 4, 4), dtype="f4")
        shared = source.map_blocks(expensive, dtype="f4")
        utils.write_cubes(
            [
                (str(self.tmpdir / "c.fits"), shared, self.header),
                (str(self.tmpdir / "d.fits"), shared + 1, self.header),
            ]
        )

        assert len(calls) == source.numblocks[0], f"expected one call per block, got {len(calls)}"

    def test_an_existing_file_is_refused_unless_overwrite(self):
        path = self.tmpdir / "e.fits"
        array = da.zeros((2, 4, 4), chunks=-1, dtype="f4")
        utils.write_cubes([(str(path), array, self.header)])

        with self.assertRaises(OSError):
            utils.write_cubes([(str(path), array, self.header)])

        utils.write_cubes([(str(path), array + 5, self.header)], overwrite=True)
        np.testing.assert_allclose(fitsio.getdata(path), 5.0)

    def test_the_data_unit_is_block_aligned(self):
        """A file that stops short of a 2880-byte block reads back as truncated."""
        for shape in ((7, 13, 5), (1, 32, 4, 4)):
            path = self.tmpdir / f"pad{len(shape)}{shape[0]}.fits"
            utils.write_cubes([(str(path), da.zeros(shape, chunks=-1, dtype="f4"), self.header)], overwrite=True)
            assert os.path.getsize(path) % utils.FITS_BLOCK == 0, shape
