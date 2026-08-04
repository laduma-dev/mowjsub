"""Image-plane Doppler correction: `im-mowjsub --doppler-frame`.

These build synthetic FITS cubes rather than Measurement Sets, so nothing here
needs casacore.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import astropy.io.fits as fitsio
import numpy as np
from astropy import units
from astropy.wcs import WCS
from click.testing import CliRunner

from mowjsub.doppler import FITS_SPECSYS, resample_cube
from mowjsub.parser.im_mowjsub import command
from mowjsub.utils import (
    FitsHeader,
    apply_cube_doppler,
    chans_in_velwidth,
    cube_spectral_axis,
    plan_cube_doppler,
    spectral_frequencies,
)

# MeerKAT, as ITRF metres, matching tests/test_doppler.py.
MEERKAT = (5109360.133, 2006852.586, -3238948.127)


def _header(nchan=32, f0=1.4e9, df=1e6, npix=4, stokes=False, spectral_axis=3):
    """A minimal but WCS-complete spectral cube header.

    NAXIS* are set explicitly because there is no data array to infer the
    channel count from; ``spectral_frequencies`` reads it off the header, for
    whichever axis the WCS says is spectral.

    ``spectral_axis=4`` puts STOKES on NAXIS3 and FREQ on NAXIS4 -- legal, and
    what CASA ``exportfits`` can emit. Every path handles that layout: the
    continuum write-back transposes by name, so a cube round-trips in whatever
    order it arrived in.
    """
    header = fitsio.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"] = 201.36506
    header["CDELT1"] = -1.0 / 3600
    header["CRPIX1"] = npix / 2
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"] = -43.01911
    header["CDELT2"] = 1.0 / 3600
    header["CRPIX2"] = npix / 2
    header["CUNIT2"] = "deg"

    spectral = ({"CTYPE": "FREQ", "CRVAL": f0, "CDELT": df, "CRPIX": 1.0, "CUNIT": "Hz"}, nchan)
    stokes_axis = ({"CTYPE": "STOKES", "CRVAL": 1.0, "CDELT": 1.0, "CRPIX": 1.0, "CUNIT": ""}, 1)

    if spectral_axis == 4:
        trailing = [stokes_axis, spectral]
    else:
        trailing = [spectral, stokes_axis] if stokes else [spectral]

    for index, (axis, length) in enumerate(trailing, start=3):
        for key, value in axis.items():
            header[f"{key}{index}"] = value
        header[f"NAXIS{index}"] = length

    header["NAXIS"] = 2 + len(trailing)
    header["NAXIS1"] = npix
    header["NAXIS2"] = npix

    header["DATE-OBS"] = "2020-01-01T00:00:00"
    header["OBSGEO-X"], header["OBSGEO-Y"], header["OBSGEO-Z"] = MEERKAT
    header["SPECSYS"] = "TOPOCENT"
    header["RESTFRQ"] = 1.42040575e9
    header["BUNIT"] = "Jy/beam"

    return header


def _cube(path, nchan=32, npix=4, line_channel=None, stokes=False, **kwargs):
    """Write a flat-spectrum cube, optionally with a one-channel spike."""
    header = _header(nchan=nchan, npix=npix, stokes=stokes, **kwargs)

    # C order is the reverse of FITS axis order: RA, DEC, FREQ[, STOKES].
    shape = ([1] if stokes else []) + [nchan, npix, npix]
    data = np.ones(shape, dtype=np.float32)

    if line_channel is not None:
        index = [slice(None)] * len(shape)
        index[1 if stokes else 0] = line_channel
        data[tuple(index)] = 10.0

    fitsio.PrimaryHDU(data, header=header).writeto(path, overwrite=True)

    return header


def _velocity_header(ctype, nchan=32, crval=1000.0, cdelt=10.0, cunit="km/s"):
    """A cube whose spectral axis is not a frequency.

    Defaults to a 1000 km/s optical-velocity axis in 10 km/s channels, which is
    an ordinary way to store an HI cube and the layout that used to be read as
    though the numbers were Hz.
    """
    header = _header(nchan=nchan)
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CUNIT3"] = ctype, crval, cdelt, cunit

    return header


def _reference_frequencies(header, convention):
    """The same grid via astropy's own conversion, as an independent check.

    Deliberately not the code under test's route: this goes through
    ``Quantity.to`` with the equivalency stated here rather than derived from
    the CTYPE, so a wrong convention in the lookup table shows up as a mismatch.
    """
    axis = WCS(header).wcs.spec
    nchan = int(header[f"NAXIS{axis + 1}"])
    velocities = (header["CRVAL3"] + np.arange(nchan) * header["CDELT3"]) * units.Unit(header["CUNIT3"])

    return velocities.to_value(units.Hz, equivalencies=convention(header["RESTFRQ"] * units.Hz))


class TestSpectralAxis(unittest.TestCase):
    def test_naxis3_is_accepted(self):
        assert cube_spectral_axis(_header()) == 2

    def test_a_spectral_axis_elsewhere_is_returned_not_refused(self):
        """A lookup, not a check.

        This used to refuse anything but NAXIS3, because the continuum
        write-back placed its axes by fixed transpose. It goes by name now, so
        every path here is order-agnostic.
        """
        assert cube_spectral_axis(_header(spectral_axis=4)) == 3

    def test_a_cube_with_no_spectral_axis_is_refused(self):
        header = _header()
        for key in ("CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3"):
            header.pop(key, None)

        with self.assertRaises(RuntimeError) as raised:
            cube_spectral_axis(header)

        assert "no spectral axis" in str(raised.exception)


class TestSpectralFrequencies(unittest.TestCase):
    """The channel grid, read through the low-level WCS.

    Its predecessor -- ``FitsHeader.retFreq`` reading ``NAXIS3`` through the
    high-level WCS -- was wrong in two ways that these pin: it took the channel
    count from a fixed axis, and it could not produce a grid at all without an
    observation time.
    """

    def test_the_grid_matches_crval_and_cdelt(self):
        freqs = spectral_frequencies(_header(nchan=32, f0=1.4e9, df=1e6))

        assert freqs.size == 32
        np.testing.assert_allclose(freqs, 1.4e9 + np.arange(32) * 1e6, rtol=1e-12)

    def test_a_swapped_axis_order_is_read_correctly(self):
        """STOKES on NAXIS3, FREQ on NAXIS4: the layout that used to give 1 channel."""
        header = _header(nchan=32, spectral_axis=4)
        assert header["NAXIS3"] == 1, "the Stokes length the old code would have used"

        freqs = spectral_frequencies(header)

        assert freqs.size == 32
        np.testing.assert_allclose(freqs, 1.4e9 + np.arange(32) * 1e6, rtol=1e-12)

    def test_no_observation_time_is_needed(self):
        header = _header()
        for key in ("DATE-OBS", "MJD-OBS", "TIMESYS"):
            header.pop(key, None)

        np.testing.assert_allclose(spectral_frequencies(header), 1.4e9 + np.arange(32) * 1e6, rtol=1e-12)

    def test_a_non_si_cunit_still_yields_hz(self):
        """CUNIT is the header's business; this returns SI either way."""
        header = _header()
        header["CUNIT3"], header["CRVAL3"], header["CDELT3"] = "MHz", 1400.0, 1.0

        np.testing.assert_allclose(spectral_frequencies(header), 1.4e9 + np.arange(32) * 1e6, rtol=1e-12)

    def test_a_cube_with_no_spectral_axis_is_refused(self):
        header = _header()
        for key in ("CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3"):
            header.pop(key, None)

        with self.assertRaises(RuntimeError) as raised:
            spectral_frequencies(header)

        assert "no spectral axis" in str(raised.exception)

    def test_ret_freq_is_the_same_grid_in_mhz(self):
        """The fitters work in MHz; keep the two definitions tied together."""
        header = _header()

        np.testing.assert_allclose(FitsHeader(header).retFreq(), spectral_frequencies(header) / 1e6, rtol=1e-12)


class TestNonFrequencySpectralAxes(unittest.TestCase):
    """Axes the low-level WCS does not hand back in Hz.

    ``array_index_to_world_values`` returns each axis in its *own* SI unit, so a
    FREQ axis in MHz does arrive as Hz -- but a VOPT axis arrives as m/s and a
    WAVE axis as metres, and neither says so. Taking them on trust is silent
    rather than loud, which is what these pin: on the 10 km/s axis below, the
    numbers read as Hz make ``chans_in_velwidth`` compute a 2600 km/s channel
    instead of a 10 km/s one, so it clamps the fit window to a single channel.
    """

    def test_an_optical_velocity_axis_becomes_frequency(self):
        header = _velocity_header("VOPT")

        freqs = spectral_frequencies(header)

        np.testing.assert_allclose(freqs, _reference_frequencies(header, units.doppler_optical), rtol=1e-12)
        # Not the raw m/s the axis is stored in.
        assert freqs.min() > 1.4e9

    def test_the_doppler_convention_comes_from_the_ctype(self):
        """VOPT, VRAD and VELO are three different grids, not one grid named three ways."""
        conventions = {
            "VOPT": units.doppler_optical,
            "VRAD": units.doppler_radio,
            "VELO": units.doppler_relativistic,
        }

        grids = {}
        for ctype, convention in conventions.items():
            header = _velocity_header(ctype)
            grids[ctype] = spectral_frequencies(header)
            np.testing.assert_allclose(grids[ctype], _reference_frequencies(header, convention), rtol=1e-12)

        # At 1000 km/s the optical and radio conventions part company by ~15 kHz
        # at L band, which is most of a MeerKAT channel. Reading one as the
        # other is a real error, not a rounding one.
        assert abs(grids["VOPT"][0] - grids["VRAD"][0]) > 1e4

    def test_an_algorithm_code_does_not_change_the_convention(self):
        """'VOPT-F2W' is still optical; only the first four characters name the convention.

        The 'F2W' code says the axis is sampled linearly in wavelength rather
        than in velocity, so the grid is not identical to a plain VOPT one --
        1.5 kHz apart at the far end of this band. The reference channel is
        identical, and the whole grid stays an order of magnitude nearer optical
        than the 15.8 kHz that separates optical from radio.
        """
        freqs = spectral_frequencies(_velocity_header("VOPT-F2W"))
        optical = _reference_frequencies(_velocity_header("VOPT"), units.doppler_optical)
        radio = _reference_frequencies(_velocity_header("VOPT"), units.doppler_radio)

        np.testing.assert_allclose(freqs[0], optical[0], rtol=1e-12)
        assert np.abs(freqs - optical).max() < 0.2 * np.abs(optical - radio).min()

    def test_a_wavelength_axis_becomes_frequency(self):
        header = _velocity_header("WAVE", crval=0.21, cdelt=1e-4, cunit="m")

        freqs = spectral_frequencies(header)

        np.testing.assert_allclose(freqs, (np.asarray([0.21 + index * 1e-4 for index in range(32)]) * units.m).to_value(units.Hz, equivalencies=units.spectral()), rtol=1e-12)
        # Frequency decreases as wavelength increases; the grid keeps the axis's
        # own direction rather than being resorted behind the caller's back.
        assert freqs[0] > freqs[-1]

    def test_a_velocity_axis_with_no_rest_frequency_is_refused(self):
        """Refused, not guessed at: there is no default rest frequency to fall back on."""
        header = _velocity_header("VOPT")
        del header["RESTFRQ"]

        with self.assertRaises(RuntimeError) as raised:
            spectral_frequencies(header)

        assert "rest frequency" in str(raised.exception)
        assert "--rest-freq" in str(raised.exception)

    def test_the_deprecated_restfreq_spelling_is_honoured(self):
        """``--rest-freq`` writes RESTFREQ, which is exactly how a user rescues such a cube."""
        header = _velocity_header("VOPT")
        rest = header.pop("RESTFRQ")
        header["RESTFREQ"] = rest

        np.testing.assert_allclose(spectral_frequencies(header), _reference_frequencies(_velocity_header("VOPT"), units.doppler_optical), rtol=1e-12)

    def test_a_dimensionless_spectral_axis_is_refused(self):
        """ZOPT is a spectral axis with no unit at all, so it cannot be waved through as Hz."""
        header = _velocity_header("ZOPT", crval=0.003, cdelt=1e-5, cunit="")

        with self.assertRaises(RuntimeError) as raised:
            spectral_frequencies(header)

        assert "ZOPT" in str(raised.exception)

    def test_the_fit_window_matches_the_equivalent_frequency_cube(self):
        """What the bug actually cost: the fitters size their window off this grid."""
        velocity = spectral_frequencies(_velocity_header("VOPT"))
        frequency = spectral_frequencies(_header(nchan=32, f0=velocity[0], df=velocity[1] - velocity[0]))

        assert chans_in_velwidth(velocity, 250e3) == chans_in_velwidth(frequency, 250e3)
        assert chans_in_velwidth(velocity, 250e3) == 25


class TestAlreadyCorrected(unittest.TestCase):
    """The cube-side counterpart of the MEAS_FREQ_REF check on an MS.

    ``doppler_factors`` converts *from* topocentric frequency, so a cube that has
    already been shifted -- imaged from an ``mstransform`` output, or produced by
    an earlier ``im-mowjsub --doppler-frame`` run -- would get the conversion
    applied a second time. Nothing about the result would look wrong.
    """

    def test_a_barycentric_cube_is_refused(self):
        header = _header()
        header["SPECSYS"] = "BARYCENT"

        with self.assertRaises(RuntimeError) as raised:
            plan_cube_doppler(header, "bary")

        assert "BARY" in str(raised.exception)
        assert "twice" in str(raised.exception)

    def test_the_refusal_does_not_depend_on_the_target_frame(self):
        """A BARYCENT cube cannot be taken to LSRK either; the input grid is the problem."""
        header = _header()
        header["SPECSYS"] = "BARYCENT"

        with self.assertRaises(RuntimeError):
            plan_cube_doppler(header, "lsrk")

    def test_a_frame_name_mowjsub_has_no_code_for_is_still_refused(self):
        """HELIOCEN is legal FITS and not in FITS_SPECSYS; report it rather than pass it."""
        header = _header()
        header["SPECSYS"] = "HELIOCEN"

        with self.assertRaises(RuntimeError) as raised:
            plan_cube_doppler(header, "bary")

        assert "HELIOCEN" in str(raised.exception)

    def test_a_topocentric_cube_is_accepted(self):
        assert plan_cube_doppler(_header(), "bary") is not None

    def test_a_cube_with_no_specsys_is_taken_on_trust(self):
        """Same bet as the MS path: a silent header is usually an imager that omitted it."""
        header = _header()
        del header["SPECSYS"]

        assert plan_cube_doppler(header, "bary") is not None

    def test_a_corrected_cube_will_not_go_round_again(self):
        """apply_cube_doppler stamps the new frame, so its own output is refused."""
        header = _header()
        _, corrected = apply_cube_doppler(np.ones((32, 4, 4), dtype=np.float32), header, plan_cube_doppler(header, "bary"))

        with self.assertRaises(RuntimeError):
            plan_cube_doppler(corrected, "bary")


class TestPlan(unittest.TestCase):
    def test_topo_is_the_identity_factor(self):
        plan = plan_cube_doppler(_header(), "topo")

        assert plan.factor == 1.0
        # 'auto' still drops a guard channel at each end.
        assert plan.freqs_out.size == 30

    def test_bary_factor_is_physical(self):
        plan = plan_cube_doppler(_header(), "bary")

        assert abs(1.0 - plan.factor) * 299792458.0 < 32e3

    def test_explicit_grid_is_used_verbatim(self):
        plan = plan_cube_doppler(_header(), "bary", chan_grid="10,1401MHz,2MHz")

        assert plan.freqs_out.size == 10
        assert np.isclose(plan.freqs_out[0], 1401e6)
        assert np.isclose(plan.chanwidth, 2e6)

    def test_source_frame_needs_a_velocity(self):
        with self.assertRaises(RuntimeError) as raised:
            plan_cube_doppler(_header(), "source")

        assert "doppler-source-vel" in str(raised.exception)

    def test_duration_gives_a_drift(self):
        plan = plan_cube_doppler(_header(), "bary", obs_duration=8.0)

        # Eight hours of Earth rotation, so hundreds of m/s but under a km/s.
        assert plan.drift is not None
        assert 50.0 < plan.drift < 1000.0

    def test_no_duration_means_no_drift_check(self):
        assert plan_cube_doppler(_header(), "bary").drift is None

    def test_missing_location_is_reported(self):
        header = _header()
        for key in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z"):
            header.pop(key, None)

        with self.assertRaises(RuntimeError) as raised:
            plan_cube_doppler(header, "bary")

        assert "doppler-telescope" in str(raised.exception)

    def test_missing_time_is_reported(self):
        header = _header()
        header.pop("DATE-OBS", None)

        with self.assertRaises(RuntimeError) as raised:
            plan_cube_doppler(header, "bary")

        assert "doppler-time" in str(raised.exception)

    def test_doppler_time_rescues_a_header_with_no_epoch(self):
        """The epoch is needed for the Doppler factor, and only for that.

        It used to be needed for the channel grid as well, which is why the
        override had to be resolved before the grid was read. The grid no longer
        cares; this pins that the factor still honours the override.
        """
        header = _header()
        header.pop("DATE-OBS", None)

        plan = plan_cube_doppler(header, "bary", obs_time="2020-01-01T00:00:00")

        assert np.isclose(plan.factor, plan_cube_doppler(_header(), "bary").factor)

    def test_overrides_are_honoured(self):
        """With no header metadata at all, the overrides must carry the plan."""
        header = _header()
        for key in ("DATE-OBS", "OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z"):
            header.pop(key, None)

        plan = plan_cube_doppler(
            header,
            "bary",
            obs_time="2020-01-01T00:00:00",
            telescope="MeerKAT",
            phase_centre="201.36506,-43.01911",
        )

        assert np.isclose(plan.factor, plan_cube_doppler(_header(), "bary").factor, rtol=0, atol=1e-9)


class TestApply(unittest.TestCase):
    def test_a_whole_channel_shift_moves_the_line(self):
        """An output grid offset by one channel moves the spike by one channel."""
        header = _header(nchan=32, f0=1.4e9, df=1e6)
        data = np.ones((32, 4, 4), dtype=np.float32)
        data[10] = 10.0

        plan = plan_cube_doppler(header, "topo", chan_grid="30,1401MHz,1MHz")
        shifted, out = apply_cube_doppler(data, header, plan)

        # Output channel 0 is input channel 1, so the spike lands at 10 - 1.
        assert np.argmax(shifted[:, 0, 0]) == 9
        assert out["CRVAL3"] == 1401e6
        assert out["NAXIS3"] == 30

    def test_header_is_rewritten(self):
        header = _header()
        data = np.ones((32, 4, 4), dtype=np.float32)

        plan = plan_cube_doppler(header, "lsrk")
        _, out = apply_cube_doppler(data, header, plan)

        assert out["SPECSYS"] == FITS_SPECSYS["lsrk"]
        assert out["CUNIT3"] == "Hz"
        assert out["CRPIX3"] == 1.0
        assert np.isclose(out["CRVAL3"], plan.freqs_out[0])
        assert np.isclose(out["CDELT3"], plan.chanwidth)
        # Untouched keywords survive.
        assert out["BUNIT"] == "Jy/beam"
        assert out["RESTFRQ"] == 1.42040575e9

    def test_stale_velocity_keywords_are_dropped(self):
        header = _header()
        header["ALTRVAL"] = 1.0e5
        header["ALTRPIX"] = 3.0
        header["VELREF"] = 257

        plan = plan_cube_doppler(header, "bary")
        _, out = apply_cube_doppler(np.ones((32, 4, 4), dtype=np.float32), header, plan)

        for stale in ("ALTRVAL", "ALTRPIX", "VELREF"):
            assert stale not in out, stale

    def test_a_cd_matrix_is_kept_consistent(self):
        header = _header()
        header["CD3_3"] = 1e6

        plan = plan_cube_doppler(header, "bary")
        _, out = apply_cube_doppler(np.ones((32, 4, 4), dtype=np.float32), header, plan)

        assert np.isclose(out["CD3_3"], plan.chanwidth)

    def test_dtype_is_preserved(self):
        plan = plan_cube_doppler(_header(), "bary")
        resampled, _ = apply_cube_doppler(np.ones((32, 4, 4), dtype=np.float32), _header(), plan)

        assert resampled.dtype == np.float32

    def test_channels_off_the_band_become_nan(self):
        freqs_in = 1.4e9 + 1e6 * np.arange(16)
        data = np.ones((16, 2, 2), dtype=np.float32)
        # Ask for four channels beyond the top of the band.
        freqs_out = freqs_in + 4e6

        resampled = resample_cube(data, 1.0, freqs_in, freqs_out, axis=0)

        assert np.isnan(resampled[-4:]).all()
        assert not np.isnan(resampled[:-4]).any()

    def test_stokes_cube_resamples_on_the_right_axis(self):
        """The spectral axis is NAXIS3, i.e. numpy axis 1 in a 4-D cube."""
        header = _header(stokes=True)
        data = np.ones((1, 32, 4, 4), dtype=np.float32)
        data[0, 10] = 10.0

        plan = plan_cube_doppler(header, "topo", chan_grid="30,1401MHz,1MHz")
        shifted, _ = apply_cube_doppler(data, header, plan)

        assert shifted.shape == (1, 30, 4, 4)
        assert np.argmax(shifted[0, :, 0, 0]) == 9


class TestEndToEnd(unittest.TestCase):
    """Run im-mowjsub over a synthetic cube and inspect what it writes."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="mowjsub-imdoppler-"))
        self.cube = self.tmpdir / "cube.fits"
        _cube(self.cube, nchan=32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, *extra):
        prefix = str(self.tmpdir / "out")
        return (
            CliRunner().invoke(
                command,
                [str(self.cube), "--output-prefix", prefix, "--fit-model", "polynomial", "--order", "2", *extra],
                catch_exceptions=True,
            ),
            Path(f"{prefix}-cont.fits"),
            Path(f"{prefix}-line.fits"),
        )

    def test_without_a_frame_the_grid_is_untouched(self):
        result, cont, line = self._run()
        assert result.exit_code == 0, result.output

        for path in (cont, line):
            with fitsio.open(path) as hdul:
                assert hdul[0].header["NAXIS3"] == 32
                assert hdul[0].header["SPECSYS"] == "TOPOCENT"

    def test_both_cubes_land_on_the_corrected_grid(self):
        result, cont, line = self._run("--doppler-frame", "bary")
        assert result.exit_code == 0, result.output

        headers = []
        for path in (cont, line):
            with fitsio.open(path) as hdul:
                headers.append(hdul[0].header)
                assert hdul[0].header["SPECSYS"] == FITS_SPECSYS["bary"]
                # Guard channels mean the grid is narrower than the input band,
                # by exactly the one channel dropped at each end. This was a
                # range: a shifted grid could land either side of the floor() in
                # common_channel_grid and lose one more, unpredictably.
                assert hdul[0].header["NAXIS3"] == 30

        # The pair must remain recombinable, i.e. be on one identical grid.
        for key in ("NAXIS3", "CRVAL3", "CDELT3", "CRPIX3", "SPECSYS"):
            assert headers[0][key] == headers[1][key], key

    def test_no_scratch_continuum_is_written(self):
        """The residual is taken from arrays, so nothing is staged on disk.

        The Doppler path used to write a topocentric continuum to
        ``{prefix}-cont-topo.fits`` and delete it in a ``finally``, because the
        residual was formed by re-reading the continuum off disk.
        """
        result, _, _ = self._run("--doppler-frame", "bary")
        assert result.exit_code == 0, result.output

        assert not (self.tmpdir / "out-cont-topo.fits").exists()
        assert sorted(p.name for p in self.tmpdir.glob("*.fits")) == ["cube.fits", "out-cont.fits", "out-line.fits"]

    def test_the_continuum_and_line_add_back_up(self):
        """Whatever else the pair go through, they have to sum to the input."""
        result, cont, line = self._run()
        assert result.exit_code == 0, result.output

        source = fitsio.getdata(self.cube)
        np.testing.assert_allclose(fitsio.getdata(cont) + fitsio.getdata(line), source, rtol=1e-5, atol=1e-5)

    def test_explicit_grid_is_honoured(self):
        result, _, line = self._run("--doppler-frame", "lsrk", "--doppler-chan-grid", "20,1405MHz,1MHz")
        assert result.exit_code == 0, result.output

        with fitsio.open(line) as hdul:
            assert hdul[0].header["NAXIS3"] == 20
            assert np.isclose(hdul[0].header["CRVAL3"], 1405e6)
            assert hdul[0].header["SPECSYS"] == FITS_SPECSYS["lsrk"]

    def test_a_large_drift_warns_but_still_runs(self):
        result, _, line = self._run("--doppler-frame", "bary", "--doppler-obs-duration", "8")
        assert result.exit_code == 0, result.output

        assert line.exists()

    def test_a_swapped_axis_order_runs_and_writes_that_order_back(self):
        """CTYPE3=STOKES, CTYPE4=FREQ: legal, and what CASA exportfits can emit.

        This used to be refused outright. Axes are matched by what the WCS calls
        them on the way in and placed by name on the way out, so the layout
        survives the round trip rather than being reordered or rejected.
        """
        swapped = self.tmpdir / "swapped.fits"
        header = _header(spectral_axis=4)
        # C order is reversed: FREQ, STOKES, DEC, RA.
        data = np.ones((32, 1, 4, 4), dtype=np.float32)
        data[10] = 10.0
        fitsio.PrimaryHDU(data, header=header).writeto(swapped)

        for extra in ([], ["--doppler-frame", "bary"]):
            prefix = str(self.tmpdir / f"sw{len(extra)}")
            result = CliRunner().invoke(
                command,
                [str(swapped), "--output-prefix", prefix, "--fit-model", "polynomial", "--order", "2", *extra],
                catch_exceptions=True,
            )

            assert result.exit_code == 0, f"{extra}: {result.output}{result.exception}"

            for path in (Path(f"{prefix}-cont.fits"), Path(f"{prefix}-line.fits")):
                with fitsio.open(path) as hdul:
                    out = hdul[0].header
                    # The file's own layout, not a normalised one.
                    assert out["CTYPE3"] == "STOKES", extra
                    assert out["CTYPE4"] == "FREQ", extra
                    assert out["NAXIS3"] == 1, extra
                    assert hdul[0].data.shape[1] == 1, extra
                    # The spectral axis is NAXIS4, so it is outermost in C order.
                    assert hdul[0].data.shape[0] == out["NAXIS4"], extra

    def test_a_swapped_cube_subtracts_the_same_continuum(self):
        """The fit must not depend on where the spectral axis sits.

        The same data in the two layouts has to give the same residual, which is
        what catches an axis being matched positionally somewhere.
        """
        rng = np.random.default_rng(4)
        spectra = rng.normal(loc=5.0, scale=0.1, size=(32, 4, 4)).astype(np.float32)

        normal = self.tmpdir / "normal.fits"
        fitsio.PrimaryHDU(spectra, header=_header()).writeto(normal)

        swapped = self.tmpdir / "swap.fits"
        fitsio.PrimaryHDU(spectra[:, np.newaxis], header=_header(spectral_axis=4)).writeto(swapped)

        lines = []
        for index, path in enumerate((normal, swapped)):
            prefix = str(self.tmpdir / f"cmp{index}")
            result = CliRunner().invoke(
                command,
                [str(path), "--output-prefix", prefix, "--fit-model", "polynomial", "--order", "2"],
                catch_exceptions=True,
            )
            assert result.exit_code == 0, result.output
            lines.append(np.squeeze(fitsio.getdata(f"{prefix}-line.fits")))

        np.testing.assert_allclose(lines[0], lines[1], rtol=1e-5, atol=1e-6)

    def test_source_frame_without_a_velocity_is_refused(self):
        result, _, _ = self._run("--doppler-frame", "source")

        assert result.exit_code != 0
        assert "doppler-source-vel" in str(result.exception)


if __name__ == "__main__":
    unittest.main()
