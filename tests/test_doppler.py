import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation

from mowjsub.doppler import (
    FRAME_CODES,
    C,
    common_channel_grid,
    doppler_factors,
    grid_frequencies,
    parse_channel_grid,
    regrid_rows,
    resample_map,
)

# MeerKAT array centre (ITRF metres) and a southern target, for repeatability.
MEERKAT = EarthLocation.from_geocentric(5109360.133, 2006852.586, -3238948.127, unit="m")
TARGET = (np.radians(201.36506), np.radians(-43.01911))

# A day and a half-year later, to exercise both the diurnal and orbital terms.
TIMES = np.array([58849.0, 58849.5, 59031.75]) * 86400.0


class TestDopplerFactors(unittest.TestCase):
    def test_topo_is_identity(self):
        """Topocentric is the frame the MS already uses, so nothing shifts."""
        factors = doppler_factors(TIMES, TARGET, MEERKAT, "topo")

        assert factors.shape == TIMES.shape
        assert np.all(factors == 1.0)

    def test_barycentric_magnitude_is_physical(self):
        """The correction cannot exceed Earth's orbital speed over c."""
        factors = doppler_factors(TIMES, TARGET, MEERKAT, "bary")

        velocities = np.abs(1.0 - factors) * C
        assert np.all(velocities < 32e3)
        # Half a year apart the sign of the orbital term must flip.
        assert (factors[0] - 1.0) * (factors[2] - 1.0) < 0

    def test_geocentric_is_diurnal_only(self):
        """Removing only Earth rotation is far smaller than the orbital term."""
        geo = doppler_factors(TIMES, TARGET, MEERKAT, "geo")
        bary = doppler_factors(TIMES, TARGET, MEERKAT, "bary")

        assert np.all(np.abs(1.0 - geo) * C < 500.0)
        assert np.all(np.abs(1.0 - geo) < np.abs(1.0 - bary))

    def test_frames_differ_from_bary_by_a_fixed_offset(self):
        """Each frame adds a constant apex term on top of the barycentric one."""
        bary = doppler_factors(TIMES, TARGET, MEERKAT, "bary")

        for frame, expected in (("lsrk", 20e3), ("lsrd", 16.55e3), ("lgroup", 308e3), ("cmb", 369.5e3)):
            factors = doppler_factors(TIMES, TARGET, MEERKAT, frame)
            offsets = (factors / bary - 1.0) * C
            # The offset is the apex velocity projected onto this line of sight,
            # so it is constant in time and cannot exceed the apex speed.
            assert np.allclose(offsets, offsets[0], atol=1e-6), frame
            assert abs(offsets[0]) <= expected * 1.001, frame

    def test_galacto_routes_through_lsrd(self):
        """casacore reaches GALACTO via LSRD, so the LSRD term must be present."""
        lsrd = doppler_factors(TIMES, TARGET, MEERKAT, "lsrd")
        galacto = doppler_factors(TIMES, TARGET, MEERKAT, "galacto")

        offsets = (galacto / lsrd - 1.0) * C
        assert np.allclose(offsets, offsets[0], atol=1e-6)
        assert 0.0 < abs(offsets[0]) <= 220e3 * 1.001

    def test_source_frame_needs_a_velocity(self):
        with self.assertRaises(ValueError):
            doppler_factors(TIMES, TARGET, MEERKAT, "source")

    def test_source_frame_applies_systemic_velocity(self):
        """A receding source puts the rest frequency above the observed one."""
        bary = doppler_factors(TIMES, TARGET, MEERKAT, "bary")
        source = doppler_factors(TIMES, TARGET, MEERKAT, "source", source_vel=1000e3)

        assert np.all(source > bary)
        assert np.allclose((source / bary - 1.0) * C, 1000e3, rtol=1e-2)

    def test_unknown_frame_rejected(self):
        with self.assertRaises(ValueError):
            doppler_factors(TIMES, TARGET, MEERKAT, "not-a-frame")


class TestCasacoreAgreement(unittest.TestCase):
    """Cross-check against casacore's own measures engine, when it has its data."""

    def setUp(self):
        try:
            from casacore.measures import measures
        except ImportError:  # pragma: no cover - casacore always ships with dask-ms
            self.skipTest("python-casacore is unavailable")

        self.dm = measures()
        try:
            self.dm.do_frame(self.dm.position("ITRF", "5109360.133m", "2006852.586m", "-3238948.127m"))
            self.dm.do_frame(self.dm.direction("J2000", "201.36506deg", "-43.01911deg"))
            self.dm.do_frame(self.dm.epoch("UTC", "58849.0d"))
            self.dm.measure(self.dm.frequency("TOPO", "1420405751.786Hz"), "BARY")
        except Exception:
            self.skipTest("casacore measures tables are not installed")

    def test_matches_casacore(self):
        rest = 1420405751.786

        for frame in ("geo", "bary", "lsrk", "lsrd", "galacto", "lgroup", "cmb"):
            for mjd in (58849.0, 59031.75):
                self.dm.do_frame(self.dm.epoch("UTC", f"{mjd}d"))
                reference = self.dm.measure(self.dm.frequency("TOPO", f"{rest}Hz"), frame.upper())["m0"]["value"] / rest
                ours = doppler_factors(np.array([mjd * 86400.0]), TARGET, MEERKAT, frame)[0]

                # Agreement well inside 1 m/s; the residual is ephemeris choice.
                assert abs(ours - reference) * C < 1.0, f"{frame} at MJD {mjd}"


class TestChannelGrid(unittest.TestCase):
    def setUp(self):
        self.nchan = 32
        self.ascending = 1.4e9 + 1e6 * np.arange(self.nchan)
        self.descending = self.ascending[::-1].copy()

    def test_auto_grid_trims_one_channel_each_end(self):
        """With no Doppler drift the grid loses exactly its two guard channels."""
        for freqs in (self.ascending, self.descending):
            nchan, chan0, width = common_channel_grid(freqs, np.ones(4))

            assert nchan == self.nchan - 2
            assert np.isclose(abs(width), 1e6)
            assert np.sign(width) == np.sign(freqs[-1] - freqs[0])
            # Guard channels keep the grid strictly inside the input band.
            edges = grid_frequencies(nchan, chan0, width)[[0, -1]]
            assert edges.min() > min(freqs)
            assert edges.max() < max(freqs)

    def test_auto_grid_shrinks_as_the_band_drifts(self):
        """A wider spread of factors must not widen the common coverage."""
        steady, _, _ = common_channel_grid(self.ascending, np.ones(4))
        drifting, _, _ = common_channel_grid(self.ascending, np.array([0.9999, 1.0001]))

        assert drifting < steady

    def test_auto_grid_rejects_a_drift_beyond_the_band(self):
        with self.assertRaises(ValueError):
            common_channel_grid(self.ascending, np.array([0.5, 1.5]))

    def test_parse_explicit_grid(self):
        assert parse_channel_grid("1000,1419.5MHz,26.1kHz") == (1000, 1419.5e6, 26.1e3)
        # A descending grid is expressed with a negative width.
        assert parse_channel_grid("10, 1.42GHz, -1Hz") == (10, 1.42e9, -1.0)

    def test_parse_rejects_bad_grids(self):
        for bad in ("1000,1419.5MHz", "1000,1419.5MHz,26.1kHz,extra", "1000,1419.5,26.1kHz", "1000,1419.5MHz,26.1km/s", "abc,1419.5MHz,26.1kHz", "0,1419.5MHz,26.1kHz"):
            with self.assertRaises(ValueError, msg=bad):
                parse_channel_grid(bad)

    def test_grid_frequencies(self):
        assert np.allclose(grid_frequencies(3, 1e9, 1e6), [1e9, 1.001e9, 1.002e9])


class TestResampling(unittest.TestCase):
    def setUp(self):
        self.freqs = 1.4e9 + 1e6 * np.arange(16)

    def test_identity_when_the_grid_is_unchanged(self):
        for interpolation in ("nearest", "linear"):
            indices, weights, valid = resample_map(self.freqs, 1.0, self.freqs, interpolation)

            assert np.all(valid)
            resolved = (indices * weights).sum(axis=1)
            assert np.allclose(resolved, np.arange(16))

    def test_nearest_snaps_a_whole_channel_shift(self):
        """Shifting by one channel must renumber, not interpolate."""
        shifted = self.freqs + 1e6
        indices, _, valid = resample_map(self.freqs, 1.0, shifted, "nearest")

        assert np.array_equal(indices[valid, 0], np.arange(1, 16))

    def test_descending_input_handled(self):
        descending = self.freqs[::-1].copy()
        indices, _, valid = resample_map(descending, 1.0, descending, "nearest")

        assert np.all(valid)
        assert np.array_equal(indices[:, 0], np.arange(16))

    def test_channels_outside_the_band_are_invalid(self):
        outside = np.array([1.3e9, 1.4e9, 1.5e9])
        _, _, valid = resample_map(self.freqs, 1.0, outside, "nearest")

        assert list(valid) == [False, True, False]

    def test_linear_interpolates_a_half_channel_offset(self):
        offset = self.freqs[:-1] + 0.5e6
        indices, weights, valid = resample_map(self.freqs, 1.0, offset, "linear")

        assert np.all(valid)
        assert np.allclose(weights, 0.5)
        assert np.array_equal(indices[0], [0, 1])

    def test_unknown_interpolation_rejected(self):
        with self.assertRaises(ValueError):
            resample_map(self.freqs, 1.0, self.freqs, "cubic")


class TestRegridRows(unittest.TestCase):
    def setUp(self):
        self.nrow, self.nchan, self.ncorr = 4, 16, 2
        self.freqs = 1.4e9 + 1e6 * np.arange(self.nchan)
        rng = np.random.default_rng(0)
        shape = (self.nrow, self.nchan, self.ncorr)
        self.data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        self.flags = np.zeros(shape, dtype=bool)
        self.weights = np.ones(shape, dtype=np.float32)

    def _regrid(self, freqs_out, **kwargs):
        return regrid_rows(self.data, self.flags, self.weights, np.ones(self.nrow), self.freqs, freqs_out, **kwargs)

    def test_unshifted_regrid_is_lossless(self):
        data, flags, weights = self._regrid(self.freqs)

        assert np.array_equal(data, self.data)
        assert not flags.any()
        assert np.allclose(weights, 1.0)

    def test_whole_channel_shift(self):
        data, flags, _ = self._regrid(self.freqs[1:])

        assert data.shape == (self.nrow, self.nchan - 1, self.ncorr)
        assert np.array_equal(data, self.data[:, 1:, :])
        assert not flags.any()

    def test_flags_propagate_to_the_output_channel(self):
        self.flags[0, 5, 0] = True
        data, flags, weights = self._regrid(self.freqs)

        assert flags[0, 5, 0]
        assert data[0, 5, 0] == 0
        assert weights[0, 5, 0] == 0
        # Its neighbours and the other rows are untouched.
        assert not flags[0, 4, 0] and not flags[0, 6, 0]
        assert not flags[1].any()

    def test_channels_off_the_band_are_flagged(self):
        beyond = self.freqs + 4e6
        _, flags, _ = self._regrid(beyond)

        # The last four output channels run past the top of the input band.
        assert flags[:, -4:, :].all()
        assert not flags[:, :-4, :].any()

    def test_linear_halves_the_weight_of_a_midpoint_channel(self):
        """Averaging two unit-weight channels doubles the variance."""
        midpoints = self.freqs[:-1] + 0.5e6
        _, _, weights = self._regrid(midpoints, interpolation="linear")

        # 1/w = 0.5**2/1 + 0.5**2/1 = 0.5, so w = 2.
        assert np.allclose(weights, 2.0)

    def test_rows_with_different_factors_shift_independently(self):
        one_channel = 1.0 + 1e6 / self.freqs[0]
        factors = np.array([1.0, 1.0, one_channel, one_channel])
        data, flags, _ = regrid_rows(self.data, self.flags, self.weights, factors, self.freqs, self.freqs[:-2])

        # Unshifted rows come through as they were.
        assert np.array_equal(data[0], self.data[0, :-2, :])
        # A factor that lifts the band by one channel moves each input channel
        # up one output channel, leaving the first output channel uncovered.
        assert flags[2, 0].all()
        assert np.array_equal(data[2, 1:], self.data[2, :13, :])


class TestFrameCodes(unittest.TestCase):
    def test_codes_match_the_ms_convention(self):
        """MEAS_FREQ_REF values follow casacore's MFrequency::Types ordering."""
        assert FRAME_CODES["lsrk"] == 1
        assert FRAME_CODES["lsrd"] == 2
        assert FRAME_CODES["bary"] == 3
        assert FRAME_CODES["geo"] == 4
        assert FRAME_CODES["topo"] == 5
        assert FRAME_CODES["galacto"] == 6
        assert FRAME_CODES["lgroup"] == 7
        assert FRAME_CODES["cmb"] == 8


class TestEndToEnd(unittest.TestCase):
    """Run the CLI over a synthetic MS and inspect the MS it writes."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="mowjsub-doppler-"))
        cls.ms = cls.tmpdir / "test.ms"
        try:
            _make_ms(cls.ms)
        except Exception as exc:  # pragma: no cover
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
            raise unittest.SkipTest(f"could not build a test MS: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run(self, output, *extra):
        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        result = CliRunner().invoke(
            runit,
            [str(self.ms), "--fit-model", "polynomial", "--order", "2", "--output-column", "LINE_DATA", "--output-ms", str(output), *extra],
            catch_exceptions=True,
        )
        return result

    def test_plain_output_ms_is_a_complete_ms(self):
        """Without a Doppler frame, --output-ms must still be a usable MS."""
        from casacore.tables import table
        from daskms import xds_from_ms

        output = self.tmpdir / "plain.ms"
        result = self._run(output)
        assert result.exit_code == 0, result.output

        with table(str(output), ack=False) as main:
            keywords = {k for k, v in main.getkeywords().items() if isinstance(v, str) and v.startswith("Table:")}
            assert {"ANTENNA", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION"} <= keywords
            # Row metadata an imager needs, none of which the caller supplies.
            for column in ("TIME", "ANTENNA1", "ANTENNA2", "UVW", "FIELD_ID", "DATA_DESC_ID", "FLAG", "WEIGHT_SPECTRUM"):
                assert column in main.colnames(), column
            # The channel grid is untouched here, unlike the Doppler path.
            assert main.getcol("LINE_DATA").shape == (main.nrows(), 32, 2)

        with table(f"{output}/SPECTRAL_WINDOW", ack=False) as spw:
            assert spw.getcell("NUM_CHAN", 0) == 32
            assert spw.getcell("MEAS_FREQ_REF", 0) == FRAME_CODES["topo"]

        # And dask-ms must be able to open the result as an MS.
        assert xds_from_ms(str(output))[0].LINE_DATA.shape[1] == 32

    def test_plain_output_ms_leaves_the_input_alone(self):
        """--output-ms must not add the line column to the input MS."""
        from casacore.tables import table

        self._run(self.tmpdir / "untouched.ms")

        with table(str(self.ms), ack=False) as main:
            assert "LINE_DATA" not in main.colnames()

    def test_doppler_run_writes_a_regridded_ms(self):
        from casacore.tables import table

        output = self.tmpdir / "bary.ms"
        result = self._run(output, "--doppler-frame", "bary")
        assert result.exit_code == 0, result.output

        with table(str(output), ack=False) as main:
            # Subtables must be present for this to be a usable MS.
            keywords = {k for k, v in main.getkeywords().items() if isinstance(v, str) and v.startswith("Table:")}
            assert {"ANTENNA", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION"} <= keywords
            assert "FIELD_ID" in main.colnames() and "DATA_DESC_ID" in main.colnames()
            nchan_out = main.getcol("LINE_DATA").shape[1]

        with table(f"{output}/SPECTRAL_WINDOW", ack=False) as spw:
            assert spw.getcell("NUM_CHAN", 0) == nchan_out
            assert spw.getcell("MEAS_FREQ_REF", 0) == FRAME_CODES["bary"]
            assert spw.getcell("CHAN_FREQ", 0).size == nchan_out
            # Guard channels mean the grid is narrower than the input band.
            assert nchan_out < 32

    def test_explicit_grid_is_honoured(self):
        from casacore.tables import table

        output = self.tmpdir / "lsrk.ms"
        result = self._run(output, "--doppler-frame", "lsrk", "--doppler-chan-grid", "20,1405MHz,1MHz")
        assert result.exit_code == 0, result.output

        with table(f"{output}/SPECTRAL_WINDOW", ack=False) as spw:
            assert spw.getcell("NUM_CHAN", 0) == 20
            assert spw.getcell("MEAS_FREQ_REF", 0) == FRAME_CODES["lsrk"]
            assert np.isclose(spw.getcell("CHAN_FREQ", 0)[0], 1405e6)

    def test_doppler_without_an_output_ms_is_refused(self):
        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        result = CliRunner().invoke(
            runit,
            [str(self.ms), "--fit-model", "polynomial", "--order", "2", "--output-column", "LINE_DATA", "--doppler-frame", "bary"],
            catch_exceptions=True,
        )

        assert result.exit_code != 0
        assert "output-ms" in str(result.exception)

    def test_output_column_has_no_default(self):
        """Omitting --output-column must be an error, not a silent LINE_DATA."""
        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        result = CliRunner().invoke(
            runit,
            [str(self.ms), "--fit-model", "polynomial", "--order", "2", "--output-ms", str(self.tmpdir / "nope.ms")],
            catch_exceptions=True,
        )

        assert result.exit_code != 0
        assert "output-column" in result.output

    def test_writing_the_input_column_in_place_is_refused(self):
        """In place, --output-column == --input-column destroys the input."""
        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        result = CliRunner().invoke(
            runit,
            [str(self.ms), "--fit-model", "polynomial", "--order", "2", "--input-column", "DATA", "--output-column", "DATA"],
            catch_exceptions=True,
        )

        assert result.exit_code != 0
        assert "overwriting it" in str(result.exception)


class TestStandaloneDoppler(unittest.TestCase):
    """doppler-mowjsub over an MS whose continuum has already gone."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="mowjsub-standalone-"))
        cls.ms = cls.tmpdir / "test.ms"
        try:
            _make_ms(cls.ms)
        except Exception as exc:  # pragma: no cover
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
            raise unittest.SkipTest(f"could not build a test MS: {exc}") from exc

        # The line MS the standalone command is meant to consume: continuum
        # subtracted on the native topocentric grid, no Doppler correction yet.
        cls.line_ms = cls.tmpdir / "line.ms"
        result = cls._contsub(cls.line_ms)
        if result.exit_code != 0:  # pragma: no cover
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
            raise unittest.SkipTest(f"could not build a line MS: {result.output}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    @classmethod
    def _contsub(cls, output, *extra):
        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        return CliRunner().invoke(
            runit,
            [str(cls.ms), "--fit-model", "polynomial", "--order", "2", "--output-column", "LINE_DATA", "--output-ms", str(output), *extra],
            catch_exceptions=True,
        )

    def _run(self, ms, output, *extra):
        from click.testing import CliRunner

        from mowjsub.parser.doppler_mowjsub import runit

        return CliRunner().invoke(
            runit,
            [str(ms), "--input-column", "LINE_DATA", "--output-column", "LINE_DATA", "--output-ms", str(output), *extra],
            catch_exceptions=True,
        )

    def test_writes_a_regridded_ms(self):
        from casacore.tables import table

        output = self.tmpdir / "bary.ms"
        result = self._run(self.line_ms, output, "--doppler-frame", "bary")
        assert result.exit_code == 0, result.output

        with table(str(output), ack=False) as main:
            keywords = {k for k, v in main.getkeywords().items() if isinstance(v, str) and v.startswith("Table:")}
            assert {"ANTENNA", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION"} <= keywords
            assert "FIELD_ID" in main.colnames() and "DATA_DESC_ID" in main.colnames()
            nchan_out = main.getcol("LINE_DATA").shape[1]

        with table(f"{output}/SPECTRAL_WINDOW", ack=False) as spw:
            assert spw.getcell("NUM_CHAN", 0) == nchan_out
            assert spw.getcell("MEAS_FREQ_REF", 0) == FRAME_CODES["bary"]
            assert nchan_out < 32

    def test_matches_the_fused_single_pass(self):
        """Subtract-then-correct must equal vis-mowjsub's fused --doppler-frame."""
        from casacore.tables import table

        fused = self.tmpdir / "fused.ms"
        assert self._contsub(fused, "--doppler-frame", "bary").exit_code == 0

        split = self.tmpdir / "split.ms"
        assert self._run(self.line_ms, split, "--doppler-frame", "bary").exit_code == 0

        with table(str(fused), ack=False) as a, table(str(split), ack=False) as b:
            # Same grid, and the same visibilities on it. Both orders fit on the
            # topocentric grid and regrid only the residual, so this is exact --
            # the intermediate MS round-trip preserves complex64.
            assert np.array_equal(a.getcol("LINE_DATA"), b.getcol("LINE_DATA"))
            assert np.array_equal(a.getcol("FLAG"), b.getcol("FLAG"))

    def test_explicit_grid_is_honoured(self):
        from casacore.tables import table

        output = self.tmpdir / "lsrk.ms"
        result = self._run(self.line_ms, output, "--doppler-frame", "lsrk", "--doppler-chan-grid", "20,1405MHz,1MHz")
        assert result.exit_code == 0, result.output

        with table(f"{output}/SPECTRAL_WINDOW", ack=False) as spw:
            assert spw.getcell("NUM_CHAN", 0) == 20
            assert spw.getcell("MEAS_FREQ_REF", 0) == FRAME_CODES["lsrk"]
            assert np.isclose(spw.getcell("CHAN_FREQ", 0)[0], 1405e6)

    def test_input_ms_is_left_alone(self):
        from casacore.tables import table

        self._run(self.line_ms, self.tmpdir / "untouched.ms", "--doppler-frame", "bary")

        with table(str(self.line_ms), ack=False) as main:
            assert main.getcol("LINE_DATA").shape[1] == 32

    def test_the_intermediate_ms_has_no_data_column(self):
        """Pins why --input-column has no default: DATA is simply not there."""
        from casacore.tables import table

        with table(str(self.line_ms), ack=False) as main:
            assert "LINE_DATA" in main.colnames()
            assert "DATA" not in main.colnames()

    def test_a_missing_column_is_reported(self):
        result = self._run_columns(self.line_ms, self.tmpdir / "missing.ms", "NOPE_DATA")

        assert result.exit_code != 0
        assert "NOPE_DATA" in str(result.exception)
        # The error must name what is actually available.
        assert "LINE_DATA" in str(result.exception)

    def _run_columns(self, ms, output, input_column):
        from click.testing import CliRunner

        from mowjsub.parser.doppler_mowjsub import runit

        return CliRunner().invoke(
            runit,
            [str(ms), "--input-column", input_column, "--output-column", "LINE_DATA", "--output-ms", str(output), "--doppler-frame", "bary"],
            catch_exceptions=True,
        )

    def test_a_non_topocentric_input_is_refused(self):
        """Correcting an already-regridded MS would apply the shift twice."""
        import shutil as _shutil

        from casacore.tables import table

        already = self.tmpdir / "already-bary.ms"
        _shutil.rmtree(already, ignore_errors=True)
        _shutil.copytree(self.line_ms, already)
        with table(f"{already}/SPECTRAL_WINDOW", readonly=False, ack=False, lockoptions="auto") as spw:
            spw.putcell("MEAS_FREQ_REF", 0, FRAME_CODES["bary"])

        result = self._run(already, self.tmpdir / "twice.ms", "--doppler-frame", "lsrk")

        assert result.exit_code != 0
        assert "BARY" in str(result.exception)
        assert "twice" in str(result.exception)

    def test_the_fused_path_refuses_a_non_topocentric_input_too(self):
        """The guard lives in the shared helper, so vis-mowjsub gets it as well."""
        import shutil as _shutil

        from casacore.tables import table

        already = self.tmpdir / "already-lsrk.ms"
        _shutil.rmtree(already, ignore_errors=True)
        _shutil.copytree(self.ms, already)
        with table(f"{already}/SPECTRAL_WINDOW", readonly=False, ack=False, lockoptions="auto") as spw:
            spw.putcell("MEAS_FREQ_REF", 0, FRAME_CODES["lsrk"])

        from click.testing import CliRunner

        from mowjsub.parser.vis_mowjsub import runit

        result = CliRunner().invoke(
            runit,
            [
                str(already),
                "--fit-model",
                "polynomial",
                "--order",
                "2",
                "--output-column",
                "LINE_DATA",
                "--output-ms",
                str(self.tmpdir / "nope2.ms"),
                "--doppler-frame",
                "bary",
            ],
            catch_exceptions=True,
        )

        assert result.exit_code != 0
        assert "LSRK" in str(result.exception)

    def test_required_options_are_enforced(self):
        from click.testing import CliRunner

        from mowjsub.parser.doppler_mowjsub import runit

        for missing, args in (
            ("doppler-frame", ["--input-column", "LINE_DATA", "--output-column", "L", "--output-ms", "x.ms"]),
            ("output-ms", ["--input-column", "LINE_DATA", "--output-column", "L", "--doppler-frame", "bary"]),
            ("input-column", ["--output-column", "L", "--output-ms", "x.ms", "--doppler-frame", "bary"]),
            ("output-column", ["--input-column", "LINE_DATA", "--output-ms", "x.ms", "--doppler-frame", "bary"]),
        ):
            result = CliRunner().invoke(runit, [str(self.line_ms), *args], catch_exceptions=True)
            assert result.exit_code != 0, missing
            assert missing in result.output, missing


def _make_ms(path, nant=4, ntime=4, nchan=32, ncorr=2, f0=1.4e9, df=1e6):
    """Write a minimal but valid MS for the end-to-end test."""
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc, table

    path = str(path)
    ms = default_ms(path)
    # default_ms leaves out the optional data columns, so declare them.
    ms.addcols(
        maketabdesc(
            [
                makearrcoldesc("DATA", 0j, ndim=2, shape=[nchan, ncorr], valuetype="complex"),
                makearrcoldesc("WEIGHT_SPECTRUM", 0.0, ndim=2, shape=[nchan, ncorr], valuetype="float"),
            ]
        )
    )

    nbl = nant * (nant - 1) // 2
    nrow = ntime * nbl
    ms.addrows(nrow)

    ant1, ant2 = np.triu_indices(nant, k=1)
    times = np.repeat(58849.0 * 86400.0 + np.arange(ntime) * 60.0, nbl)
    rng = np.random.default_rng(42)

    ms.putcol("TIME", times)
    ms.putcol("TIME_CENTROID", times)
    ms.putcol("ANTENNA1", np.tile(ant1, ntime))
    ms.putcol("ANTENNA2", np.tile(ant2, ntime))
    ms.putcol("DATA_DESC_ID", np.zeros(nrow, int))
    ms.putcol("FIELD_ID", np.zeros(nrow, int))
    ms.putcol("UVW", rng.normal(scale=1000.0, size=(nrow, 3)))
    ms.putcol("DATA", (rng.normal(size=(nrow, nchan, ncorr)) + 1j * rng.normal(size=(nrow, nchan, ncorr))).astype(np.complex64))
    ms.putcol("FLAG", np.zeros((nrow, nchan, ncorr), bool))
    ms.putcol("WEIGHT_SPECTRUM", np.ones((nrow, nchan, ncorr), np.float32))
    ms.putcol("WEIGHT", np.ones((nrow, ncorr), np.float32))
    ms.putcol("SIGMA", np.ones((nrow, ncorr), np.float32))
    ms.putcol("INTERVAL", np.full(nrow, 60.0))
    ms.putcol("EXPOSURE", np.full(nrow, 60.0))
    ms.close()

    freqs = f0 + df * np.arange(nchan)
    with table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False, lockoptions="auto") as spw:
        spw.addrows(1)
        spw.putcell("NUM_CHAN", 0, nchan)
        spw.putcell("CHAN_FREQ", 0, freqs)
        spw.putcell("CHAN_WIDTH", 0, np.full(nchan, df))
        spw.putcell("RESOLUTION", 0, np.full(nchan, df))
        spw.putcell("EFFECTIVE_BW", 0, np.full(nchan, df))
        spw.putcell("REF_FREQUENCY", 0, f0)
        spw.putcell("TOTAL_BANDWIDTH", 0, nchan * df)
        spw.putcell("MEAS_FREQ_REF", 0, FRAME_CODES["topo"])
        spw.putcell("NET_SIDEBAND", 0, 1)
        spw.putcell("NAME", 0, "test-spw")

    with table(f"{path}/POLARIZATION", readonly=False, ack=False, lockoptions="auto") as pol:
        pol.addrows(1)
        pol.putcell("NUM_CORR", 0, ncorr)
        pol.putcell("CORR_TYPE", 0, np.array([9, 12][:ncorr]))
        pol.putcell("CORR_PRODUCT", 0, np.zeros((ncorr, 2), int))

    with table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False, lockoptions="auto") as dd:
        dd.addrows(1)
        dd.putcell("SPECTRAL_WINDOW_ID", 0, 0)
        dd.putcell("POLARIZATION_ID", 0, 0)
        dd.putcell("FLAG_ROW", 0, False)

    with table(f"{path}/FIELD", readonly=False, ack=False, lockoptions="auto") as fld:
        fld.addrows(1)
        for column in ("PHASE_DIR", "REFERENCE_DIR", "DELAY_DIR"):
            fld.putcell(column, 0, np.array([[TARGET[0], TARGET[1]]]))
        fld.putcell("NAME", 0, "testfield")
        fld.putcell("SOURCE_ID", 0, 0)
        fld.putcell("NUM_POLY", 0, 0)

    centre = np.array([5109360.133, 2006852.586, -3238948.127])
    with table(f"{path}/ANTENNA", readonly=False, ack=False, lockoptions="auto") as ant:
        ant.addrows(nant)
        for i in range(nant):
            ant.putcell("POSITION", i, centre + rng.normal(scale=500.0, size=3))
            ant.putcell("NAME", i, f"m{i:03d}")
            ant.putcell("STATION", i, f"m{i:03d}")
            ant.putcell("DISH_DIAMETER", i, 13.5)
            ant.putcell("MOUNT", i, "ALT-AZ")
            ant.putcell("TYPE", i, "GROUND-BASED")

    with table(f"{path}/OBSERVATION", readonly=False, ack=False, lockoptions="auto") as obs:
        obs.addrows(1)
        obs.putcell("TELESCOPE_NAME", 0, "MeerKAT")
        obs.putcell("TIME_RANGE", 0, np.array([times[0], times[-1]]))
        obs.putcell("OBSERVER", 0, "test")
        obs.putcell("PROJECT", 0, "test")

    return path
