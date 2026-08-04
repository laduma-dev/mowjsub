"""The click layer in `mowjsub.parser._cli`.

A pystep's typed signature is the schema, but two things are policy rather than
type and so cannot live in it: which field is positional, and which fields have
to exist before the step runs. ``Path`` says what a value is, not whether it is
already there. Both are arguments to ``make_command``, and these check the
second one is actually wired up -- it is keyed by field *name*, so a rename
would otherwise stop it applying without anything failing.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import astropy.io.fits as fitsio
import numpy as np
from click.testing import CliRunner

from mowjsub import utils
from mowjsub.parser._cli import make_command
from mowjsub.parser.doppler_mowjsub import command as doppler_command
from mowjsub.parser.doppler_mowjsub import doppler_mowjsub
from mowjsub.parser.im_mowjsub import command as im_command
from mowjsub.parser.vis_mowjsub import command as vis_command

#: Enough of each command line to get past the required options, so what the
#: test observes is the path check rather than a missing parameter.
COMMANDS = {
    "im-mowjsub": (im_command, ["--fit-model", "polynomial", "--order", "3"]),
    "vis-mowjsub": (vis_command, ["--output-column", "LINE", "--fit-model", "polynomial", "--order", "3"]),
    "doppler-mowjsub": (doppler_command, ["--input-column", "A", "--output-column", "B", "--doppler-frame", "bary"]),
}


#: A small cube with a real spectral WCS, and enough structure that a continuum
#: fit has something to do -- a flat cube would model identically at any window
#: width, so the width tests below would pass however the option were wired.
_CUBE_NCHAN = 64
_CUBE_F0 = 1.36e9
_CUBE_DF = 6.5e3
#: Channel frequencies in Hz, as `utils.chans_in_velwidth` wants them.
_CUBE_FREQS = _CUBE_F0 + _CUBE_DF * np.arange(_CUBE_NCHAN)


def _write_cube(path, npix=3):
    """Write that cube: noise on a sloping continuum, with a one-channel line."""
    header = fitsio.Header()
    header["CTYPE1"], header["CRVAL1"], header["CDELT1"], header["CRPIX1"], header["CUNIT1"] = "RA---SIN", 201.36506, -1.0 / 3600, npix / 2, "deg"
    header["CTYPE2"], header["CRVAL2"], header["CDELT2"], header["CRPIX2"], header["CUNIT2"] = "DEC--SIN", -43.01911, 1.0 / 3600, npix / 2, "deg"
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"], header["CUNIT3"] = "FREQ", _CUBE_F0, _CUBE_DF, 1.0, "Hz"
    header["RESTFRQ"] = 1.42040575e9
    header["SPECSYS"] = "TOPOCENT"
    header["BUNIT"] = "Jy/beam"

    rng = np.random.default_rng(20260804)
    data = rng.normal(scale=0.1, size=(_CUBE_NCHAN, npix, npix)).astype(np.float32)
    data += np.linspace(1.0, 2.0, _CUBE_NCHAN, dtype=np.float32)[:, None, None]
    data[_CUBE_NCHAN // 2] += 10.0

    fitsio.PrimaryHDU(data, header=header).writeto(path, overwrite=True)


class TestMissingInputs(unittest.TestCase):
    """A missing input is a usage error, not a traceback.

    scabha's ``must_exist: yes`` used to do this; it went with the YAML schemas
    in the Stimela 3 port, and a missing input then surfaced as whatever the
    reader happened to raise part-way into the run -- for a cube, a bare
    ``FileNotFoundError`` from inside fitstoolz.
    """

    def test_every_command_refuses_a_missing_positional(self):
        for name, (command, required) in COMMANDS.items():
            result = CliRunner().invoke(command, ["/nonexistent/input"] + required)

            assert result.exit_code == 2, f"{name}: expected a usage error, got {result.exit_code}"
            assert "does not exist" in result.output, name

    def test_the_message_names_the_parameter(self):
        result = CliRunner().invoke(im_command, ["/nonexistent/cube.fits", "--fit-model", "polynomial", "--order", "3"])

        assert "INPUT_IMAGE" in result.output
        assert "/nonexistent/cube.fits" in result.output

    def test_an_optional_input_is_checked_when_given(self):
        """--mask-image is not required, but a name that is not there is still wrong."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            cube = tmpdir / "cube.fits"
            fitsio.PrimaryHDU(np.zeros((4, 4, 4), dtype=np.float32)).writeto(cube)

            result = CliRunner().invoke(im_command, [str(cube), "--mask-image", "/nonexistent/mask.fits", "--fit-model", "polynomial", "--order", "3"])

            assert result.exit_code == 2
            assert "--mask-image" in result.output
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_an_optional_input_left_unset_is_not_checked(self):
        """A default of None must not be run through the path check."""
        result = CliRunner().invoke(im_command, ["/nonexistent/cube.fits", "--fit-model", "polynomial", "--order", "3"])

        assert "--mask-image" not in result.output, "the unset mask was checked as though it had been given"


class TestOutputMsIsNotTheInput(unittest.TestCase):
    """`--output-ms` naming the MS being read, checked through the commands.

    `utils.require_distinct_ms` is unit-tested in test_main; what this adds is
    that both commands taking an ``--output-ms`` actually call it. Missing the
    call in one of them is the whole failure mode, and it would leave no trace
    until someone lost an MS.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # An empty directory: the guard runs before the MS is opened, so nothing
        # here needs casacore or a real table.
        self.ms = self.tmpdir / "input.ms"
        self.ms.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _invoke(self, name, output_ms):
        """Run a command with an otherwise valid line, varying only --output-ms.

        ``--output-ms`` is optional on vis-mowjsub and required on
        doppler-mowjsub, so it is appended here rather than carried in COMMANDS.
        """
        command, required = COMMANDS[name]

        return CliRunner().invoke(command, [str(self.ms), *required, "--output-ms", str(output_ms)], catch_exceptions=True)

    def test_vis_mowjsub_refuses_it(self):
        result = self._invoke("vis-mowjsub", self.ms)

        assert result.exit_code != 0
        assert "is the MS being read" in str(result.exception)

    def test_doppler_mowjsub_refuses_it(self):
        result = self._invoke("doppler-mowjsub", self.ms)

        assert result.exit_code != 0
        assert "is the MS being read" in str(result.exception)

    def test_a_distinct_output_gets_past_the_guard(self):
        """It fails afterwards on the fake MS, which is the point: not on this check."""
        result = self._invoke("vis-mowjsub", self.tmpdir / "output.ms")

        assert "is the MS being read" not in str(result.exception)


class TestMakeCommand(unittest.TestCase):
    def test_must_exist_must_name_real_parameters(self):
        """The policy is keyed by field name, so a rename has to fail loudly.

        Silently not applying is the failure mode worth guarding: the command
        would keep working, and only a missing file would reveal it.
        """
        with self.assertRaises(ValueError) as raised:
            make_command(doppler_mowjsub, positional="ms", must_exist=("ms", "no_such_field"))

        assert "no_such_field" in str(raised.exception)

    def test_help_and_version_still_work(self):
        for name, (command, _) in COMMANDS.items():
            assert CliRunner().invoke(command, ["--help"]).exit_code == 0, name

            result = CliRunner().invoke(command, ["--version"])
            assert result.exit_code == 0, name
            assert result.output.strip(), name


class TestChanWidthIsHonoured(unittest.TestCase):
    """``--chan-width`` reaches the fitter.

    Both entry points accepted the option and neither passed it on. It
    therefore satisfied no validation check and reached no ``FitFunc``: a run
    given only ``--chan-width`` was refused for want of ``--vel-width``, and a
    run given both quietly used the velocity. ``FitFunc.default_prepare`` has
    taken either throughout -- ``test_main.TestFitsFunc`` pins that the two
    agree -- so what was missing is CLI wiring, which is why this lives here
    and not beside those.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.cube = self.tmpdir / "cube.fits"
        _write_cube(self.cube)
        # An empty directory is enough for vis-mowjsub: the width checks run
        # before the MS is opened, the same trick TestOutputMsIsNotTheInput uses.
        self.ms = self.tmpdir / "input.ms"
        self.ms.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _im(self, *args):
        return CliRunner().invoke(im_command, [str(self.cube), *args], catch_exceptions=True)

    def _vis(self, *args):
        return CliRunner().invoke(vis_command, [str(self.ms), "--output-column", "LINE", *args], catch_exceptions=True)

    def test_chan_width_alone_satisfies_the_width_requirement(self):
        for name, result in (
            ("im-mowjsub", self._im("--fit-model", "scipy-median-filter", "--chan-width", "5", "--output-prefix", str(self.tmpdir / "out"))),
            ("vis-mowjsub", self._vis("--fit-model", "scipy-median-filter", "--chan-width", "5")),
        ):
            assert "is required for fit-model" not in str(result.exception), name

    def test_neither_width_is_still_refused(self):
        for name, result in (
            ("im-mowjsub", self._im("--fit-model", "scipy-median-filter")),
            ("vis-mowjsub", self._vis("--fit-model", "scipy-median-filter")),
        ):
            assert "is required for fit-model" in str(result.exception), name

    def test_both_widths_together_are_refused(self):
        """They set the same window in different units, and `default_prepare`
        silently prefers the velocity -- so accepting both would mean ignoring
        one of two values the caller stated explicitly."""
        for name, result in (
            ("im-mowjsub", self._im("--fit-model", "scipy-median-filter", "--vel-width", "15", "--chan-width", "5")),
            ("vis-mowjsub", self._vis("--fit-model", "scipy-median-filter", "--vel-width", "15", "--chan-width", "5")),
        ):
            assert "Give one, not both" in str(result.exception), name

    def test_a_chan_width_below_one_is_refused(self):
        """`default_prepare` bumps an even width up by one, which turns 0 into a
        one-channel window and leaves a negative width negative."""
        for name, result in (
            ("im-mowjsub", self._im("--fit-model", "scipy-median-filter", "--chan-width", "0")),
            ("vis-mowjsub", self._vis("--fit-model", "scipy-median-filter", "--chan-width", "0")),
        ):
            assert "is not a channel count" in str(result.exception), name

    def test_chan_width_gives_the_same_continuum_as_the_equivalent_vel_width(self):
        """The end-to-end check that the value is used, not merely accepted.

        `scipy-median-filter` rather than a spline: `FitBSpline` jitters its
        knots and exposes no seed on the command line, so a CLI-level equality
        check on it would be flaky for reasons that have nothing to do with the
        width.
        """
        chanwidth = utils.chans_in_velwidth(_CUBE_FREQS, 15 * 1e3)

        by_vel = self.tmpdir / "by-vel"
        by_chan = self.tmpdir / "by-chan"

        for prefix, width in ((by_vel, ("--vel-width", "15")), (by_chan, ("--chan-width", str(chanwidth)))):
            result = self._im("--fit-model", "scipy-median-filter", *width, "--output-prefix", str(prefix))
            assert result.exit_code == 0, (prefix, result.exception)

        with fitsio.open(f"{by_vel}-cont.fits") as vel, fitsio.open(f"{by_chan}-cont.fits") as chan:
            np.testing.assert_array_equal(chan[0].data, vel[0].data)

    def test_an_unconverted_chan_width_would_have_failed_that_check(self):
        """Guards the test above: a different width must give a different fit,
        or the equality it asserts would hold however the option were wired.
        """
        other = self.tmpdir / "other"
        result = self._im("--fit-model", "scipy-median-filter", "--chan-width", "3", "--output-prefix", str(other))
        assert result.exit_code == 0, result.exception

        reference = self.tmpdir / "reference"
        result = self._im("--fit-model", "scipy-median-filter", "--vel-width", "15", "--output-prefix", str(reference))
        assert result.exit_code == 0, result.exception

        with fitsio.open(f"{other}-cont.fits") as a, fitsio.open(f"{reference}-cont.fits") as b:
            assert not np.array_equal(a[0].data, b[0].data)


class TestSigmaClipIsScalar(unittest.TestCase):
    """``--sigma-clip`` takes one value, and the automask reaches the fit.

    It used to be a ``list[float]`` ("one per iteration"), with a companion
    ``--automask-per-iter``, but no iteration was ever implemented:
    ``get_automask`` does one fit and one clip, and never read the flag.
    ``PixSigmaClip`` multiplies the whole list against the noise array in a
    single operation, so anything but one value mis-broadcast -- usually a
    crash, and a silently wrong mask when the list happened to be as long as
    the spectral axis.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.cube = self.tmpdir / "cube.fits"
        _write_cube(self.cube)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _im(self, *args):
        return CliRunner().invoke(im_command, [str(self.cube), *args], catch_exceptions=True)

    def test_a_single_value_is_accepted(self):
        result = self._im("--fit-model", "scipy-median-filter", "--vel-width", "15", "--sigma-clip", "3", "--output-prefix", str(self.tmpdir / "out"))

        assert result.exit_code == 0, result.exception

    def test_a_second_value_is_a_usage_error(self):
        """Not a traceback from inside numpy's broadcasting, which is what a
        list-typed option gave for every count the clipper could not use."""
        result = self._im("--fit-model", "scipy-median-filter", "--vel-width", "15", "--sigma-clip", "5", "3", "--output-prefix", str(self.tmpdir / "out"))

        assert result.exit_code == 2, result.output

    def test_automask_per_iter_is_gone(self):
        assert "--automask-per-iter" not in CliRunner().invoke(im_command, ["--help"]).output

    def test_the_automask_changes_the_fit(self):
        """Guards the acceptance test above: an option that parsed but did not
        reach `get_automask` would still exit 0.

        The cube carries a one-channel line, so clipping it out of the fit has
        to move the continuum model.
        """
        masked, unmasked = self.tmpdir / "masked", self.tmpdir / "unmasked"

        for prefix, extra in ((masked, ("--sigma-clip", "3")), (unmasked, ())):
            result = self._im("--fit-model", "scipy-median-filter", "--vel-width", "15", *extra, "--output-prefix", str(prefix))
            assert result.exit_code == 0, (prefix, result.exception)

        with fitsio.open(f"{masked}-cont.fits") as a, fitsio.open(f"{unmasked}-cont.fits") as b:
            assert not np.array_equal(a[0].data, b[0].data)
