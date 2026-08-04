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
