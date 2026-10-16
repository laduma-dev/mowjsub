"""The native entry points and the optional Stimela adapter have separate lifetimes."""

import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Literal, get_type_hints

import pytest
from click.testing import CliRunner
from pydantic import Field

from mowjsub.parser._cli import make_command

MODULES = ("im_mowjsub", "vis_mowjsub", "doppler_mowjsub")


def test_native_imports_and_commands_never_import_shinobi():
    code = """
import importlib, importlib.abc, sys
from pathlib import Path
class RefuseNinja(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'shinobi', 'scabha'}:
            raise AssertionError('Native import attempted ' + fullname)
sys.meta_path.insert(0, RefuseNinja())
from click.testing import CliRunner
for name in ('im_mowjsub', 'vis_mowjsub', 'doppler_mowjsub'):
    module = importlib.import_module('mowjsub.parser.' + name)
    assert callable(module.runit)
    for flag in ('--help', '--version'):
        result = CliRunner().invoke(module.command, [flag])
        assert result.exit_code == 0, result.exception
assert 'shinobi' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_parser_passes_typed_values_and_defaults(tmp_path):
    received = {}

    def example(
        source: Path = Field(...),
        count: int = Field(2, json_schema_extra={"abbreviation": "hi"}),
        enabled: bool = Field(True),
        mode: Literal["one", "two"] = Field("one"),
        optional: float | None = Field(None),
        labels: list[str] = Field(default_factory=list),
    ) -> None:
        received.update({key: value for key, value in locals().items() if key != "received"})

    command = make_command(example, positional="source", must_exist=("source",))
    result = CliRunner().invoke(command, [str(tmp_path), "-hi", "7", "--no-enabled", "--mode", "two", "--labels", "a", "--labels", "b"])
    assert result.exit_code == 0, result.exception
    assert received == dict(source=tmp_path, count=7, enabled=False, mode="two", optional=None, labels=["a", "b"])
    assert CliRunner().invoke(command, [str(tmp_path), "--mode", "invalid"]).exit_code == 2
    assert CliRunner().invoke(command, [str(tmp_path), "-hi", "invalid"]).exit_code == 2


def test_required_columns_and_output_are_usage_errors(tmp_path):
    for name, missing in (("vis_mowjsub", "--output-column"), ("doppler_mowjsub", "--input-column")):
        module = importlib.import_module("mowjsub.parser." + name)
        result = CliRunner().invoke(module.command, [str(tmp_path)])
        assert result.exit_code == 2
        assert missing in result.output


def test_missing_optional_dependency_has_install_hint():
    code = """
import importlib, importlib.abc, sys
class RefuseNinja(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'shinobi':
            raise ModuleNotFoundError("No module named 'shinobi'", name='shinobi')
sys.meta_path.insert(0, RefuseNinja())
module = importlib.import_module('mowjsub.parser.im_mowjsub')
try:
    module.step
except ModuleNotFoundError as error:
    assert 'mowjsub[ninja]' in str(error)
else:
    raise AssertionError('step unexpectedly resolved')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(importlib.util.find_spec("shinobi") is None, reason="requires mowjsub[ninja]")
@pytest.mark.parametrize("name", MODULES)
def test_lazy_adapter_is_a_genuine_stepref(name, monkeypatch):
    from shinobi.steps.dispatch import _dispatch
    from shinobi.steps.schema import StepRef

    module = importlib.import_module("mowjsub.parser." + name)
    received = {}
    monkeypatch.setattr(module, "runit", lambda opts: received.update(vars(opts)))
    step = module.step
    assert isinstance(step, StepRef)
    assert getattr(module, name) is step
    assert module.step is step
    native = getattr(module, "_" + name)
    hints = get_type_hints(native)
    signature = inspect.signature(native)
    fields = step.step.inputs_model.model_fields
    assert list(fields) == list(signature.parameters)
    command_parameters = {parameter.name: parameter for parameter in module.command.params}
    for key, parameter in signature.parameters.items():
        field = fields[key]
        assert field.annotation == hints[key]
        assert field.is_required() == parameter.default.is_required()
        assert field.default == parameter.default.default
        assert field.alias == parameter.default.alias
        assert field.description == parameter.default.description
        assert field.json_schema_extra == parameter.default.json_schema_extra
        assert command_parameters[key].required == field.is_required()
        if not field.is_required():
            assert command_parameters[key].default == field.default
    required = {"ms": Path("input.ms"), "input_image": Path("input.fits"), "output_column": "LINE", "input_column": "DATA", "output_ms": Path("output.ms"), "doppler_frame": "bary"}
    values = {key: required[key] for key, field in step.step.inputs_model.model_fields.items() if field.is_required()}
    result = _dispatch(step.step, step.func, **values)
    assert result.success
    assert received["ms" if name != "im_mowjsub" else "input_image"] == required["ms" if name != "im_mowjsub" else "input_image"]
