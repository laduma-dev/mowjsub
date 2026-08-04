"""Build a `click.Command` from a `@shinobi.pystep` StepRef.

Replaces scabha's `clickify_parameters` + YAML parser configs: the options are
derived from the step's pydantic ``inputs_model`` by shinobi's ``build_options``
(dtype, choices, abbreviations, bool and list handling), so a step's signature is
the single schema authority. Modelled on fitstoolz's ``apps/_cli.py``, which in
turn follows simms 3.0.

Unlike fitstoolz, mowjsub's commands are standalone console scripts rather than
subcommands of one group, so there is no root group to take ``--version`` or the
log level from; both belong to the command itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import click
from shinobi.clickutil import build_options, unflatten_kwargs
from shinobi.steps.dispatch import _dispatch

import mowjsub


def _show_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(mowjsub.__version__)
    ctx.exit()


#: An input that has to be there before the step runs. Both kinds are ordinary
#: here -- a FITS cube is a file, an MS is a directory -- so neither is excluded.
EXISTING_PATH = click.Path(exists=True, path_type=Path)


def make_command(step, *, positional: str | None = None, must_exist: Iterable[str] = ()) -> click.Command:
    """Build a `click.Command` for a `@shinobi.pystep` StepRef.

    Args:
        step: The ``StepRef`` produced by ``@shinobi.pystep``.
        positional: Name of the input field to render as a ``click.Argument``
            rather than an option -- ``build_options`` only emits ``--options``.
            This is the scabha ``policies.positional`` equivalent.
        must_exist: Names of input fields click should check exist before the
            step is dispatched. The scabha ``must_exist: yes`` equivalent, and
            like ``positional`` it is a policy rather than a type, so the
            signature cannot carry it: ``Path`` says what the value is, not
            whether it has to be there already. Without this a missing input
            surfaces as whatever the reader raises somewhere inside the run --
            a bare ``FileNotFoundError`` traceback rather than a usage error
            naming the parameter.

    Returns:
        click.Command: Its callback re-nests the flat kwargs and dispatches the
        step in process, exactly as ``shinobi.cli``'s ``run`` does. An exception
        raised inside the step propagates with its message intact, which is what
        the error-path tests read.
    """
    model = step.step.inputs_model
    must_exist = set(must_exist)

    unknown = must_exist - set(model.model_fields)
    if unknown:
        raise ValueError(f"make_command(must_exist=...) names {sorted(unknown)}, which {step.step.name!r} has no parameter for.")

    params: list[click.Parameter] = [
        click.Option(
            ["--version"],
            is_flag=True,
            expose_value=False,
            is_eager=True,
            callback=_show_version,
            help="Show the version and exit.",
        )
    ]

    for option in build_options(model):
        if option.name in must_exist:
            # An optional input is only checked when one is actually given;
            # click leaves a default of None alone.
            option.type = EXISTING_PATH

        if option.name == positional:
            params.append(click.Argument([positional], required=True, type=option.type))
        else:
            params.append(option)

    def _callback(**raw):
        kwargs = unflatten_kwargs(model, raw)
        result = _dispatch(step.step, step.func, **kwargs)
        if not result.success:
            raise click.ClickException(f"{step.step.name!r} failed (returncode {result.returncode}).")

    return click.Command(
        name=step.step.name,
        params=params,
        callback=_callback,
        help=step.step.info,
        no_args_is_help=True,
    )
