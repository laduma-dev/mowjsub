"""Native Click commands and optional Stimela adapters from typed signatures."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

import click
from pydantic import ValidationError, create_model

import mowjsub
from mowjsub import LOGGER

#: What ``--loglevel`` accepts. 'trace' is not a Python level; it is kept
#: because the option has always offered it, and maps to DEBUG.
LOG_LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def configure_logging(level: str | int = "info") -> None:
    """Attach mowjsub's console handler, at `level`.

    Called once per command invocation, from the callback below, so a
    command that offers no ``--loglevel`` (``vis-mowjsub``,
    ``doppler-mowjsub``) is still audible at INFO -- which is what
    ``set_logger()`` with no argument used to do for them.

    Args:
        level: A name from :data:`LOG_LEVELS`, or a logging level.
    """
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)

    logger = logging.getLogger(LOGGER)
    logger.setLevel(level)

    if not any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setLevel(level)


def _show_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(mowjsub.__version__)
    ctx.exit()


#: An input that has to be there before the step runs. Both kinds are ordinary
#: here -- a FITS cube is a file, an MS is a directory -- so neither is excluded.
EXISTING_PATH = click.Path(exists=True, path_type=Path)


def _click_type(annotation):
    """Map the small public parameter vocabulary explicitly; refuse unknown types."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        types = [item for item in get_args(annotation) if item is not type(None)]
        if len(types) == 1:
            return _click_type(types[0])
    if origin is Literal:
        return click.Choice(get_args(annotation))
    if annotation is Path:
        return click.Path(path_type=Path)
    if annotation in (str, int, float, bool):
        return annotation
    raise TypeError(f"Unsupported CLI parameter type: {annotation!r}")


def make_command(func, *, name: str | None = None, info: str | None = None, positional: str | None = None, must_exist: Iterable[str] = ()) -> click.Command:
    """Build a command from a typed function with Pydantic Field defaults.

    ``positional`` and ``must_exist`` are CLI policies keyed by parameter name.
    The callback validates parsed values and calls the function directly, so
    processing exceptions retain their original type and message.
    """
    name = name or func.__name__.lstrip("_").replace("_", "-")
    hints = get_type_hints(func)
    fields = {key: (hints[key], ... if value.default is inspect.Parameter.empty else value.default) for key, value in inspect.signature(func).parameters.items()}
    model = create_model(f"{func.__name__}Inputs", **fields)
    must_exist = set(must_exist)
    for policy, names in (("must_exist", must_exist), ("positional", {positional} if positional else set())):
        unknown = names - set(model.model_fields)
        if unknown:
            raise ValueError(f"make_command({policy}=...) names {sorted(unknown)}, which {name!r} has no parameter for.")

    params: list[click.Parameter] = [click.Option(["--version"], is_flag=True, expose_value=False, is_eager=True, callback=_show_version, help="Show the version and exit.")]
    for key, field in model.model_fields.items():
        annotation = field.annotation
        multiple = get_origin(annotation) is list
        param_type = _click_type(get_args(annotation)[0] if multiple else annotation)
        if key in must_exist:
            if not isinstance(param_type, click.Path):
                raise TypeError(f"must_exist parameter {key!r} must be a Path")
            param_type = EXISTING_PATH
        required = field.is_required()
        default = None if required else field.get_default(call_default_factory=True)
        if key == positional:
            params.append(click.Argument([key], required=required, **({} if required else {"default": default}), type=param_type))
            continue
        flag = annotation is bool
        option = "--" + key.replace("_", "-")
        declarations = [f"{option}/--no-{key.replace('_', '-')}" if flag else option]
        abbreviation = (field.json_schema_extra or {}).get("abbreviation")
        if abbreviation:
            declarations.append("-" + abbreviation)
        params.append(
            click.Option(
                declarations,
                type=param_type,
                required=required,
                **({} if required else {"default": default}),
                multiple=multiple,
                is_flag=flag,
                help=field.description,
                show_default=True,
            )
        )

    def _callback(**raw):
        try:
            values = model(**raw)
        except ValidationError as error:
            raise click.UsageError(str(error)) from error
        configure_logging(getattr(values, "loglevel", "info"))
        return func(**{key: getattr(values, key) for key in model.model_fields})

    return click.Command(name=name, params=params, callback=_callback, help=info or inspect.getdoc(func), no_args_is_help=True)


def optional_step(namespace, attribute, func, *, name, info):
    """Resolve and cache genuine StepRefs only when an integration is requested."""
    public_name = func.__name__.lstrip("_")
    if attribute not in ("step", public_name):
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {attribute!r}")
    try:
        import shinobi
    except ModuleNotFoundError as error:
        if error.name != "shinobi":
            raise
        raise ModuleNotFoundError("Stimela integration requires the optional extra: pip install 'mowjsub[ninja]'", name="shinobi") from error
    step = shinobi.pystep(name=name, info=info)(func)
    namespace["step"] = namespace[public_name] = step
    return step
