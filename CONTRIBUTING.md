# Contributing to mowjsub

Thanks for your interest in contributing! **mowjsub** is a continuum-subtraction
library for spectral-line radio data, with two entry points: `im-mowjsub` for
image-plane work on FITS cubes and `vis-mowjsub` for visibility-plane work on
CASA Measurement Sets. Useful contributions include new fitting functions, bug
reports, focused fixes, tests, and documentation.

## Scope and philosophy

Keep things simple and robust, favouring plain Python and NumPy over new layers
of machinery. Two conventions in this repo are worth knowing before you start:

- **Fitters are self-contained.** Every fitting function subclasses `FitFunc`
  (`src/mowjsub/fitfuncs.py`) and implements
  `fit(data, mask, weights) -> np.ndarray`, where `mask` is boolean and `True`
  means *excluded from the fit*. `prepare()` runs before `fit()` and converts
  `velwidth` (km/s) into a channel count. Raise `BadFitError` when a spectrum
  has too few valid points; the callers turn those spectra into NaN rather than
  guessing.
- **CLI options live in YAML, not in decorators.** Parameters are declared in
  `src/mowjsub/parser/im_mowjsub.yaml` and `vis_mowjsub.yaml`, loaded through
  scabha's `paramfile_loader` and turned into Click options by
  `clickify_parameters`. Add a new option by editing the schema — hand-written
  `@click.option` decorators will drift out of step with the stimela cabs in
  `src/mowjsub/stimelating/`.

The Doppler path (`src/mowjsub/doppler.py`) has two load-bearing details that
are easy to "fix" into being wrong: the topocentric-to-barycentric step is a
plain Newtonian projection because that is what casacore does, and the fit
happens on the native topocentric grid with only the residual regridded. Both
are explained in `AGENTS.md` and pinned by tests — read that section before
touching it.

## Ways to contribute

- **Add a fitting function** — subclass `FitFunc`, and expose it through the
  CLI schemas so both the CLI and the stimela cab pick it up.
- **Report bugs** and request features via
  [issues](https://github.com/laduma-dev/mowjsub/issues). For a fitting or
  regridding bug, the input header or `SPECTRAL_WINDOW` details and the exact
  command line are usually what make it reproducible.
- **Improve documentation** under `docs/source/`, or the docstrings that feed
  the API reference.
- **Submit code** — bug fixes, new fitters, tests.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python
>=3.11,<3.14.

```bash
git clone git@github.com:laduma-dev/mowjsub.git
cd mowjsub
uv sync
uv run pytest
uv run ruff check src/mowjsub/

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

That last line is the only setup step that is not `uv`'s job. The hook
(`.githooks/pre-commit`, tracked in the repo) runs `ruff check` and
`ruff format --check` over the Python files you are committing. It installs
nothing and pins nothing — it uses whatever ruff `uv run` resolves for this
project, so it agrees with `uv run ruff check src/mowjsub/` by construction.
Git will not let a repository enable an executable hook by itself, which is why
the `git config` is manual; skip it and you simply get no hook.

The hook is check-only. It will not rewrite a file mid-commit, because that
leaves the staged content different from what was just verified. When it fails
it prints the command to fix things:

```bash
uv run ruff check --fix src/mowjsub/ tests/ && uv run ruff format src/mowjsub/ tests/
```

`git commit --no-verify` bypasses it when you genuinely need to.

Note `uv.lock` is **not** committed (see `.gitignore`), so a fresh clone
resolves whatever is current at that moment. **Always `uv lock && uv sync` after
changing `pyproject.toml`**, so a stale environment doesn't hide a resolution
problem that CI will hit.

**Always run tests through `uv run`.** A bare `pytest` picks up the system
interpreter, and since `pyproject.toml` sets `pythonpath = ["src"]` it gets far
enough to import `mowjsub` before dying on `No module named 'omegaconf'` — a
confusing failure that has nothing to do with your change.

## Testing

```bash
uv run pytest                                                   # full suite
uv run pytest tests/test_doppler.py                             # one module
uv run pytest tests/test_main.py::TestFitsFunc::test_b_spline    # one test
```

`tests/test_main.py` covers the fitters and the image-plane path;
`tests/test_doppler.py` covers frame conversion, grid derivation, resampling,
and end-to-end MS writing. **A new fitter or a change to the Doppler maths
should come with a test.**

`TestCasacoreAgreement` pins mowjsub's frame conversions against casacore's own
measures engine, and skips when python-casacore or the casacore measures tables
are unavailable. A skip there is expected locally; don't read it as a pass if
you changed `doppler.py`.

## Code style

- **Lint must be clean**: `ruff check src/mowjsub/` should report no errors. The
  rule set is stated outright in `ruff.toml` (`E4`/`E7`/`E9`/`F`/`E501`/isort
  `I`, `line-length = 180`) rather than inherited from ruff's defaults, which
  have grown over time and would otherwise redefine the config's meaning on a
  version bump. Broadening it is a fine thing to decide — but decide it, don't
  inherit it from a dependency bump.
- `ruff format src/mowjsub/ tests/` for autoformatting; the hook checks both.
- Match the surrounding code's naming, idiom, and comment density.

## Documentation

Docs are built with Sphinx (Furo theme) and hosted on Read the Docs:

```bash
uv sync --group docs
uv run sphinx-build -b html docs/source docs/_build/html
```

If you add a documentation dependency, add it to the `docs` dependency group in
`pyproject.toml` — `.readthedocs.yaml` installs via `uv sync --group docs`, so
that group is the single source of truth.

## Pull requests

1. Branch off `main` and keep PRs **small and focused**.
2. Make sure `uv run pytest` and `uv run ruff check src/mowjsub/` pass locally
   before opening a PR. CI (`.github/workflows/tests-builds.yaml`) runs both
   across Python 3.11, 3.12 and 3.13.
3. Push and open a PR against `main`. Reference any related issue.

### Commit messages

Write clear, descriptive commit messages explaining *why* a change is made. No
formal convention (Conventional Commits, sign-off/DCO, or CLA) is required.

## Code of conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
