# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mowjsub` is a Python library for radio astronomy continuum subtraction, supporting both image-plane (FITS cubes) and visibility-plane (Measurement Set) workflows. It exposes three CLI entry points: `im-mowjsub`, `vis-mowjsub` and `doppler-mowjsub`.

## Commands

This project uses `uv` for dependency management, with a tracked `uv.lock`. A `.venv` is already set up at `.venv/`.

```bash
# Install with dev/test dependencies (`dev` is uv's default group)
uv sync

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_main.py::TestFitsFunc::test_b_spline

# Lint
uv run ruff check src/mowjsub/

# Format
uv run ruff format src/mowjsub/

# Audit dependencies for known vulnerabilities
uv run --with pip-audit python -m pip_audit --skip-editable

# Run the image-plane CLI
uv run im-mowjsub <input.fits> [options]

# Run the visibility-plane CLI
uv run vis-mowjsub [options]

# Run the standalone Doppler-correction CLI
uv run doppler-mowjsub <input.ms> [options]
```

The repo ships a tracked pre-commit hook at `.githooks/pre-commit`, enabled per clone with `git config core.hooksPath .githooks`. It runs `ruff check` and `ruff format --check` over the staged Python files, check-only — it never rewrites a file mid-commit. It uses whatever ruff `uv run` resolves, so it agrees with `uv run ruff check src/mowjsub/` by construction. Bypass with `git commit --no-verify`. There is no `pre-commit` framework dependency; don't reintroduce one.

Dependency updates are configured in `.github/dependabot.yml` (a config GitHub reads directly, not an Actions workflow): weekly `uv` PRs, monthly `github-actions` PRs, with dev tooling and routine runtime bumps grouped and runtime majors left individual on purpose.

When the commit includes `pyproject.toml`, the hook also runs `pip-audit` over the dependency tree — the audit is skipped otherwise, so ordinary commits don't pay a network round-trip. It distinguishes a real finding from pip-audit failing to run (no network, PyPI down) and only blocks on the former. CI runs the same audit unconditionally on every matrix entry.

Linting config is in `ruff.toml`: line length 180, target Python 3.11, isort enabled. The rule set is pinned with `select` rather than `extend-select`, deliberately — see the comment in that file before widening it.

`uv.lock` is tracked. After changing `pyproject.toml` always run `uv lock && uv sync` and commit the lock — CI's `build` job installs with `uv sync --locked` and fails if the two disagree. A second CI job, `latest`, ignores the lock and resolves fresh on a weekly schedule; it is the intended early warning for upstream releases breaking us, so a red `latest` is not necessarily a red branch.

## Architecture

### Processing pipelines

**Image plane** (`im-mowjsub`): Operates on FITS spectral cubes. Loads the cube via `fitstoolz.FitsData` into an `xr.Dataset` with dims `[ra, dec, spectral]`, chunks along RA using Dask, fits a continuum baseline per-pixel per-spectrum, and writes two FITS outputs: `*-cont.fits` (continuum model) and `*-line.fits` (residual). Entry point: `mowjsub/parser/im_mowjsub.py` (`command` for the CLI, `runit` for the work).

**Visibility plane** (`vis-mowjsub`): Operates on CASA Measurement Sets. Reads via `dask-ms`, reshapes row-based data to `[time, baseline, freq, corr]`, fits per-baseline-per-correlation, and writes the result back to an MS column. Optionally applies a Doppler correction on the way out (see below). Entry point: `mowjsub/parser/vis_mowjsub.py` (`command` for the CLI, `runit` for the work).

### Doppler correction (`mowjsub/doppler.py`)

`--doppler-frame` resamples the continuum-subtracted visibilities onto a channel grid fixed in a chosen spectral frame, replacing the `regridms=True` half of a CASA `mstransform` pass. `doppler.py` is pure NumPy/astropy: frame apex velocities, per-timestamp conversion factors, common-grid derivation, and channel resampling. The dask orchestration and MS I/O live in `utils.py` (`doppler_regrid_dataset`, `finalise_regridded_ms`, `observatory_location`).

The correction is reachable three ways, and the differences are physical, not cosmetic:

- `vis-mowjsub --doppler-frame` — fused into a contsub run, one pass.
- `doppler-mowjsub` — the same correction standalone, over an already-subtracted MS (`parser/doppler_mowjsub.py`). Reuses `doppler_regrid_dataset`/`finalise_regridded_ms` unchanged and does no fitting. Exists so a pipeline can separate the two stages without the regrid landing first.
- `im-mowjsub --doppler-frame` — image plane, via `plan_cube_doppler`/`apply_cube_doppler` in `utils.py` and `resample_cube` in `doppler.py`.

The first two apply **one factor per timestamp**. The image-plane path cannot: a cube has already been integrated over time, so it gets a single factor and the intra-track smearing is unrecoverable. It is safe only when the drift over the observation is much smaller than a channel — `--doppler-obs-duration` makes that check explicit and warns past a tenth of a channel. Do not present the three as interchangeable.

Both planes require a topocentric input grid, and refuse rather than correct twice: `doppler_regrid_dataset` calls `require_topocentric` against `SPECTRAL_WINDOW::MEAS_FREQ_REF`, and `plan_cube_doppler` calls `require_topocentric_cube` against the cube's `SPECSYS`. Keep the pair symmetric. The cube case is the easier one to reach, not the more obscure: imaging an `mstransform` output produces such a cube, and so does a previous `im-mowjsub --doppler-frame` run, because `apply_cube_doppler` stamps the new frame into `SPECSYS` on the way out — its own output is therefore refused as input, which is the intended behaviour and is pinned. An input that records no frame at all is taken on trust in both cases; a silent header is far more often an imager that omitted the keyword than a regrid hiding itself.

The cube's channel grid comes from `utils.spectral_frequencies`, which reads the *low-level* WCS: it takes the channel count from whichever axis the WCS calls spectral, and needs no observation time. That last point matters -- the high-level WCS refuses to convert pixels to frequencies without a usable obstime, which is why this used to resolve the epoch and stamp it into a header copy first. `plan_cube_doppler` now resolves the epoch for the Doppler factor alone.

What the low-level WCS gives back is the axis in *its own* SI unit, and that is the one thing that must not be taken on trust. A `FREQ` axis in MHz does arrive as Hz, but a `VOPT` axis arrives as m/s and a `WAVE` axis as metres, and nothing in the return value says which. `spectral_frequencies` therefore converts explicitly, through `DOPPLER_CONVENTIONS` (`VOPT`/`FELO` optical, `VRAD` radio, `VELO` relativistic -- only the first four characters of the `CTYPE` name the convention; `-F2W` and `-LSR` are sampling and frame) and the header's `RESTFRQ`/`RESTWAV`. Velocity with no rest frequency, and anything dimensionless (`ZOPT`, `BETA`), raise rather than return. **Do not restore a `or "Hz"` fallback on the unit**: wcslib already defaults a bare `FREQ` axis to Hz, so the only axes reporting no unit are the dimensionless ones the fallback would wave through. This was silent and expensive when it was wrong -- on a 10 km/s velocity axis the numbers taken as Hz make `chans_in_velwidth` compute a channel 2600 km/s wide instead of 10, so it clamps the fit window to a single channel and the continuum model is worthless with nothing raised.

Axis order is never assumed. `zds_from_fits` reads through `fitstoolz.FitsData` and matches each axis by what the WCS calls it, mapping fitstoolz's dimension names onto mowjsub's through `FITSTOOLZ_DIMS`; it records the file's own order in `attrs['fits_dims']`, and `im_mowjsub.runit` transposes the continuum back into that order **by name**. A cube with `CTYPE3=STOKES` and `CTYPE4=FREQ` -- legal, and what CASA `exportfits` can emit -- therefore round-trips in its own layout. Both halves used to be positional: the reader labelled that cube's Stokes axis `spectral` (the origin of `conflicting sizes for dimension 'spectral'`) and the write-back did a fixed `.transpose((2, 1, 0))`. `cube_spectral_axis` is consequently a lookup now, not a refusal.

One thing the reader must keep saying explicitly: **every axis but RA gets a single chunk**. `get_xds` leaves an unlisted axis at whatever chunking the file's layout implies, and a spectral axis split across chunks hands `ContSub.fitContinuum` partial spectra to fit -- silently, since the gufunc is set to `allow_rechunk`. `tests/test_main.py::TestZdsFromFits::test_the_spectral_axis_is_never_split_across_chunks` pins it, and has to shrink dask's `array.chunk-size` to do so, because a test-sized cube fits in one chunk regardless.

`FitsHeader.retFreq()` is the same grid in **MHz**, and exists for the fitters: `FitFunc.prepare` converts back with `self.freqs * 1e6`, and `FitPolynomial` runs `numpy.polyfit` against those values, where the scale sets the conditioning. Measuring a frequency wants `spectral_frequencies`; fitting against one wants `retFreq`. Do not swap them.

`FitBSpline` places its knots against two FITPACK limits that are easy to reintroduce: they must be **strictly** interior (the last usable index is `size - 2`, not `size - 1`), and there can be at most `m - k - 1` of them for `m` valid points of order `k`. The count comes from `--vel-width`, a physical width, so a cube whose channels are a large fraction of it asks for one segment per channel and breaches both. FITPACK signals this as a bare `TypeError: An error occurred` from inside Fortran, and the exception escapes `ContSub.fitContinuum`, so it kills the run instead of marking the spectrum bad. Bound the knots by the *valid* point count, not the channel count -- masking tightens the limit without the caller changing anything. `tests/test_main.py::TestBSplineKnotBounds` sweeps the parameter grid; its last test asserts the bounds stay inert for a realistic cube, which is what keeps existing fits bit-identical.

Neither grid is guaranteed ascending -- a negative `CDELT` descends, and a velocity or wavelength axis always does, since both run opposite to frequency. `FitFunc.ascending` exists for that: `FitBSpline` and `FitGCVSpline` route their fitting input through it because `splrep` and `make_smoothing_spline` both require strictly increasing x. Only the *input* is reordered; `splev` and a smoothing spline evaluate at whatever order they are handed, so the fit still comes back on the cube's own grid, channel for channel. Anything channel-indexed must reverse together -- data, weights and mask -- which is what `tests/test_main.py::TestDescendingFrequencyGrid` pins. `FitPolynomial` and the median filters need none of this: `polyfit` is order-agnostic and the filters are index-based.

`FITS_SPECSYS` (FITS keyword names) and `FRAME_CODES` (MS integer codes) describe the same frames for different formats and must not be substituted for one another.

Constants and the composition of conversion steps are taken from casacore's `MeasTable`/`MCFrequency`, so grids agree with CASA. Two details are load-bearing:

- The topocentric-to-barycentric step is a **plain Newtonian projection** of the observer's velocity, *not* astropy's `radial_velocity_correction`. The latter carries relativistic terms casacore omits and would offset results from CASA by ~4.7 m/s. `tests/test_doppler.py::TestCasacoreAgreement` pins this against casacore's own measures engine (it skips unless casacore measures data is installed).
- The fit happens on the native topocentric grid and only the residual is regridded — the opposite order to `mstransform`. The continuum structure being modelled is stationary in topocentric frequency.

Since the channel count changes, this path requires `--output-ms`.

`vis-mowjsub --output-column` is deliberately **required with no default**. It used to default to `LINE_DATA`, which is not an MS-standard column name; don't reintroduce a default. The same reasoning makes both `--input-column` and `--output-column` required on `doppler-mowjsub` — an MS written by `vis-mowjsub --output-ms` contains no `DATA` column at all (`output_ms_dataset` drops `CHANNEL_COLUMNS` and re-adds only what the caller passes), while an in-place run leaves `DATA` holding the raw visibilities, so any default is wrong in one of the two cases. Writing the input column back in place is refused.

### Writing a new MS

`xds_to_table` writes only the columns it is handed and creates no subtables, so any `--output-ms` path (Doppler or not) goes through two shared helpers in `utils.py`:

- `output_ms_dataset` — carries every row-shaped column of the input across and restores `FIELD_ID`/`DATA_DESC_ID`, which dask-ms groups on and therefore exposes as attrs rather than columns. The caller supplies only the per-channel columns.
- `copy_ms_subtables` — copies the subtables over with `casacore.tables`. `finalise_regridded_ms` calls it and then rewrites `SPECTRAL_WINDOW` for the new grid.

Three casacore quirks are worked around there and are easy to reintroduce: open tables with `lockoptions="auto"` (the default user locking leaves dask-ms-cached tables unlocked), address subtables as `path/SUBTABLE` rather than `ms::SUBTABLE` (the latter inherits the parent's read-only access), and close the handle `copy()` returns.

### Fitting functions (`mowjsub/fitfuncs.py`)

All fitters inherit from `FitFunc` and implement `fit(data, mask, weights) -> np.ndarray`. Mask is a boolean array where `True` = excluded from fit. The `prepare()` method must be called before `fit()` — it converts `velwidth` (km/s) to channel count via `utils.chans_in_velwidth`.

Available fitters:
- `FitBSpline` — B-spline via scipy `splrep`/`splev`, knots placed at random offsets for robustness
- `FitGCVSpline` — Smoothing spline via `make_smoothing_spline` (GCV penalty)
- `FitMedFilter` / `FitMedFilterFast` — Sliding median filter; Fast variant uses `scipy.ndimage.median_filter` with NaN interpolation
- `FitPolynomial` — `numpy.polyfit`
- `FitDCT` — DCT-based filter (runs `FitMedFilterFast` internally then zeroes low-amplitude DCT coefficients)

`BadFitError` is raised when a spectrum has fewer valid points than `cont_fit_tol`; the caller sets those spectra to NaN.

### Image-plane execution flow

```
im_mowjsub.py:runit
  └─ zds_from_fits()          # FITS → xr.Dataset, via fitstoolz.FitsData
  └─ [optional] get_automask() # sigma-clip automask using ContSub + PixSigmaClip
  └─ da.gufunc(ContSub.fitContinuum)  # Dask parallel over RA blocks
  └─ line = data − continuum  # an array op, in the file's own axis order
  └─ write_cubes()            # both cubes, one pass over the graph
```

`ContSub.fitContinuum` iterates over all `(ra, dec)` pixels, calling `fitfunc.fit` on each 1D spectrum.

**Both cubes are written in one `da.store`, and that is load-bearing.** They share the per-pixel fit, which is the expensive part of a run; a `writeto` each makes dask walk that graph twice and refit every spectrum — measured at 2x on a 128×128×256 cube. `utils.write_cubes` reserves each output on disk with `allocate_fits` and streams into the memory map, so neither cube is held whole either. If you add a third output derived from the same fit, add it to the same `write_cubes` call rather than writing it separately.

This replaced a `subtract_fits` that took *paths*: it wrote the continuum, then read it straight back alongside the input to form the residual from arrays the caller already had. The Doppler path had to stage a topocentric continuum in `{prefix}-cont-topo.fits` and delete it in a `finally`, purely to satisfy that. Both are gone, and with them the last use of `xarray-fits`, which is no longer a dependency.

### CLI parameters: the signature is the schema

Each entry point is a `@shinobi.pystep` (Stimela 3) whose **typed signature is the single schema authority**. To add a CLI option, add a parameter to that function — there are no YAML schemas and no Click decorators to keep in step.

```python
@shinobi.pystep(name=app, info="...")
def im_mowjsub(
    input_image: Path = Field(..., description="Input image"),
    hdu_index: int = Field(0, description="...", json_schema_extra={"abbreviation": "hi"}),
    fit_model: FIT_MODELS = Field("b-spline", description="..."),
) -> None:
    return runit(SimpleNamespace(**locals()))
```

The scabha equivalents map onto it directly: `dtype: File` → `Path`, `choices:` → a `Literal`, `required: yes` → `Field(...)`, `abbreviation:` → `json_schema_extra`, `policies.repeat` → a `list[...]` annotation. `policies.positional` is the one that is not a type: `parser/_cli.py:make_command(step, positional=...)` renders that field as a `click.Argument`, since shinobi's `build_options` only emits `--options`.

Each module exposes three names: `runit(opts)` does the work against a namespace, `step` is the `StepRef` (so a Stimela 3 recipe can use it directly), and `command` is the `click.Command` the console script points at. **The console scripts point at `command`, not `runit`** — `runit` is a plain function now, so tests drive `command` through `CliRunner`.

The callback dispatches through shinobi's `_dispatch`, which propagates an exception from inside the step with its message intact — the error-path tests read those messages, so don't swap it for a bare call.

The Stimela 2 cab YAMLs under `mowjsub/stimelating/` are gone. A pystep *is* a step, so the cabs were a second copy of every parameter with nothing keeping the two honest. A Stimela 2 pipeline that `_include`d `(mowjsub.stimelating)mowjsub_cabs.yaml` needs updating to use the steps directly.

### Logging

One logger, named `mowjsub` (`LOGGER`). Library modules take it with `logging.getLogger` and never attach a handler; an entry point calls `mowjsub.set_logger(level)` once. It used to be a logger per entry point via `scabha.init_logger`, which meant `utils` — shared by all three commands — logged to the image-plane logger whichever command you had actually run.

### Key dependencies

- `stimela-ninja` (`shinobi`) — Stimela 3: `pystep`, `clickutil.build_options`, step dispatch
- `pydantic` — the parameter models behind those signatures
- `fitstoolz` — FITS I/O (`FitsData`); reading cubes, axis naming by WCS
- `dask-ms` — Measurement Set I/O
