# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mowjsub` is a Python library for radio astronomy continuum subtraction, supporting both image-plane (FITS cubes) and visibility-plane (Measurement Set) workflows. It exposes two CLI entry points: `im-mowjsub` and `vis-mowjsub`.

## Commands

This project uses `uv` for dependency management (a `uv.lock` is present). A `.venv` is already set up at `.venv/`.

```bash
# Install with dev/test extras
uv sync --group tests

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_main.py::TestFitsFunc::test_b_spline

# Lint
uv run ruff check mowjsub/

# Format
uv run ruff format mowjsub/

# Run the image-plane CLI
uv run im-mowjsub <input.fits> [options]

# Run the visibility-plane CLI
uv run vis-mowjsub [options]
```

Pre-commit hooks run ruff check + ruff format automatically on commit (`pre-commit install` to activate).

Linting config is in `ruff.toml`: line length 180, target Python 3.11, isort enabled.

## Architecture

### Two processing pipelines

**Image plane** (`im-mowjsub`): Operates on FITS spectral cubes. Loads the cube via `xarray-fits` into an `xr.Dataset` with dims `[ra, dec, spectral]`, chunks along RA using Dask, fits a continuum baseline per-pixel per-spectrum, and writes two FITS outputs: `*-cont.fits` (continuum model) and `*-line.fits` (residual). Entry point: `mowjsub/parser/im_mowjsub.py:runit`.

**Visibility plane** (`vis-mowjsub`): Operates on CASA Measurement Sets. Reads via `dask-ms`, reshapes row-based data to `[time, baseline, freq, corr]`, fits per-baseline-per-correlation, and writes the result back to an MS column. Optionally applies a Doppler correction on the way out (see below). Entry point: `mowjsub/parser/vis_mowjsub.py:runit`.

### Doppler correction (`mowjsub/doppler.py`)

`--doppler-frame` resamples the continuum-subtracted visibilities onto a channel grid fixed in a chosen spectral frame, replacing the `regridms=True` half of a CASA `mstransform` pass. `doppler.py` is pure NumPy/astropy: frame apex velocities, per-timestamp conversion factors, common-grid derivation, and channel resampling. The dask orchestration and MS I/O live in `utils.py` (`doppler_regrid_dataset`, `finalise_regridded_ms`, `observatory_location`).

Constants and the composition of conversion steps are taken from casacore's `MeasTable`/`MCFrequency`, so grids agree with CASA. Two details are load-bearing:

- The topocentric-to-barycentric step is a **plain Newtonian projection** of the observer's velocity, *not* astropy's `radial_velocity_correction`. The latter carries relativistic terms casacore omits and would offset results from CASA by ~4.7 m/s. `tests/test_doppler.py::TestCasacoreAgreement` pins this against casacore's own measures engine (it skips unless casacore measures data is installed).
- The fit happens on the native topocentric grid and only the residual is regridded — the opposite order to `mstransform`. The continuum structure being modelled is stationary in topocentric frequency.

Since the channel count changes, this path requires `--output-ms`.

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
  └─ zds_from_fits()          # FITS → xr.Dataset with FREQS coord
  └─ [optional] get_automask() # sigma-clip automask using ContSub + PixSigmaClip
  └─ da.gufunc(ContSub.fitContinuum)  # Dask parallel over RA blocks
  └─ subtract_fits()          # data − continuum → line cube
```

`ContSub.fitContinuum` iterates over all `(ra, dec)` pixels, calling `fitfunc.fit` on each 1D spectrum.

### CLI parameter schemas

CLI parameters are defined in YAML schemas under `mowjsub/parser/`:
- `im_mowjsub.yaml` — image-plane parameters
- `vis_mowjsub.yaml` — visibility-plane parameters

These are loaded via `scabha.schema_utils.paramfile_loader` and converted to Click options via `clickify_parameters`. To add a new CLI option, edit the appropriate YAML schema — do not add Click decorators manually.

The stimela integration (`mowjsub/stimelating/`) exposes both tools as stimela cabs (`mowjsub_cabs.yaml`), with parameter files in `im_mowjsub_param.yaml` and `vis_mowjsub_param.yaml`.

### Key dependencies

- `scabha` — logging (`init_logger`), CLI schema utilities (`clickify_parameters`, `paramfile_loader`)
- `dask-ms` — Measurement Set I/O
- `xarray-fits` (`xarrayfits`) — FITS I/O into xarray
- `stimela` — workflow orchestration (optional, for cab-based pipelines)
- `omegaconf` — config object from CLI kwargs
