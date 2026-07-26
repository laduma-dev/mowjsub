# Changelog

Notable changes to mowjsub. Versions before 2.0 were released under the
project's former name, **contsub**.

## Unreleased (2.0rc2)

### Added

- **Doppler correction for the visibility plane.** `--doppler-frame` resamples
  continuum-subtracted visibilities onto a channel grid fixed in a chosen
  spectral frame, replacing the `regridms=True` half of a CASA `mstransform`
  pass. `src/mowjsub/doppler.py` holds the pure-NumPy/astropy maths (frame apex
  velocities, per-timestamp conversion factors, common-grid derivation, channel
  resampling); the dask orchestration and MS I/O live in `utils.py`. Constants
  and the composition of conversion steps follow casacore's
  `MeasTable`/`MCFrequency` so the grids agree with CASA, and
  `tests/test_doppler.py::TestCasacoreAgreement` pins that agreement against
  casacore's own measures engine.

  Two choices there are deliberate: the topocentric-to-barycentric step is a
  plain Newtonian projection of the observer's velocity rather than astropy's
  `radial_velocity_correction`, whose relativistic terms casacore omits and
  which would offset results by ~4.7 m/s; and the continuum fit happens on the
  native topocentric grid with only the residual regridded, the opposite order
  to `mstransform`, because the continuum structure being modelled is
  stationary in topocentric frequency. Since the channel count changes, this
  path requires `--output-ms`.

- **Complete `--output-ms` writes.** `xds_to_table` writes only the columns it
  is handed and creates no subtables, so both the Doppler and plain output-MS
  paths now go through `output_ms_dataset` (carries every row-shaped column
  across and restores the `FIELD_ID`/`DATA_DESC_ID` that dask-ms exposes as
  attrs rather than columns) and `copy_ms_subtables` (copies subtables via
  `casacore.tables`). `finalise_regridded_ms` additionally rewrites
  `SPECTRAL_WINDOW` for the new grid.

### Fixed

- **Image-plane out-of-memory failures**, along with fitters that mutated their
  input arrays. The median-filter options are now wired through to the CLI.
- Dead DCT references and stale parser imports removed; the `nworkers`
  documentation corrected.

### Changed

- **`src/` layout.** The package moved from `mowjsub/` to `src/mowjsub/`.
- **Dependency groups consolidated into one `dev` group** holding ruff and
  pytest. The former `tests`/`ruff` split only ever added a way to build a venv
  in which a bare `pytest` fell through to the system interpreter and failed on
  an unrelated import.
- **`stimela` replaced by a direct `scabha>=2.2.0rc2` dependency.** Nothing in
  the package imports stimela; `src/mowjsub/stimelating/` is cab definitions
  only.
- **Pre-commit framework replaced by a tracked git hook.**
  `.githooks/pre-commit` runs `ruff check` and `ruff format --check` over staged
  Python files, enabled per clone with `git config core.hooksPath .githooks`. It
  uses whatever ruff `uv run` resolves instead of fetching its own, which is
  what had drifted: the old `.pre-commit-config.yaml` pinned ruff v0.12.9 while
  the venv resolved 0.16.0.
- **Ruff's rule set is now stated explicitly** in `ruff.toml` (`select`, not
  `extend-select`). Inheriting ruff's defaults meant a ruff release could
  redefine the config's intent, and under 0.16 it did — 24 new violations in
  files nobody had touched.
- Added `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`
  and this changelog.

## 2.0rc1 — 2026-06-26

The first release under the name **mowjsub**, and the first with a
visibility-plane CLI.

### Added

- **`vis-mowjsub`**, the visibility-plane entry point: reads Measurement Sets
  via `dask-ms`, reshapes row-based data to `[time, baseline, freq, corr]`, fits
  per baseline per correlation, and writes back to an MS column. Supports
  writing to a new MS and loading from a cache.
- **New fitting functions** — sliding median filter (`FitMedFilter`,
  `FitMedFilterFast`), DCT-based filtering (`FitDCT`), and polynomial fitting in
  the visibility plane.
- **Stimela integration** (`mowjsub/stimelating/`), exposing both tools as cabs.
- **Unit tests**, and a CI workflow running them across the supported Python
  versions.

### Changed

- **Renamed from `contsub` to `mowjsub`.** The `imcontsub` entry point became
  `im-mowjsub`.
- **Poetry replaced by uv** for dependency management and builds; CI migrated to
  match. `uv.lock` is deliberately untracked.
- **Ruff replaced the previous linting setup.**
- **Python 3.10 dropped, 3.13 added** — the supported range is now
  `>=3.11,<3.14`.
- `FitFunc.prepare()` reworked to accommodate the new fit models.

### Fixed

- Polynomial fit sign error.
- Read the Docs builds, which now install into RTD's own virtualenv.
- B-spline velocity/channel-width comparison now uses median absolute deviation.

## 1.0.5 — 2025-09-01

### Changed

- `--segments` is now required.
- `FREQ` data is returned only when requested.

### Fixed

- Conflicting parameter defaults removed.

## 1.0.4 — 2025-08-30

### Fixed

- Issues #10 and #25.

## 1.0.3 — 2025-06-23

### Added

- Usage documentation and Read the Docs hosting; docs switched to
  reStructuredText.

### Changed

- Build and install reconfigured; docs no longer built through Poetry, and the
  Poetry lock file untracked.

## 1.0.2 — 2025-06-21

Earliest tagged release. Image-plane continuum subtraction (`imcontsub`) with
B-spline fitting, random knot placement and sigma-clip automasking, developed
from the original scripts starting April 2023.
