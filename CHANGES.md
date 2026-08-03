# Changelog

Notable changes to mowjsub. Versions before 2.0 were released under the
project's former name, **contsub**.

## Unreleased (2.0rc2)

### Changed

- **The CLI is built on Stimela 3 (`stimela-ninja`/shinobi) instead of scabha.** Each
  entry point is now a `@shinobi.pystep` whose typed signature is the single schema
  authority, so the same function backs both the command line and a Stimela 3 recipe
  step. The flags, their abbreviations, defaults and choices are unchanged — this is a
  change of machinery, not of interface.

  - **The YAML parameter schemas are gone** (`mowjsub/parser/*.yaml`), along with
    `scabha` and `omegaconf` as dependencies. Parameters are pydantic `Field`s;
    `dtype: File` became `Path`, `choices:` a `Literal`, `abbreviation:` a
    `json_schema_extra` entry, and `policies.positional` an argument to
    `parser/_cli.py:make_command`.
  - **The Stimela 2 cab YAMLs are gone** (`mowjsub/stimelating/`). A pystep is itself a
    step, so the cabs were a second copy of every parameter with nothing keeping the two
    in agreement. **A Stimela 2 pipeline that `_include`d
    `(mowjsub.stimelating)mowjsub_cabs.yaml` will break** and needs to use the steps
    directly.
  - **`mowjsub.parser.<x>.runit` is a plain function** taking an options namespace, not a
    `click.Command`. Each module exposes `runit`, `step` (the `StepRef`) and `command`
    (the `click.Command`); the console scripts point at `command`.
  - `--loglevel` is now honoured rather than accepted and ignored.

### Fixed

- **An `auto` Doppler channel grid no longer loses a channel to floating point.**
  `common_channel_grid` derives the output length by dividing the common span by
  the channel width and flooring. When every timestamp shares one Doppler factor
  — always true of a cube, and effectively true of a short track — that factor
  cancels out of the ratio, so the span is a whole number of channels in real
  arithmetic. Multiplying ~1e9 Hz by the factor first leaves the division a few
  ULPs either side of the integer, and on the low side `floor` charged a whole
  channel for the rounding.

  The result was an `auto` grid of `nchan_in - 3` instead of the intended
  `nchan_in - 2`, for **about half of all factors**, with no way to predict
  which. The ratio is now snapped to the integer when it is within 1e-6 of one;
  the observed error is ~4e-13 channels, and a real drift falls whole channels
  short, not a millionth of one.

  **This changes output channel counts.** A grid that came out one channel short
  now comes out at its intended width — one channel wider than the same run
  produced before. No data moves: the grid was guard-trimmed either way, and
  both edges stay strictly inside the shifted band. `vis-mowjsub` and
  `im-mowjsub` are affected identically, as is `doppler-mowjsub`.

### Changed

- **`vis-mowjsub --output-column` no longer defaults to `LINE_DATA`.** It is now
  required. `LINE_DATA` is not an MS-standard column name, and defaulting to it
  pushed a non-standard name into every tool downstream of the result. There is
  no standard name for continuum-subtracted visibilities, so mowjsub no longer
  invents one; pass `--output-column DATA` for the common case of feeding an
  imager directly. This is a breaking CLI change, taken during the 2.0 release
  candidates rather than after.

  Dropping the default exposed a hazard it had been hiding: without
  `--output-ms` the residual is written back into the input MS, so
  `--output-column DATA` there would overwrite the visibilities the fit was made
  from. That combination is now refused.

- **An already-regridded MS is refused rather than corrected twice.**
  `doppler_regrid_dataset` checks `SPECTRAL_WINDOW::MEAS_FREQ_REF` and requires a
  topocentric input grid. This closes a live hole in `vis-mowjsub
  --doppler-frame`, which would previously have applied the frame conversion on
  top of one CASA `mstransform` had already applied.

### Added

- **`doppler-mowjsub`, the Doppler correction as a standalone command.** The same
  `--doppler-*` parameters `vis-mowjsub` takes, over an MS whose continuum has
  already been subtracted. This lets a pipeline run continuum subtraction and the
  frame transformation as separate stages while keeping them in the right order —
  previously the correction was only reachable *through* a contsub run, so a
  workflow that wanted the two separate had to run the regrid elsewhere, and if
  it landed first the continuum was then fitted across an interpolated grid.
  `vis-mowjsub --doppler-frame` is unchanged and remains the convenient default;
  `tests/test_doppler.py::TestStandaloneDoppler::test_matches_the_fused_single_pass`
  pins the two against each other.

- **`im-mowjsub --doppler-frame`, Doppler correction in the image plane.**
  Applied to both output cubes after the fit, so it is one interpolation at the
  very end and nothing downstream inherits a correlated channel grid. The catch is
  physical, not implementational: the Doppler factor is time-dependent, and a cube
  has already been integrated over time, so only a single factor can be applied
  and the intra-track smearing is unrecoverable. Safe when the drift over the
  observation is much smaller than a channel; `--doppler-obs-duration` makes
  mowjsub compute the drift, log it against the channel width and warn past a
  tenth of a channel. Unambiguously right for placing several observations on one
  grid for stacking, which is a pure shift per cube. Cube metadata is resolved
  from the header with `--doppler-time`, `--doppler-telescope` and
  `--doppler-phase-centre` as overrides. Requires the spectral axis on `NAXIS3`.

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

- **`chans_in_velwidth` reference frequency and rounding.** A channel's velocity
  width is `dv = c·df/f`, so it varies as 1/f across the band; the reference
  frequency is now stated explicitly as the band centre instead of being implied
  by an average of two edge estimates. Three real defects went with that: the
  upper edge was selected with `np.partition(freqs, 2)[-2:]`, which does **not**
  return the two largest elements (partition only guarantees position `kth`) and
  silently picked arbitrary channels on any grid that was not already sorted;
  the channel count was truncated with `int()` rather than rounded, biasing every
  conversion downwards by up to a full channel; and `c` was the rounded
  `2.998e8` instead of the exact value already defined in `doppler.py`.

  **This changes results.** On a 1000-channel L-band grid, 300 km/s now converts
  to 210 channels where it previously gave 209 (the true value is 209.82).
  Continuum fits using `--velwidth` will differ slightly from 2.0rc1.
- **`test_b_spline`'s flaky tolerance.** `FitBSpline` jitters its knots by up to
  ±25 channels, and the test compared two independently-seeded fitters, so the
  assertion was measuring knot placement rather than the velwidth/chanwidth
  equivalence it claimed to test. The residual MAD reached ~0.05 on roughly 8%
  of datasets, which is why the tolerance had been raised repeatedly. `FitFunc`
  now takes an optional `seed`; with knots held fixed the two paths agree
  *exactly*, so the test asserts equality instead of a tolerance. Verified over
  40 fresh random datasets. Default behaviour is unchanged — `seed=None` still
  draws fresh entropy.
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
- **`pytest` raised to `>=9.0.3,<10`** for GHSA-6w46-j5rx-g56g (tmpdir
  handling) — a floor rather than the previous exact `==8.4.1` pin, so the next
  advisory fix does not need a pyproject edit.
- **`uv.lock` is now tracked**, reversing the earlier decision to ignore it. The
  lock is never read by anyone installing mowjsub — PyPI and
  `pip install git+…` both resolve from `[project.dependencies]` — so this only
  affects this repo's own environments, where it buys reproducible dev setups
  and a CI that cannot change behaviour without a commit. That had already cost
  something: ruff 0.16 turned CI's lint step red with no code change, because
  each run re-resolved from scratch.

  CI's `build` job installs with `uv sync --locked`, which also fails when
  `pyproject.toml` and the lock disagree. Because pinning the lock would
  otherwise hide upstream breakage until someone relocked, a second job,
  `latest`, ignores the lock and resolves fresh on a weekly schedule; it is the
  intended early warning for a new astropy/dask/numpy breaking the package, and
  does not gate pushes.
- **Dependabot configured** (`.github/dependabot.yml`) for the `uv` and
  `github-actions` ecosystems — weekly and monthly respectively. Dev tooling and
  routine runtime bumps are grouped; runtime major bumps are left individual, so
  an astropy or dask major arrives as its own reviewable PR. This only became
  useful once `uv.lock` was tracked.
- **`pip-audit` added to CI and to the pre-commit hook.** CI runs it on every
  matrix entry, against the tree actually installed and before a PR merges,
  which Dependabot (default branch, ingested advisories only) does not cover.
  The hook runs it only when the commit includes `pyproject.toml`, and does not
  block when pip-audit itself fails to run.
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
