# Security Policy

## Supported versions

mowjsub is pre-release software (currently `2.0rc2`). Security fixes are applied
to the latest revision only; there are no long-term-support branches.

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report vulnerabilities privately by email to **a0069067@wits.ac.za**, or via
GitHub's [private vulnerability reporting][ghsa] on this repository if enabled.
Include enough detail to reproduce — affected version, which entry point
(`im-mowjsub` or `vis-mowjsub`), the input data or config involved, and the
impact you observed.

We aim to acknowledge reports within a reasonable time and work with you on a
fix.

[ghsa]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

## Security posture

mowjsub is a data-processing library: it reads FITS cubes and CASA Measurement
Sets, fits and subtracts a continuum, and writes the results back out. It runs
no subprocesses, opens no network connections, and does not `eval()`/`exec()`
anything.

The inputs it does trust are worth naming, since they are where a problem would
most plausibly live:

- **CLI parameter schemas** (`src/mowjsub/parser/*.yaml`) and stimela cab
  definitions are repo-shipped configuration parsed via scabha/OmegaConf, not
  user-supplied documents. If you are loading a *third-party* parameter file
  through these paths, treat it with the same care as any executable config.
- **FITS and MS inputs** are parsed by astropy, xarray-fits, dask-ms and
  casacore. A malformed or hostile file is handled by those libraries; report
  parser vulnerabilities to them, and to us if mowjsub's handling makes an
  otherwise-contained problem worse.
- **Output paths** (`--output-ms` and the `*-cont.fits`/`*-line.fits` siblings)
  are written where you point them, and `copy_ms_subtables` copies subtables
  with `casacore.tables`. mowjsub does not sandbox these writes; don't run it
  with more filesystem privilege than the job needs.

If you find a way around any of this, it's a security issue; please report it as
above.
