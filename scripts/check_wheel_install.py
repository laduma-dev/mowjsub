"""Smoke check an installed wheel, from outside the source checkout.

Run with the target environment's Python. The base check deliberately fails if
any transitive dependency brings Stimela back; do not mask it with --no-deps.
"""

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument("--ninja", action="store_true")
args = parser.parse_args()
installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in importlib.metadata.distributions()}
for name in ("im_mowjsub", "vis_mowjsub", "doppler_mowjsub"):
    module = importlib.import_module("mowjsub.parser." + name)
    assert callable(module.runit)
    for flag in ("--help", "--version"):
        subprocess.run([name.replace("_", "-"), flag], check=True, stdout=subprocess.DEVNULL)
    if args.ninja:
        from shinobi.steps.schema import StepRef

        assert isinstance(module.step, StepRef)
        assert getattr(module, name) is module.step
        for target in ("step", name):
            subprocess.run(["ninja", "run", f"{module.__name__}:{target}", "--help"], check=True, stdout=subprocess.DEVNULL)
with tempfile.TemporaryDirectory(prefix="mowjsub-wheel-smoke-") as temporary:
    work = Path(temporary)
    cube = work / "input.fits"
    header = fits.Header()
    for axis, ctype, unit, start, step in ((1, "RA---SIN", "deg", 201.0, -0.001), (2, "DEC--SIN", "deg", -43.0, 0.001), (3, "FREQ", "Hz", 1.36e9, 6.5e3)):
        for key, value in (("CTYPE", ctype), ("CUNIT", unit), ("CRVAL", start), ("CRPIX", 1.0), ("CDELT", step)):
            header[f"{key}{axis}"] = value
    data = np.broadcast_to(np.linspace(1, 2, 64)[:, None, None], (64, 2, 2)).astype(np.float32).copy()
    fits.writeto(cube, data, header)
    prefix = work / "native"
    subprocess.run(["im-mowjsub", str(cube), "--fit-model", "polynomial", "--order", "1", "--output-prefix", str(prefix)], check=True, cwd=work, stdout=subprocess.DEVNULL)
    with fits.open(f"{prefix}-line.fits") as residual:
        np.testing.assert_allclose(residual[0].data, 0, atol=1e-6)
    ms = work / "input.ms"
    ms.mkdir()
    for command, options in (
        ("vis-mowjsub", ["--output-column", "LINE", "--fit-model", "polynomial", "--order", "1"]),
        ("doppler-mowjsub", ["--input-column", "LINE", "--output-column", "DATA", "--doppler-frame", "bary"]),
    ):
        result = subprocess.run([command, str(ms), *options, "--output-ms", str(ms)], cwd=work, capture_output=True, text=True)
        assert result.returncode != 0
        assert "is the MS being read" in result.stderr, result.stderr
    if args.ninja:
        config = work / "ninja.yml"
        config.write_text("cache:\n  enabled: true\nprovenance:\n  enabled: true\n")
        for target in ("step", "im_mowjsub"):
            command = [
                "ninja",
                "--config",
                str(config),
                "run",
                f"mowjsub.parser.im_mowjsub:{target}",
                "--input-image",
                str(cube),
                "--fit-model",
                "polynomial",
                "--order",
                "1",
                "--output-prefix",
                str(work / target),
            ]
            for _ in range(2):
                subprocess.run(command, check=True, cwd=work, stdout=subprocess.DEVNULL)
        manifests = [json.loads(path.read_text()) for path in (work / ".shinobi/runs").glob("*.run.json")]
        assert manifests, "Ninja wrote no provenance manifests"
        assert all(manifest["returncode"] == 0 for manifest in manifests)
        assert any(manifest["root"]["cached"] for manifest in manifests), "Repeated Ninja run did not use its cache"
        assert {manifest["target"] for manifest in manifests} == {"mowjsub.parser.im_mowjsub:step", "mowjsub.parser.im_mowjsub:im_mowjsub"}
        for target in ("step", "im_mowjsub"):
            assert (work / f"{target}-cont.fits").exists()
            assert (work / f"{target}-line.fits").exists()
if not args.ninja:
    assert "shinobi" not in sys.modules
    unwanted = installed & {"stimela-ninja", "shinobi", "scabha", "stimela"}
    assert not unwanted, f"Base wheel pulls in the optional framework: {sorted(unwanted)}"
print("Installed wheel smoke checks passed")
