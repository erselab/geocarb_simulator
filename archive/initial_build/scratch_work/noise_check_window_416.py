#!/usr/bin/env python3
"""Capture the real y_true/Sy_inv_diag gauss_newton_state receives for
window [416,440] (index 38) WITHOUT running the actual (expensive) GN
solve -- monkeypatch gauss_newton_state to just record its args and
return immediately, then report the noise model's implied sigma/SNR
against the real scene signal in this window.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model, RADIOMETRIC_SPEC_BY_FPA  # noqa: E402

captured = {}


def _capture_gn(forward, y_true, spec, Sy_inv_diag, *args, **kwargs):
    captured["y_true"] = np.asarray(y_true, dtype=float).copy()
    captured["Sy_inv_diag"] = np.asarray(Sy_inv_diag, dtype=float).copy()
    captured["label"] = kwargs.get("label", "")
    n = spec.n_free
    return (np.ones(n), np.eye(n), np.eye(n)) if kwargs.get("return_avk") else np.ones(n)


sweep.gauss_newton_state = _capture_gn

# The real driver solves via a fork-based multiprocessing.Pool, so writes
# `_capture_gn` makes to `captured` land in a forked CHILD's own copy-on-
# write memory, invisible back in this (parent) process once the pool
# returns -- only _solve_window's own RETURN VALUE round-trips via the
# pool's pickling. Swap in a thread-based dummy pool instead: same process,
# real shared memory, so `captured` updates directly.
import multiprocessing.dummy as _mp_dummy  # noqa: E402


class _FakeCtx:
    @staticmethod
    def Pool(n):
        return _mp_dummy.Pool(n)


sweep.mp = type("mp_shim", (), {"get_context": staticmethod(lambda name: _FakeCtx())})

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--prior-fields", "structural", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-anchor-density", "4", "--hires-only", "--jacobian", "analytic",
    "--prior-form", "exponential", "--overlap", "2", "--n-workers", "1",
    "--task-id", "38", "--n-tasks", "58",
    "--out", str(REPO_ROOT / "scratch_work" / "noise_check_out.pkl"),
]

try:
    sweep.main()
except Exception as e:
    print(f"main() raised {e!r} (expected -- captured what we need already)")

FPA = 2
y_true = captured.get("y_true")
Sy_inv_diag = captured.get("Sy_inv_diag")
if y_true is None:
    print("FAILED to capture y_true/Sy_inv_diag -- gauss_newton_state was never called "
          "(check label filter / task routing)")
    raise SystemExit(1)

print(f"\ncaptured for label={captured['label']!r}, n_pixels={y_true.size}")
print(f"RADIOMETRIC_SPEC_BY_FPA[{FPA}] = {RADIOMETRIC_SPEC_BY_FPA[FPA]}")

nm = geocarb_noise_model(FPA)
print(f"LinearShotNoise: N0={nm.N0:.6g}  N1={nm.N1:.6g}  I_max={nm.I_max}")

sigma_direct = nm.sigma([y_true], [None])
sigma_from_Sy = 1.0 / np.sqrt(np.maximum(Sy_inv_diag, 1e-300))

print(f"\nscene signal y_true: min={y_true.min():.4g} max={y_true.max():.4g} "
      f"mean={y_true.mean():.4g} median={np.median(y_true):.4g}")
print(f"sigma (radiance units), from noise model direct on y_true:")
print(f"  min={sigma_direct.min():.4g} max={sigma_direct.max():.4g} "
     f"mean={sigma_direct.mean():.4g} median={np.median(sigma_direct):.4g}")
print(f"sigma (radiance units), BACKED OUT from the actual Sy_inv_diag used in the solve:")
print(f"  min={sigma_from_Sy.min():.4g} max={sigma_from_Sy.max():.4g} "
     f"mean={sigma_from_Sy.mean():.4g} median={np.median(sigma_from_Sy):.4g}")
print(f"max |sigma_direct - sigma_from_Sy|: {np.max(np.abs(sigma_direct - sigma_from_Sy)):.4g}")

snr = np.abs(y_true) / np.maximum(sigma_direct, 1e-300)
print(f"\nimplied SNR = |signal|/sigma:")
print(f"  min={snr.min():.4g} max={snr.max():.4g} mean={snr.mean():.4g} median={np.median(snr):.4g}")
print(f"  10th/50th/90th pct: {np.percentile(snr,10):.4g} / {np.percentile(snr,50):.4g} / {np.percentile(snr,90):.4g}")
