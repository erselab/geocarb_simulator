#!/usr/bin/env python3
"""Direct test of the "perfect representability" claim: with truth built
on the retrieval's OWN bin-centers grid (whole_slit_bin_centers) and
prior_fields=exact (prior == truth at every bin center), the forward
model evaluated AT THE PRIOR (x = x_a, before any GN step) should
reproduce the rendered scene EXACTLY -- resid = y_true - forward(x_a)
should be ~0 (floating-point noise), not just small relative to the
noise level. Checks this for EVERY window (not just one), against the
tiling-fixed (correctly-tiled) binceil truth, to see whether it holds
everywhere or only away from window seams.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

captured = []  # list of dicts, one per window


def _capture_gn(forward, y_true, spec, Sy_inv_diag, *args, **kwargs):
    x_a = spec.x0()  # ones, since every free row here is kind="scale"
    resid_at_prior = np.asarray(y_true, dtype=float) - forward(x_a)
    sigma = 1.0 / np.sqrt(np.maximum(np.asarray(Sy_inv_diag, dtype=float), 1e-300))
    captured.append(dict(
        label=kwargs.get("label", ""),
        rms=float(np.sqrt(np.mean(resid_at_prior ** 2))),
        max_abs=float(np.max(np.abs(resid_at_prior))),
        sigma_mean=float(np.mean(sigma)),
        n=resid_at_prior.size,
    ))
    n = spec.n_free
    return (np.ones(n), np.eye(n), np.eye(n)) if kwargs.get("return_avk") else np.ones(n)


sweep.gauss_newton_state = _capture_gn

# Thread-based dummy pool so `captured` (appended to by every window,
# across workers) is visible back in this process -- see noise_check_
# window_416.py's own note on why fork-based multiprocessing hides this.
import multiprocessing.dummy as _mp_dummy  # noqa: E402


class _FakeCtx:
    @staticmethod
    def Pool(n):
        return _mp_dummy.Pool(n)


sweep.mp = type("mp_shim", (), {"get_context": staticmethod(lambda name: _FakeCtx())})

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--prior-fields", "exact", "--n-windows", "58", "--anchor-density", "16",
    "--resolution-matched-g-ratio-bins", "1", "--hires-only", "--jacobian", "analytic",
    "--prior-form", "exponential", "--overlap", "2", "--n-workers", "4",
    "--out", str(REPO_ROOT / "scratch_work" / "repr_check_binceil_out.pkl"),
]

sweep.main()

print(f"\n{len(captured)} windows captured")
rms_all = np.array([c["rms"] for c in captured])
max_all = np.array([c["max_abs"] for c in captured])
sigma_all = np.array([c["sigma_mean"] for c in captured])

print(f"\nresid-at-prior RMS across windows: min={rms_all.min():.4g} max={rms_all.max():.4g} "
     f"mean={rms_all.mean():.4g}")
print(f"typical sigma (noise) across windows: mean={sigma_all.mean():.4g}")
print(f"resid-at-prior RMS / sigma ratio: min={(rms_all/sigma_all).min():.4g} "
     f"max={(rms_all/sigma_all).max():.4g} mean={(rms_all/sigma_all).mean():.4g}")

order = np.argsort(-rms_all)
print(f"\nworst 10 windows by resid-at-prior RMS:")
for i in order[:10]:
    c = captured[i]
    print(f"  {c['label']:20s} rms={c['rms']:.4g}  max_abs={c['max_abs']:.4g}  "
         f"sigma_mean={c['sigma_mean']:.4g}  rms/sigma={c['rms']/c['sigma_mean']:.4g}")

print(f"\nbest 5 windows by resid-at-prior RMS (should show near-zero if representability holds):")
for i in order[-5:]:
    c = captured[i]
    print(f"  {c['label']:20s} rms={c['rms']:.4g}  max_abs={c['max_abs']:.4g}  "
         f"sigma_mean={c['sigma_mean']:.4g}  rms/sigma={c['rms']/c['sigma_mean']:.4g}")
