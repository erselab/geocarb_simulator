#!/usr/bin/env python3
"""Standalone driver: re-solve window [416,440] (index 38 of the standard
58-window g1/ad4/structural tiling) with gauss_newton_state's own verbose
iteration trace forced on, to see whether the wild co2/p_surface posterior
is a genuine converged (if near-degenerate) MAP point or GN oscillation/
divergence. Reuses the exact same sbatch args job 1790501 used.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from geocarb_gert.joint_state import gauss_newton_state as _orig_gn  # noqa: E402


def _verbose_gn(*args, **kwargs):
    kwargs["verbose"] = True
    return _orig_gn(*args, **kwargs)


sweep.gauss_newton_state = _verbose_gn

sys.argv = [
    "gd_joint_block_whole_slit_sweep.py",
    "--g-ratio", "1", "--free", "co2_ppm,p_surface_hpa,albedo", "--vary-albedo",
    "--prior-fields", "structural", "--n-windows", "58", "--anchor-density", "4",
    "--resolution-matched-anchor-density", "4", "--hires-only", "--jacobian", "analytic",
    "--prior-form", "exponential", "--overlap", "2", "--n-workers", "1",
    "--task-id", "38", "--n-tasks", "58",
    "--out", "/scratch/scrowel3_lab/geocarb_simulator/scratch_work/trace_window_416_lm_out.pkl",
]

raise SystemExit(sweep.main())
