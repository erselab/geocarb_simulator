"""Toy check of gauss_newton_state's opt-in speedups (lm_reuse_lin, lm_gain_ratio): same solution, fewer RT-like calls.
    PYTHONPATH=.:<gert> python scripts/test_lm_fast.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocarb_gert.joint_state import gauss_newton_state  # noqa: E402


class Spec:
    def __init__(self, n):
        self.n = n
    def x0(self): return np.ones(self.n)
    def Sa_inv(self): return np.eye(self.n) * 1.0
    def dx_scale(self): return np.ones(self.n)
    def clip_trial(self, x): return x
    free_params = ()


def run(n, m, seed, **kw):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(m, n)) / np.sqrt(n)
    xt = 1.0 + 0.8 * rng.normal(size=n)
    f = lambda x: M @ x + 0.4 * np.tanh(M @ (x ** 2) * 0.5) + 0.3 * (M @ x) ** 2
    def jac(x):
        h = 1e-6
        K = np.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(n)], axis=1)
        cnt["lin"] += 1
        return f(x), K, {}
    def fwd(x):
        cnt["fwd"] += 1
        return f(x)
    y = f(xt)
    cnt = {"fwd": 0, "lin": 0}
    x = gauss_newton_state(fwd, y, Spec(n), np.full(m, 100.0), verbose=False, jacobian_fn=jac, max_iter=40, tol=1e-8, **kw)
    return x, cnt


for seed in range(4):
    x0, c0 = run(20, 60, seed)
    x1, c1 = run(20, 60, seed, lm_reuse_lin=True, lm_gain_ratio=True)
    print(f"seed {seed}: default fwd {c0['fwd']:3d} lin {c0['lin']:3d} | fast fwd {c1['fwd']:3d} lin {c1['lin']:3d} | "
          f"|dx| between solutions {np.linalg.norm(x0 - x1):.2e} (|x| {np.linalg.norm(x0):.2f})")
