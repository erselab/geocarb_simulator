"""Phase-1 gate for the multi-band joint-block state (docs/MULTIBAND_PLAN.md Sec.5):
StateSpec views round-trip, Jacobian column embedding, independent per-band albedo.
No aerosol (user, 2026-09-20). Exit status 1 on any failure.

    PYTHONPATH=.:<gert> python scripts/check_multiband_state.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from geocarb_gert import along_slit_scene as als, GEOCARB_BANDS  # noqa: E402
from geocarb_gert.joint_state import state_spec_from_scene  # noqa: E402
from geocarb_gert.multiband import BandRef, MultiBandState, joint_spec_from_scene  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(0)
bins = np.linspace(-0.30, -0.22, 13)
bands = [BandRef(0, GEOCARB_BANDS[0][0]), BandRef(2, GEOCARB_BANDS[2][0])]
FREE = ("p_surface_hpa", "h2o_surface_vmr", "t_offset_k", "albedo")
kw = dict(free=FREE, prior_form="exponential", fields=als.PRIOR_FIELD_SETS["structural"],
          surface_fields=als.SURFACE_PRIOR_FIELD_SETS["structural"], surface_positions=bins,
          kinds=gjr.ROW_KINDS, gamma=3.0, row_bounds=gjr.ROW_BOUNDS, sigmas=gjr.ROW_SIGMAS)
mb = joint_spec_from_scene(bins, bands, **kw)
J = mb.joint

# 1 -- joint structure
names = [p.name for p in J.params]
check("joint has one row per shared variable and one albedo row per band",
      names.count("p_surface_hpa") == 1 and "albedo_O2_A" in names and "albedo_CO2_strong" in names
      and "albedo" not in names, str(names))
single = {b.label: state_spec_from_scene(bins, band_label=b.label, **kw) for b in bands}
check("joint n_free = shared free + 2 albedo rows",
      J.n_free == single[bands[0].label].n_free + single[bands[1].label]["albedo"].n,
      f"{J.n_free}")

# 2 -- per-band albedo rows carry that band's own prior/sigma/corr length (single-band values)
for b in bands:
    a, s = J[b.row], single[b.label]["albedo"]
    check(f"{b.label}: albedo row = that band's single-band row (prior, sigma, corr_length)",
          np.array_equal(a.prior, s.prior) and a.sigma == s.sigma and a.corr_length == s.corr_length
          and a.bounds == s.bounds, f"sigma={a.sigma} corr={a.corr_length}")
check("the two bands' albedo priors differ (independent rows)",
      not np.array_equal(J[bands[0].row].prior, J[bands[1].row].prior))

# 3 -- views: shared rows are the same objects; band albedo appears as 'albedo'
for i, b in enumerate(bands):
    v = mb.view(i)
    vn = [p.name for p in v.params]
    other = bands[1 - i].row
    check(f"view {i}: single-band names, other band's albedo hidden",
          "albedo" in vn and other not in vn and b.row not in vn)
    check(f"view {i}: shared rows are the joint spec's own objects",
          all(v[n] is J[n] for n in vn if n != "albedo"))

# 4 -- round trip of values through index_map for random x
x = J.x0() * (1.0 + 0.05 * rng.standard_normal(J.n_free))
uj = J.unpack(x)
for i, b in enumerate(bands):
    v = mb.view(i)
    uv = v.unpack(mb.x_view(x, i))
    ok = all(np.array_equal(uv[n], uj[b.row if n == "albedo" else n]) for n in uv)
    check(f"view {i}: unpack(x_view) reproduces the joint values for every row", ok)

# 5 -- stacking: forward and Jacobian (nonlinear synthetic forwards) vs finite difference
nd = [40, 55]
A = [rng.standard_normal((nd[i], mb.view(i).n_free)) for i in range(2)]
fwd = [(lambda xv, A=A[i]: np.tanh(A @ xv - A @ mb.view(i).x0() * 0 + 0.1)) for i in range(2)]
jac = [(lambda xv, A=A[i]: (1.0 - np.tanh(A @ xv + 0.1) ** 2)[:, None] * A) for i in range(2)]
fj, Jj = mb.stack_forward(fwd), mb.stack_jacobian(jac)
y = fj(x)
check("stacked forward has the concatenated length", y.size == sum(nd))
Jan = Jj(x)
Jfd = np.zeros_like(Jan)
for k in range(J.n_free):
    h = 1e-6
    xp, xm = x.copy(), x.copy()
    xp[k] += h
    xm[k] -= h
    Jfd[:, k] = (fj(xp) - fj(xm)) / (2 * h)
check("stacked analytic Jacobian matches finite difference of the stacked forward",
      np.allclose(Jan, Jfd, atol=1e-7), f"max |diff|={np.abs(Jan - Jfd).max():.2e}")

# 6 -- independence of the albedo rows
sl = J.slices()
for i, b in enumerate(bands):
    o = bands[1 - i]
    cols = np.arange(sl[o.row].start, sl[o.row].stop)
    rows_b = slice(0, nd[0]) if i == 0 else slice(nd[0], nd[0] + nd[1])
    check(f"band {i} data do not depend on band {1 - i}'s albedo (Jacobian columns are zero)",
          np.all(Jan[rows_b][:, cols] == 0.0))

# 7 -- freezing one band's albedo changes only that band's view
J[bands[1].row].free = False
mb2 = MultiBandState(J, bands)
check("freezing band 1's albedo: band 1 view has albedo frozen, band 0's stays free",
      not mb2.view(1)["albedo"].free and mb2.view(0)["albedo"].free)
check("index_map still maps every free view element to a distinct joint element",
      all(len(set(mb2.index_map(i).tolist())) == mb2.index_map(i).size for i in range(2)))
J[bands[1].row].free = True

# 8 -- stacked noise
check("stack_Sy_inv_diag concatenates",
      mb.stack_Sy_inv_diag([np.ones(3), 2 * np.ones(2)]).tolist() == [1, 1, 1, 2, 2])

print("\nALL CHECKS PASSED" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
