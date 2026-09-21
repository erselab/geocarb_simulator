"""Exactness (and speed) gate for the sparse detector operator (2026-09-21).

  stage synthetic : predict_neighborhood(active_fn=...) == the dense render, bit for bit, for many
                    anchor-sparsity patterns (single anchor, first/last anchor, blocks, none, all) at
                    rows with different keystone. Cheap; run anywhere.
  stage tile      : the FULL Jacobian K of a real two-band tile computed with the dense operator
                    (jac.SPARSE_L=False) and with the sparse one (True); K must be identical
                    (np.array_equal), plus timings of each linearization.

    python scripts/check_sparse_operator.py synthetic
    ARGS=... python scripts/check_sparse_operator.py tile --fpas 0,2 --tile 12 --free ... --anchor-workers 8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geocarb_gert import gd_render, jacobians as jac  # noqa: E402
from geocarb_gert.focalplane import footprint_active_fn, footprint_average_scene  # noqa: E402
from gert.instrument import ILS  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def synthetic():
    rng = np.random.default_rng(1)
    for fpa, rows in ((2, np.arange(300, 307)), (2, np.arange(20, 27)), (2, np.arange(1000, 1007)), (0, np.arange(900, 907)), (0, np.arange(510, 517)), (0, np.arange(30, 37))):
        from geocarb_gert import GEOCARB_BANDS
        from geocarb_gert.gd_polynomials import real_wavenumber_range
        from geocarb_gert.multiband_geometry import band_tables
        w0, w1 = real_wavenumber_range(fpa, margin_cm1=0.0)
        wc = 0.5 * (w0 + w1)
        wn = np.linspace(wc - 15.0, wc + 15.0, 600)        # 30 cm-1 slice at the band centre: cheap, real signal
        ils = ILS(type="gaussian", fwhm=wc / float(GEOCARB_BANDS[fpa][4]))
        t = band_tables(fpa)
        lo, hi = t["eta_c"][rows.min() - 6], t["eta_c"][rows.max() + 6]
        for n_anchor in (40, 97):
            anchors = np.linspace(lo - 0.01, hi + 0.01, n_anchor)
            base = rng.standard_normal((n_anchor, wn.size))
            i_mid = n_anchor // 2
            patterns = {"single interior": [i_mid], "first anchor": [0], "last anchor": [n_anchor - 1],
                        "second anchor": [1], "third anchor": [2], "third-from-last": [n_anchor - 3],
                        "block of 6": list(range(i_mid, i_mid + 6)),
                        "interior +-1": [i_mid - 1, i_mid + 1], "near the top edge": [i_mid + 9],
                        "two far apart": [3, n_anchor - 4], "none": [], "all": list(range(n_anchor))}
            for pname, idx in patterns.items():
                field = np.zeros_like(base)
                field[idx] = base[idx]
                nz = np.any(field != 0.0, axis=1)
                dense = gd_render.predict_neighborhood(fpa, rows, wn, footprint_average_scene(anchors, field), ils,
                                                       pad=4, footprint=True)
                if nz.any():
                    act = None if nz.all() else footprint_active_fn(anchors, nz)
                    sparse = gd_render.predict_neighborhood(fpa, rows, wn, footprint_average_scene(anchors, field), ils,
                                                            pad=4, footprint=True, active_fn=act)
                else:
                    sparse = np.zeros_like(dense)
                check(f"FPA{fpa} rows {rows[0]}-{rows[-1]}, {n_anchor} anchors, {pname}: sparse == dense (bitwise)",
                      np.array_equal(dense, sparse), f"nonzero pixels {np.count_nonzero(dense)}/{dense.size}")


class MemSampler:
    """Samples the job cgroup's memory.current (bytes) and /dev/shm geocarb_L_<job>* segment sizes."""

    def __init__(self):
        import os, threading
        self.jid = os.environ.get("SLURM_JOB_ID", "")
        self.cg = f"/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{self.jid}/memory.current"
        self.peak_cur, self.peak_shm, self._stop = 0, 0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        import glob, os
        try:
            cur = int(open(self.cg).read())
        except Exception:
            cur = 0
        shm = sum(os.path.getsize(f) for f in glob.glob(f"/dev/shm/geocarb_L_{self.jid}_*"))
        return cur, shm

    def _run(self):
        while not self._stop.is_set():
            c, sh = self._read()
            self.peak_cur, self.peak_shm = max(self.peak_cur, c), max(self.peak_shm, sh)
            self._stop.wait(0.5)

    def __enter__(self):
        self.base = self._read()[0]
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def tile(a):
    import gd_multiband_window as gmw
    import gd_joint_block_retrieve as gjr
    from geocarb_gert.multiband_geometry import build_window_tiles_multiband
    fpas = [int(t) for t in a.fpas.split(",")]
    rows = dict(build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, 2)[a.tile].rows)
    modes = [("dense operator, dense storage (original)", False, False),
             ("sparse operator, dense storage", True, False),
             ("sparse operator, sparse storage", True, True)]
    if a.only_storage:
        modes = [m for m in modes if m[2] or m[1] is False and not m[2]]
        modes = [("sparse operator, dense storage", True, False), ("sparse operator, sparse storage", True, True)]
    out = {}
    for name, sl, ss in modes:
        jac.SPARSE_L, jac.SPARSE_STORE = sl, ss

        def hook(P, name=name):
            x0 = P["mb"].joint.x0()
            with MemSampler() as ms:
                t0 = time.time()
                y, K, _ = P["linearize"](x0)
                dt = time.time() - t0
            out[name] = (y, K, dt, ms.peak_cur - ms.base, ms.peak_shm)
            return True
        gmw.solve_window_multiband(rows, a.free.split(","), g_ratio=1.0, anchor_mode="cover",
                                   anchor_workers=a.anchor_workers, hook=hook, verbose=False)
        y, K, dt, dmem, shm = out[name]
        print(f"{name:42s}: linearization {dt:7.0f} s | cgroup memory rise {dmem / 1e9:6.1f} GB | "
              f"shared-memory buffers {shm / 1e9:6.2f} GB", flush=True)
    names = list(out)
    y0, K0 = out[names[0]][0], out[names[0]][1]
    for n in names[1:]:
        check(f"[{n}] y identical to '{names[0]}'", np.array_equal(out[n][0], y0))
        check(f"[{n}] K identical (all columns, bitwise) to '{names[0]}'", np.array_equal(out[n][1], K0),
              f"max |dK| = {np.abs(out[n][1] - K0).max():.3e}, K shape {K0.shape}")
    t0 = out[names[0]][2]
    for n in names[1:]:
        print(f"speedup vs '{names[0]}': {n}: {t0 / out[n][2]:.1f}x", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["synthetic", "tile"])
    ap.add_argument("--fpas", default="0,2")
    ap.add_argument("--tile", type=int, default=12)
    ap.add_argument("--free", default="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo")
    ap.add_argument("--anchor-workers", type=int, default=8)
    ap.add_argument("--only-storage", action="store_true",
                    help="skip the slow dense-operator baseline; compare sparse-operator dense vs sparse storage only")
    a = ap.parse_args()
    synthetic() if a.stage == "synthetic" else tile(a)
    print("\nALL CHECKS PASSED" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
