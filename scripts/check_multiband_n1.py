"""Phase-3 gate (a): ONE band through the multi-band code (--anchor-mode nominal) must
reproduce the single-band driver's hi-res solve. Compares the multiband window result
against the driver's part file for the same window (default: the `_etaslit` c4 test,
FPA2 rows 25-37).  Exit 1 if the state vectors differ beyond tolerance.

    python scripts/check_multiband_n1.py <multiband.pkl> <driver task003of58.pkl>
"""
import pickle
import sys

import numpy as np

mb = pickle.load(open(sys.argv[1], "rb"))
dr = pickle.load(open(sys.argv[2], "rb"))
tile = tuple(dr["tiles"][dr["task_id"]])
h = dr["results"][tile]["hires"]
xd, xm = np.asarray(h["x"]), np.asarray(mb["x"])
print(f"driver window {tile}: n_free {xd.size}, rms_resid {h['resid_rms']:.6e}")
print(f"multiband       : n_free {xm.size}, rms_resid {mb['resid_rms']:.6e}")
ok = xd.size == xm.size
if ok:
    d = np.abs(xd - xm)
    rel = d / np.maximum(np.abs(xd), 1e-30)
    print(f"max |dx| = {d.max():.3e}, max relative = {rel.max():.3e}")
    ok = bool(np.allclose(xd, xm, rtol=1e-6, atol=1e-9)) and abs(h["resid_rms"] - mb["resid_rms"]) < 1e-3 * h["resid_rms"]
print("GATE (a) PASSED" if ok else "GATE (a) FAILED")
sys.exit(0 if ok else 1)
