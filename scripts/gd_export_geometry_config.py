"""Export a geometry config -- per-tile eta bounds, both bands' row ranges, and bin-center eta
positions -- from a completed multi-band sweep, so a LATER run (single-band or multi-band, any
free-parameter set) can be told to use the EXACT same tile boundaries and bins instead of
recomputing its own (2026-09-23, user: single-band and multi-band tile/bin_centers geometry
differs -- build_window_tiles vs build_window_tiles_multiband use different keystone/pixel-
density logic -- "It might also be a good idea to give a config file override that allows the
user to set the window and bin definitions in eta coordinates").

Multi-band is the natural REFERENCE geometry for the dump-then-reuse workflow this script builds
(its tiling is the more constrained case -- must respect BOTH bands' keystone maxima -- and each
tile's FPA2 row range already covers exactly that tile's eta interval for FPA2 alone, so reusing
it as a single-band FPA2 run's OWN row range gives identical eta coverage, not just identical
bin_centers within two different tile boundaries).

Hand-authoring is ALSO supported (2026-09-23, user: "the ability to specify values explicitly
instead of just pointing to a reference pkl file") -- this script's own output format IS the
config schema, nothing more: {"tiles": [{"tile": i, "eta_lo":..., "eta_hi":..., "bin_centers":
[eta, eta, ...]}, ...]}. `rows_by_fpa` is optional per entry -- give it explicitly (as this
script does, from a real tiling) for exact row ranges, or omit it and let the loader derive each
band's row range from eta_lo/eta_hi via `multiband_geometry.eta_to_row` (a nearest-row lookup,
checked against real tiling: within 1-2 rows at each edge -- fine for a hand-typed window, not a
substitute for an exact reference tiling's own padding/overlap logic). `bin_centers` always has
to be explicit floats either way -- there is no "derive the bins too" shortcut, by design: that
generation logic (pixel-density placement) is exactly what a geometry config exists to bypass.

    PYTHONPATH=.:<gert> python scripts/gd_export_geometry_config.py \
        --results-glob 'results/realistic_prior/multiband/mb_fpa0-2_r0-*_r2-*_free-co2-p-h2o-t-albedo-amplitude-height_cover_g1.0_etaslit_prior-realistic_aero.pkl' \
        --out geometry_config_aero.json

Load with --geometry-config in gd_multiband_window.py (bin_centers only; --rows already covers
row-range override there) and gd_joint_block_retrieve.py (both tile row-ranges AND bin_centers,
since single-band's own tiling has no equivalent override yet).
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", required=True,
                    help="glob of completed mb_fpa0-2_*.pkl files (any free-parameter set -- "
                         "only rows_by_fpa/bin_centers are read, not the retrieved state)")
    ap.add_argument("--fpas", default="0,2")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fpas = [int(t) for t in a.fpas.split(",")]

    tiles = build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, a.overlap)
    by_rows = {tuple(t.rows[fpas[0]]): i for i, t in enumerate(tiles)}

    config = {}
    for f in sorted(glob.glob(a.results_glob)):
        d = pickle.load(open(f, "rb"))
        i = by_rows.get(tuple(d["rows_by_fpa"][fpas[0]]))
        if i is None:
            print(f"skip (no tiling match): {f}")
            continue
        t = tiles[i]
        config[i] = dict(tile=i, eta_lo=float(t.eta_lo), eta_hi=float(t.eta_hi),
                         rows_by_fpa={str(fp): list(map(int, t.rows[fp])) for fp in fpas},
                         bin_centers=[float(v) for v in d["bin_centers"]])
    missing = [i for i in range(len(tiles)) if i not in config]
    if missing:
        print(f"WARNING: {len(missing)}/{len(tiles)} tiles have no matching result file: {missing}")
    out = [config[i] for i in sorted(config)]
    json.dump(dict(fpas=fpas, overlap=a.overlap, tiles=out), open(a.out, "w"), indent=1)
    print(f"wrote {len(out)}/{len(tiles)} tiles to {a.out}")


if __name__ == "__main__":
    main()
