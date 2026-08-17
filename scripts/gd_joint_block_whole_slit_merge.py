#!/usr/bin/env python3
"""Merge the per-task output files from a SLURM-array run of
gd_joint_block_whole_slit_sweep.py (--task-id/--n-tasks) into a single
combined pickle, in exactly the same format the script's own non-array
(single-process) mode produces -- so gd_joint_block_whole_slit_plot.py
and every other downstream consumer needs zero changes.

No shared state during the run itself (each array task only ever wrote
its own file); this script is the one, deliberately simple, place any
cross-task I/O happens, run once after the array job completes.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_whole_slit_merge.py \\
        results/gd_joint_block_whole_slit_fpa2_gratio1_parts
Output: results/gd_joint_block_whole_slit_fpa2_gratio1.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts_dir", type=str, help="directory of taskNNNofM.pkl files "
                    "(the *_parts/ directory the array job wrote into)")
    ap.add_argument("--out", type=str, default=None, help="output path (default: "
                    "parts_dir with the trailing _parts stripped, plus .pkl)")
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    part_files = sorted(parts_dir.glob("task*of*.pkl"))
    if not part_files:
        print(f"no task*.pkl files found in {parts_dir}")
        return 1

    results = {}
    meta = None
    n_tasks_seen = set()
    task_ids_seen = set()
    all_tiles = None
    for pf in part_files:
        with open(pf, "rb") as f:
            d = pickle.load(f)
        n_tasks_seen.add(d["n_tasks"])
        task_ids_seen.add(d["task_id"])
        if all_tiles is None:
            all_tiles = d["tiles"]
        elif d["tiles"] != all_tiles:
            print(f"FAIL: {pf} has a different `tiles` list than earlier parts -- "
                 f"these are not outputs from the same run.")
            return 1
        if meta is None:
            meta = {k: d[k] for k in ("fpa", "uniform", "gamma", "sigma_abs", "g_ratio", "min_window", "pad")}
        else:
            mismatches = {k: (meta[k], d[k]) for k in meta if meta[k] != d[k]}
            if mismatches:
                print(f"FAIL: {pf} has different run parameters than earlier parts: {mismatches}")
                return 1
        overlap = set(d["results"]) & set(results)
        if overlap:
            print(f"FAIL: {pf} re-solves windows already covered by another part: {sorted(overlap)}")
            return 1
        results.update(d["results"])

    if len(n_tasks_seen) != 1:
        print(f"FAIL: parts disagree on n_tasks: {n_tasks_seen}")
        return 1
    n_tasks = n_tasks_seen.pop()
    missing_tasks = set(range(n_tasks)) - task_ids_seen
    if missing_tasks:
        print(f"WARNING: {len(missing_tasks)}/{n_tasks} task IDs never wrote a part file "
             f"(likely still running, or failed): {sorted(missing_tasks)}")

    expected_keys = {(lo, hi) for lo, hi in all_tiles}
    missing_windows = expected_keys - set(results)
    n_ok = sum(1 for r in results.values() if "error" not in r)
    print(f"{len(part_files)} part files, {len(results)}/{len(expected_keys)} windows present "
         f"({n_ok} solved OK, {len(results)-n_ok} errored, {len(missing_windows)} missing)")
    if missing_windows:
        print(f"  missing: {sorted(missing_windows)[:10]}{' ...' if len(missing_windows) > 10 else ''}")

    suffix = "_uniform" if meta["uniform"] else ""
    suffix += f"_gratio{meta['g_ratio']:g}"
    default_out = parts_dir.parent / f"gd_joint_block_whole_slit_fpa{meta['fpa']}{suffix}.pkl"
    out_path = Path(args.out) if args.out else default_out
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "tiles": all_tiles, **meta}, f)
    print(f"saved {out_path}")
    return 0 if not missing_tasks and not missing_windows else 2


if __name__ == "__main__":
    raise SystemExit(main())
