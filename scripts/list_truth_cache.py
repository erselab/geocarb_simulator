"""List every truth-cache entry with its stored metadata (2026-09-24, user: "Does the truth cache
have metadata that can be queried for the details of how things were run?").

    PYTHONPATH=.:<gert> python scripts/list_truth_cache.py [--root DIR] [--verbose]

Entries with no `_meta` (pre-v8) are flagged NO-META -- `truth_cache.load` refuses to serve them.
`--verbose` also prints the full input set and provenance (git head, SLURM job, host, argv).
"""
import argparse
import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import truth_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    root = truth_cache.cache_root(a.root)
    files = sorted(root.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    print(f"{root}: {len(files)} entries (current TRUTH_CACHE_VERSION = {truth_cache.TRUTH_CACHE_VERSION})\n")
    for f in files:
        key = f.stem
        meta = truth_cache.describe(key, root)
        mt = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        if meta is None:
            print(f"{key}  {mt}  {f.stat().st_size / 1e6:6.1f} MB  NO-META (never served)")
            continue
        inp, prov = meta["inputs"], meta.get("provenance", {})
        stale = "" if meta.get("cache_version") == truth_cache.TRUTH_CACHE_VERSION else "  STALE-VERSION (never served)"
        print(f"{key}  {mt}  {f.stat().st_size / 1e6:6.1f} MB  v{meta.get('cache_version')}  fpa={inp.get('fpa')} "
              f"scene={inp.get('scene')} aerosol={inp.get('with_aerosol')} solver={inp.get('render_solver')} "
              f"job={prov.get('slurm_job_id')} git={(prov.get('git_head') or '?')[:8]}{stale}")
        if a.verbose:
            for k, v in inp.items():
                print(f"      input {k} = {v}")
            for k, v in prov.items():
                print(f"      prov  {k} = {v}")


if __name__ == "__main__":
    main()
