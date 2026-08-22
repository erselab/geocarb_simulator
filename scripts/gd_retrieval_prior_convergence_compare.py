#!/usr/bin/env python3
"""Old (pre-Phase-D) vs. corrected (post-Phase-D) imperfect-prior retrieval
accuracy, as a function of resolution (g_ratio) -- "does convergence with
resolution improve" for the two Phase 2 "first try" experiments
(CO2-only and CO2+p_surface against a uniformly biased prior).

Reads the four REPORT.md tables directly (no re-solving, no pkl access):
    results/config_matrix/retrieval_prior_co2plus1pct_v1/REPORT.md  (old, stale)
    results/config_matrix/retrieval_prior_co2plus1pct_v2/REPORT.md  (corrected)
    results/config_matrix/retrieval_prior_co2p_v1/REPORT.md         (old, stale)
    results/config_matrix/retrieval_prior_co2p_v2/REPORT.md         (corrected)
-- see docs/PROJECT_STATUS.md Sec.6 for why _v1/_v2 disagree (the Phase D
Sy_inv noise-model fix).

Each config row also carries an anchor_density (1/4/16); this plot's own
"simple" scope collapses that axis to its mean per g_ratio (anchor_density
matters much less than the old-vs-new gap it would otherwise be competing
with visually -- see the REPORT.md tables themselves for the per-
anchor_density numbers).

Run:  PYTHONPATH=. python3 scripts/gd_retrieval_prior_convergence_compare.py
Output: plots/config_matrix/retrieval_prior_convergence_compare.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_matrix import parse_config  # noqa: E402

TAGS = {
    "co2plus1pct": ("CO2-only, prior=CO2+1%", "co2_ppm", "CO2 rms", "CO2 max"),
    "co2p": ("CO2+p_surface, prior=CO2+1%/p_surface-1%", "co2_ppm", "CO2 rms", "CO2 max"),
}


def _parse_report(path: Path) -> dict:
    """{(g_ratio, anchor_density): {"co2_rms":..., "co2_max":..., "dp_rms":..., "dp_max":...}}"""
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| co2"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] == "config":
            continue
        cid = cells[0]
        try:
            _, _, gratio, adens, _, _, _, _ = parse_config(cid)
            co2_rms, co2_max, dp_rms, dp_max = (float(cells[4]), float(cells[5]),
                                                float(cells[6]), float(cells[7]))
        except (ValueError, IndexError):
            continue
        rows[(gratio, adens)] = dict(co2_rms=co2_rms, co2_max=co2_max, dp_rms=dp_rms, dp_max=dp_max)
    return rows


def _mean_by_gratio(rows: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    by_g = {}
    for (g, _), vals in rows.items():
        by_g.setdefault(g, []).append(vals[field])
    gs = sorted(by_g)
    return np.array(gs), np.array([np.mean(by_g[g]) for g in gs])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag_dir = REPO_ROOT / "results" / "config_matrix"
    reports = {}
    for tag in TAGS:
        for ver in ("v1", "v2"):
            path = tag_dir / f"retrieval_prior_{tag}_{ver}" / "REPORT.md"
            if not path.exists():
                print(f"missing {path}, skipping {tag}/{ver}")
                continue
            reports[(tag, ver)] = _parse_report(path)

    panels = [("co2plus1pct", "co2_rms", "CO2 rms [ppm]"),
             ("co2p", "co2_rms", "CO2 rms [ppm]"),
             ("co2p", "dp_rms", "p_surface rms [hPa]")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (tag, field, ylabel) in zip(axes, panels):
        label, _, _, _ = TAGS[tag]
        for ver, color, ls, name in (("v1", "tab:red", "--", "old (pre-Phase-D)"),
                                     ("v2", "tab:blue", "-", "corrected (post-Phase-D)")):
            rows = reports.get((tag, ver))
            if not rows:
                continue
            g, y = _mean_by_gratio(rows, field)
            ax.plot(g, y, color=color, ls=ls, marker="o", lw=1.8, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks([0.5, 1, 3, 6, 12])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("g_ratio (log scale -- finer resolution to the right)")
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Imperfect-prior retrieval accuracy vs. resolution: old (flat-scalar Sy_inv) "
                 "vs. corrected (real noise model) -- mean across anchor_density", fontsize=12)
    fig.tight_layout()
    out = REPO_ROOT / "plots" / "config_matrix" / "retrieval_prior_convergence_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
