#!/usr/bin/env python3
"""Cross-configuration comparison: residuals and retrieved-state-vs-truth,
across the analytic-Jacobian regression matrix's own window-count/G axes.

Built to answer a specific decision, not to be a general-purpose plotter:
picking window count, G (bins per window, via `g_ratio`), and coarse vs.
hires for future full-state-vector, multi-FPA retrievals. Reads the 8
`analytic_jacobian_testing/results/<config>_analytic.pkl` files -- analytic,
not fd, since the two are validated identical to 1e-6..1e-7 relative
(`analytic_jacobian_testing/SOLVER_REGRESSION.md`) and analytic is the
solver going forward; nothing here re-derives that agreement, it is assumed.

**What this matrix does NOT cover, and these plots cannot answer**:
`anchor_density` was fixed at 1 throughout (never swept to 2, 4, ...) and
`state_interp` was fixed at `True` throughout (never run `False`). The
coarse-vs-hires split shown here is a PARTIAL, indirect proxy for "does a
finer-than-G render grid help at all" -- it is not a sweep over how much
finer. Picking an actual `anchor_density` or deciding `state_interp` needs
its own dedicated runs; see `analytic_jacobian_testing/README.md` §5.

Two figures, one per free-row set (co2-only vs. co2+p_surface), each with:
  1. per-row residual RMS, coarse (dashed) and hires (solid), all 4
     window-count x g_ratio combos overlaid
  2. CO2 truth + retrieved (hires), all 4 combos
  3. CO2 deviation from truth (hires), all 4 combos
  4/5. same two panels for p_surface (co2+p figure only)
plus a printed summary table (median/max residual and deviation per config)
so the comparison doesn't require eyeballing noisy per-row curves.

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_config_compare.py
Output: plots/joint_block/gd_joint_block_config_compare_{co2,co2p}.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "analytic_jacobian_testing" / "results"

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.gd_render import s_max  # noqa: E402

N_COLS = 1024
CONFIGS = [("58", "1"), ("58", "3"), ("29", "1"), ("29", "3")]
COLOR = {("58", "1"): "tab:blue", ("58", "3"): "tab:orange",
        ("29", "1"): "tab:green", ("29", "3"): "tab:red"}
LABEL = {("58", "1"): "58 windows, g=1 (G=width)",
        ("58", "3"): "58 windows, g=3 (G=width/3)",
        ("29", "1"): "29 windows, g=1 (G=width)",
        ("29", "3"): "29 windows, g=3 (G=width/3)"}
SOLVE_LS = {"coarse": "--", "hires": "-"}
PLOT_STYLE = {
    "font.family": "serif", "font.size": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#20242b", "text.color": "#20242b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.linewidth": 0.8,
}


def stitch(results: dict, solve: str, eta_rows: np.ndarray):
    """{row_name: per-detector-row values} and the per-row residual RMS."""
    out, rr = {}, np.full(N_COLS, np.nan)
    for w in results.values():
        snap = w.get(solve)
        if snap is None:
            continue
        lo, hi = int(w["row_lo"]), int(w["row_hi"])
        sl = slice(lo, hi + 1)
        for name, rec in snap["params"].items():
            arr = out.setdefault(name, np.full(N_COLS, np.nan))
            pos, val = np.asarray(rec["positions"]), np.asarray(rec["values"])
            arr[sl] = (np.interp(eta_rows[sl], pos, val) if pos.size > 1 else val[0])
        blk = np.asarray(w[f"resid_{solve}"], dtype=float).reshape(hi - lo + 1, N_COLS)
        rr[sl] = np.sqrt(np.mean(blk ** 2, axis=1))
    return out, rr


def load_all(free_key: str, rows: np.ndarray, eta_rows: np.ndarray):
    """{(nwin, gratio): {solve: (state_dict, resid_rms_array)}}"""
    out = {}
    for nwin, gratio in CONFIGS:
        with open(ARCHIVE / f"{free_key}-{nwin}-g{gratio}_analytic.pkl", "rb") as f:
            d = pickle.load(f)
        res = d["results"]
        out[(nwin, gratio)] = {sv: stitch(res, sv, eta_rows) for sv in ("coarse", "hires")}
    return out


def summarize(free_key: str, data: dict, truth: dict, names: list):
    """Prints the decision table: median/max residual and |deviation| per config."""
    hdr = (f"  {'config':22s} {'solve':7s} {'residRMS med':>13s} {'residRMS max':>13s}"
          + "".join(f" {n.split('_')[0]+' rms':>11s} {n.split('_')[0]+' max':>11s}" for n in names))
    print(f"=== {free_key} summary ===")
    print(hdr)
    print("-" * len(hdr))
    for nwin, gratio in CONFIGS:
        for sv in ("coarse", "hires"):
            st, rr = data[(nwin, gratio)][sv]
            ok = np.isfinite(rr)
            line = (f"  {LABEL[(nwin,gratio)]:22s} {sv:7s} {np.nanmedian(rr):13.3e} "
                   f"{np.nanmax(rr[ok]):13.3e}")
            for n in names:
                dv = st.get(n)
                if dv is None:
                    line += f" {'--':>11s} {'--':>11s}"
                    continue
                d = dv - truth[n]
                line += f" {np.sqrt(np.nanmean(d**2)):11.4f} {np.nanmax(np.abs(d)):11.4f}"
            print(line)
    print()


def plot_one(free_key: str, title: str, data: dict, truth: dict, eta_rows: np.ndarray,
            rows: np.ndarray, names: list, out_path: Path):
    n_state_panels = 2 * len(names)
    n_panels = 1 + n_state_panels
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.1 + 2.3 * n_state_panels),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1.3, 1.0] * len(names)})

    # -- panel 0: residual RMS, coarse (dashed) + hires (solid), all 4 configs --
    ax = axes[0]
    for nwin, gratio in CONFIGS:
        c = COLOR[(nwin, gratio)]
        for sv in ("coarse", "hires"):
            _, rr = data[(nwin, gratio)][sv]
            ax.semilogy(rows, rr, color=c, ls=SOLVE_LS[sv], lw=1.0 if sv == "hires" else 0.8,
                       alpha=0.95 if sv == "hires" else 0.6,
                       label=f"{LABEL[(nwin,gratio)]} ({sv})")
    ax.set_ylabel("per-row residual RMS\n[radiance]")
    ax.set_title(f"{title} -- residual RMS (solid=hires, dashed=coarse)", fontsize=12)
    ax.legend(fontsize=7.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    ax.grid(alpha=0.25, lw=0.5)

    # -- state + deviation panels, one pair per free row, hires only --
    for i, name in enumerate(names):
        ax_v, ax_d = axes[1 + 2 * i], axes[2 + 2 * i]
        ax_v.plot(rows, truth[name], color="black", lw=1.4, label="truth", zorder=5)
        for nwin, gratio in CONFIGS:
            c = COLOR[(nwin, gratio)]
            st, _ = data[(nwin, gratio)]["hires"]
            v = st.get(name)
            if v is None:
                continue
            ax_v.plot(rows, v, color=c, lw=0.9, alpha=0.85, label=LABEL[(nwin, gratio)])
            ax_d.plot(rows, v - truth[name], color=c, lw=0.85, alpha=0.85)
        ax_v.set_ylabel(name)
        ax_d.axhline(0, color="black", lw=0.6)
        ax_d.set_ylabel("retr - true")
        ax_d.grid(alpha=0.25, lw=0.5)
        ax_v.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax_v.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35))
        ax_v.set_title(f"{name} -- retrieved (hires) vs. truth", fontsize=10.5)

    axes[-1].set_xlabel("detector row")
    fig.suptitle(f"FPA2 joint block, analytic solver -- {title}", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> int:
    rows = np.arange(float(N_COLS))
    _, s = xy_to_wavelength_slit(2, np.full(N_COLS, 512.0), rows)
    eta_rows = s / s_max(2)
    x_km = eta_rows * als.SLIT_HALF_KM
    truth = {n: np.asarray(f(x_km), dtype=float) for n, f in als.STATE_FIELDS.items()}

    out_dir = REPO_ROOT / "plots" / "joint_block"

    co2_data = load_all("co2", rows, eta_rows)
    summarize("co2-only", co2_data, truth, ["co2_ppm"])
    plot_one("co2", "CO2-only", co2_data, truth, eta_rows, rows, ["co2_ppm"],
            out_dir / "gd_joint_block_config_compare_co2.png")

    co2p_data = load_all("co2p", rows, eta_rows)
    summarize("co2+p_surface", co2p_data, truth, ["co2_ppm", "p_surface_hpa"])
    plot_one("co2p", "CO2 + p_surface free", co2p_data, truth, eta_rows, rows,
            ["co2_ppm", "p_surface_hpa"],
            out_dir / "gd_joint_block_config_compare_co2p.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
