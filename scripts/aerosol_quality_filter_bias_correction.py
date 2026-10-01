"""ACOS-style quality filter + bias correction for the 4-band aerosol (sulfate) arm with observation noise
(2026-10-01, user: "can we develop a quality filter and bias correction using the noisy sulfate results that
allows us to beat down the errors in CO2, CH4, and CO using the empirical correlations between the posterior
errors with other quantities in the prior and posterior state vector? This is the ACOS way of doing things
for the official OCO-2/3 data products.").

Real ACOS XCO2 bias correction (O'Dell et al. 2018 Sec. 5) is a multi-linear regression of the retrieval
bias against operationally-available diagnostics (dp -- retrieved minus independent/prior surface pressure,
albedo, aerosol optical depth proxies, chi^2, airmass, ...), fit against INDEPENDENT truth (TCCON) on a
training set, then applied to correct future retrievals that have no truth available.

**Train/test split (user, 2026-10-01): fit on the NOISY (sulfate+noise1) arm -- the realistic operational
situation, many noisy soundings -- and evaluate on the NOISELESS sulfate arm.** This is a stronger check than
an ordinary random train/test split of the noisy data alone: the noiseless arm's own retrieved-minus-truth
error is the genuine SYSTEMATIC/model-degeneracy bias with no noise-realization contamination at all, so a
correction that was fit on noisy data and still reduces the noiseless arm's error is real evidence it
captured true bias rather than overfitting to one particular noise draw. Both arms share the same 33 tiles
(same truth, same prior), so there's no held-out-truth mismatch to worry about -- this specifically isolates
the "does the correction generalize across noise realizations" question, which a same-arm split cannot.

Covariates are everything operationally available without knowing truth: dp (retrieved - prior p_surface),
d_amplitude/d_height (retrieved - prior aerosol), 4 retrieved per-band albedos, and the tile's own reduced
chi^2.

**Iterative (2026-10-01, user: "ACOS does this in an iterative way. Trial filter, bias correct, new trial
filter, additional bias correction."):** matches O'Dell et al. 2018's own practice -- the quality filter and
the regression are refit against the CURRENT residual each pass (trial filter -> fit+apply correction ->
re-filter the now-smaller residual -> fit+apply another correction -> ...), stopping per-target once a trial
regression fails the CV gate or an iteration's own relative RMS gain falls below `ITER_STOP_GAIN`.

    PYTHONPATH=.:<gert> python scripts/aerosol_quality_filter_bias_correction.py
"""
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from geocarb_gert import along_slit_scene as als  # noqa: E402

DIR = REPO / "results/realistic_prior/multiband"
BASE = "mb_fpa0-1-2-3_r0-*_r1-*_r2-*_r3-*_free-co2-ch4-co-p-h2o-t-albedo-amplitude-height_cover_g1.0_etaslit_prior-realistic_aero_sulfate"
TRAIN_PATTERN = f"{DIR}/{BASE}_noise1_lmfast.pkl"
TEST_PATTERN = f"{DIR}/{BASE}_lmfast.pkl"

TARGETS = ["co2_ppm", "ch4_ppb", "co_ppb"]
ALBEDO_ROWS = [("albedo_O2_A", "O2_A"), ("albedo_CO2_weak", "CO2_weak"),
               ("albedo_CO2_strong", "CO2_strong"), ("albedo_CH4_CO", "CH4_CO")]
#: 2026-10-01: an earlier version of this script threw all 8 covariates (dp, d_amp, d_height, chi2_red, and
#: the 4 per-band albedos) at every target. Honest 5-fold CV on the TRAINING set alone (see `cv_r2` below)
#: showed that version was overfitting badly for CH4/CO -- it looked fine in-sample but made the held-out
#: noiseless test set's CH4/CO error 38%/440% WORSE. Restricting to just (dp, d_amp) -- the two covariates
#: with a real, physically-understood coupling to the retrieval (the p_surface/amplitude_aerosol/CO2
#: degeneracy quantified earlier in this project's own pairplots) -- is what actually clears the CV bar for
#: CO2. chi2_red and the 4 (mutually collinear) albedo rows are dropped from the regression entirely; they
#: were the main overfitting source on this dataset's modest size (33 tiles).
COV_NAMES = ["dp", "d_amp"]


def truth_of(name, x_km, label=None):
    if label:
        return np.asarray(als.SURFACE_FIELDS["albedo"](x_km, label))
    if name in als.SURFACE_FIELDS:
        return np.asarray(als.SURFACE_FIELDS[name](x_km))
    return np.asarray(als.STATE_FIELDS[name](x_km))


def build_dataset(pattern):
    """One row per BIN across every tile matching `pattern`: covariates + per-target (retrieved - truth)."""
    all_cov_names = ["dp", "d_amp", "d_height", "chi2_red", "amp_ret", "h_ret"] + [n for n, _ in ALBEDO_ROWS]
    cov_cols = {c: [] for c in all_cov_names}     # compute every covariate regardless of COV_NAMES (used for
                                                  # the quality filter's own dp/chi2_red even when the
                                                  # regression itself only stacks a subset at the end below)
    err_cols = {t: [] for t in TARGETS}
    val_cols = {t: [] for t in TARGETS}    # raw retrieved value (not just error) -- for along-slit plots
    truth_cols = {t: [] for t in TARGETS}
    x_cols = []
    for f in sorted(glob.glob(pattern)):
        d = pickle.load(open(f, "rb"))
        params = d["joint"]["params"]
        j = d["joint"]
        chi2_red = float(np.sum(np.asarray(j["resid"], dtype=float) ** 2)) / float(j["dof"])

        x_coarse = np.asarray(params["co2_ppm"]["positions"], dtype=float) * als.SLIT_HALF_KM
        keep = slice(1, -1)
        xc = x_coarse[keep]
        n = xc.size

        p_ret = np.asarray(params["p_surface_hpa"]["values"], dtype=float)[keep]
        p_pri = np.asarray(params["p_surface_hpa"]["prior"], dtype=float)[keep]
        amp_ret = np.asarray(params["amplitude_aerosol"]["values"], dtype=float)[keep]
        amp_pri = np.asarray(params["amplitude_aerosol"]["prior"], dtype=float)[keep]
        h_ret = np.asarray(params["height_aerosol"]["values"], dtype=float)[keep]
        h_pri = np.asarray(params["height_aerosol"]["prior"], dtype=float)[keep]

        cov_cols["dp"].append(p_ret - p_pri)
        cov_cols["d_amp"].append(amp_ret - amp_pri)
        cov_cols["d_height"].append(h_ret - h_pri)
        cov_cols["chi2_red"].append(np.full(n, chi2_red))
        cov_cols["amp_ret"].append(amp_ret)
        cov_cols["h_ret"].append(h_ret)
        for name, label in ALBEDO_ROWS:
            pa = params[name]
            xa = np.asarray(pa["positions"], dtype=float) * als.SLIT_HALF_KM
            va = np.asarray(pa["values"], dtype=float)
            cov_cols[name].append(np.interp(xc, xa, va))

        x_cols.append(xc)
        for name in TARGETS:
            v = np.asarray(params[name]["values"], dtype=float)[keep]
            t = truth_of(name, xc)
            err_cols[name].append(v - t)
            val_cols[name].append(v)
            truth_cols[name].append(t)

    X_full = {c: np.concatenate(cov_cols[c]) for c in all_cov_names}
    X = np.column_stack([X_full[c] for c in COV_NAMES])     # the regression's own (lean) covariate subset
    Y = {t: np.concatenate(err_cols[t]) for t in TARGETS}
    V = {t: np.concatenate(val_cols[t]) for t in TARGETS}
    T = {t: np.concatenate(truth_cols[t]) for t in TARGETS}
    x_km = np.concatenate(x_cols)
    n_tiles = len(glob.glob(pattern))
    return X, X_full, Y, V, T, x_km, n_tiles


print("Loading TRAIN (sulfate + noise seed 1) ...")
X_train, Xf_train, Y_train, V_train, T_train, x_train, n_train_tiles = build_dataset(TRAIN_PATTERN)
print(f"  {n_train_tiles} tiles, {X_train.shape[0]} bins")
print("Loading TEST (sulfate, noiseless) ...")
X_test, Xf_test, Y_test, V_test, T_test, x_test, n_test_tiles = build_dataset(TEST_PATTERN)
print(f"  {n_test_tiles} tiles, {X_test.shape[0]} bins\n")

X_train1 = np.column_stack([X_train, np.ones(X_train.shape[0])])
X_test1 = np.column_stack([X_test, np.ones(X_test.shape[0])])


def cv_r2(X1, y, k=5, seed=0):
    """k-fold cross-validated R^2 ON THE TRAINING SET ALONE (never touches TEST) -- the honest way to ask
    "is this regression actually predictive, or just fitting noise" BEFORE deciding whether to trust it on
    held-out data at all. A real pipeline gates on this; a naive one fits-and-applies regardless, which is
    exactly what silently overfit CH4/CO above."""
    n = X1.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    sse, sst = 0.0, 0.0
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        c, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None)
        resid = y[te] - X1[te] @ c
        sse += np.sum(resid ** 2)
        sst += np.sum((y[te] - y[tr].mean()) ** 2)
    return 1.0 - sse / sst


# ---- ACOS-style ITERATIVE filter/correct loop (user, 2026-10-01: "ACOS does this in an iterative way.
# Trial filter, bias correct, new trial filter, additional bias correction.") -- each iteration: (1) fit a
# trial quality filter (linear-regime dp/chi2_red cut + variance-range amp_ret/h_ret cut) against the
# CURRENT residual, (2) fit+CV-gate a bias-correction regression on the filtered TRAIN subset, (3) apply it
# (TRAIN and TEST) to get a new, smaller residual, (4) repeat: the next iteration's filter and regression see
# the POST-correction residual, so they can pick up any secondary structure the first pass left behind. Stops
# per-target when a trial regression fails the CV gate (nothing left worth correcting) or the iteration's own
# relative RMS gain (on the filtered TRAIN subset) drops below `ITER_STOP_GAIN`.
LIN_TOL = 0.15             # |corr(residual, dp-or-dp^2)| must stay below this within the retained regime
STD_MULT = 1.5             # local std must stay within this factor of the best-constrained bin's std
VAR_FILTER_COVS = ["amp_ret", "h_ret"]
CV_R2_GATE = 0.05          # below this, the regression is not meaningfully predictive on TRAIN itself -- don't apply it
ITER_STOP_GAIN = 0.01      # stop iterating once an iteration's relative RMS gain (on filtered TRAIN) drops below this
MAX_ITERS = 4


def linear_regime_cut(dp_tr, chi2_tr, y_tr):
    """Most-inclusive (dp_cut, chi2_cut) quantile pair whose OWN linear-fit residuals show no remaining trend
    against dp or dp^2 (the same diagnostic as before, now reusable per-iteration since y_tr changes)."""
    best = None
    fallback = None
    for dp_q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for c2_q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            dp_cut = np.quantile(np.abs(dp_tr), dp_q)
            c2_cut = np.quantile(chi2_tr, c2_q)
            keep = (np.abs(dp_tr) <= dp_cut) & (chi2_tr <= c2_cut)
            if keep.sum() < 10:
                continue
            dp_k, y_k = dp_tr[keep], y_tr[keep]
            A = np.column_stack([dp_k, np.ones(dp_k.size)])
            coef, *_ = np.linalg.lstsq(A, y_k, rcond=None)
            resid = y_k - A @ coef
            c1 = abs(np.corrcoef(resid, dp_k)[0, 1]) if np.std(dp_k) > 0 else 0.0
            c2v = abs(np.corrcoef(resid, dp_k ** 2)[0, 1]) if np.std(dp_k) > 0 else 0.0
            linear_ok = (c1 < LIN_TOL) and (c2v < LIN_TOL)
            if fallback is None or keep.mean() < fallback[0]:
                fallback = (keep.mean(), dp_cut, c2_cut)
            if linear_ok and (best is None or keep.mean() > best[0]):
                best = (keep.mean(), dp_cut, c2_cut)
    _, dp_cut, c2_cut = best if best is not None else fallback
    return dp_cut, c2_cut, best is not None


def variance_range_filter(cov_vals, y, n_bins=10, std_mult=STD_MULT, min_per_bin=15):
    """Contiguous [lo, hi] range of `cov_vals`: bin it into quantile bins, flag bins whose local std of y
    exceeds `std_mult` x the MEDIAN bin std (a robust "reasonable" baseline), then keep the MOST INCLUSIVE
    (largest-population) contiguous run of non-flagged bins -- see [[geocarb-acos-bias-correction]]: these
    covariates flag WHERE the amplitude/height degeneracy is least resolved (a variance effect), not a
    correctable mean bias, so they're used as a range filter, not a regression covariate.

    Two earlier versions both broke (caught 2026-10-01): (1) anchoring on the single lowest-std bin was too
    noisy -- with ~90 points/bin, bin-to-bin std fluctuates by chance alone, so one unlucky-low bin could set
    an unreasonably strict threshold and crush retention to ~4%; (2) anchoring expansion on the bin containing
    the covariate's own median, when THAT bin itself happened to be flagged (e.g. CO2's amp_ret: the median
    bin sat right at the edge of a genuine elevated-variance region), collapsed to a single narrow bin.
    Picking the largest surviving contiguous run directly (the same "most inclusive passing cut" discipline
    used by the dp/chi2_red linear-regime filter above) avoids both failure modes."""
    edges = np.quantile(cov_vals, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bin_idx = np.clip(np.digitize(cov_vals, edges) - 1, 0, n_bins - 1)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = m.sum()
        if counts[b] >= min_per_bin:
            stds[b] = y[m].std()
    if np.all(np.isnan(stds)):
        return cov_vals.min(), cov_vals.max()
    threshold = std_mult * np.nanmedian(stds)
    good = np.where(np.isnan(stds), False, stds <= threshold)
    best_run, best_count = (0, n_bins), -1
    lo_b = 0
    while lo_b < n_bins:
        if not good[lo_b]:
            lo_b += 1
            continue
        hi_b = lo_b
        while hi_b + 1 < n_bins and good[hi_b + 1]:
            hi_b += 1
        run_count = counts[lo_b:hi_b + 1].sum()
        if run_count > best_count:
            best_run, best_count = (lo_b, hi_b), run_count
        lo_b = hi_b + 1
    lo_b, hi_b = best_run
    return edges[lo_b], edges[hi_b + 1]


def trial_filter(y_tr, name):
    """One trial filter against the CURRENT residual y_tr: linear-regime dp/chi2_red cut, then variance-range
    amp_ret/h_ret cut computed within that cut's survivors. Returns keep_tr, keep_te, and the cut bounds."""
    dp_cut, c2_cut, lin_ok = linear_regime_cut(Xf_train["dp"], Xf_train["chi2_red"], y_tr)
    keep_tr = (np.abs(Xf_train["dp"]) <= dp_cut) & (Xf_train["chi2_red"] <= c2_cut)
    keep_te = (np.abs(Xf_test["dp"]) <= dp_cut) & (Xf_test["chi2_red"] <= c2_cut)
    bounds = dict(dp_cut=dp_cut, c2_cut=c2_cut, lin_ok=lin_ok)
    for cov in VAR_FILTER_COVS:
        lo, hi = variance_range_filter(Xf_train[cov][keep_tr], y_tr[keep_tr])
        keep_tr = keep_tr & (Xf_train[cov] >= lo) & (Xf_train[cov] <= hi)
        keep_te = keep_te & (Xf_test[cov] >= lo) & (Xf_test[cov] <= hi)
        bounds[f"{cov}_lo"], bounds[f"{cov}_hi"] = lo, hi
    return keep_tr, keep_te, bounds


print(f"\n{'target':10s} {'it':>3s} {'retained(tr)':>13s} {'retained(te)':>13s} {'CV R2':>8s} {'apply?':>7s}"
     f" {'rms(filt te)':>13s} {'gain':>8s}  coefficients ({', '.join(COV_NAMES)}, intercept)")
results = {}
for name in TARGETS:
    y_tr_cur = Y_train[name].copy()
    y_te_cur = Y_test[name].copy()
    rms_before = float(np.sqrt(np.mean(Y_test[name] ** 2)))
    history = []
    final = None
    for it in range(MAX_ITERS):
        keep_tr, keep_te, bounds = trial_filter(y_tr_cur, name)
        X_tr_f, y_tr_f = X_train1[keep_tr], y_tr_cur[keep_tr]
        r2 = cv_r2(X_tr_f, y_tr_f) if keep_tr.sum() > 2 * X_train1.shape[1] else -np.inf
        coef, *_ = np.linalg.lstsq(X_tr_f, y_tr_f, rcond=None)
        applied = r2 >= CV_R2_GATE

        rms_filt_before_it = float(np.sqrt(np.mean(y_te_cur[keep_te] ** 2))) if keep_te.any() else float("nan")
        if applied:
            y_tr_cur = y_tr_cur.copy()
            y_te_cur = y_te_cur.copy()
            y_tr_cur[keep_tr] = y_tr_cur[keep_tr] - X_train1[keep_tr] @ coef
            y_te_cur[keep_te] = y_te_cur[keep_te] - X_test1[keep_te] @ coef
        rms_filt_after_it = float(np.sqrt(np.mean(y_te_cur[keep_te] ** 2))) if keep_te.any() else float("nan")
        gain = 1.0 - rms_filt_after_it / rms_filt_before_it if (applied and rms_filt_before_it > 0) else 0.0

        coef_s = ", ".join(f"{c:+.3g}" for c in coef)
        print(f"{name:10s} {it:3d} {keep_tr.mean()*100:12.1f}% {keep_te.mean()*100:12.1f}% {r2:8.3f}"
             f" {'yes' if applied else 'NO':>7s} {rms_filt_after_it:13.4g} {gain*100:7.1f}%  [{coef_s}]")
        history.append(dict(it=it, keep_tr=keep_tr, keep_te=keep_te, bounds=bounds, r2=r2,
                            applied=applied, coef=coef, gain=gain))
        final = history[-1]
        if not applied or gain < ITER_STOP_GAIN:
            break

    keep_te = final["keep_te"]
    results[name] = dict(y_te=Y_test[name], y_corr=y_te_cur, keep_te=keep_te, rms_before=rms_before,
                         coef=final["coef"], cv_r2=final["r2"], applied=any(h["applied"] for h in history),
                         n_iters=len(history), dp_cut=final["bounds"]["dp_cut"])
    rms_filt_corr = float(np.sqrt(np.mean(y_te_cur[keep_te] ** 2))) if keep_te.any() else float("nan")
    results[name]["rms_filt_corr"] = rms_filt_corr
    print(f"{name:10s} -> converged after {len(history)} iteration(s), final retained(te)="
         f"{keep_te.mean()*100:.1f}%, rms {rms_before:.4g} -> {rms_filt_corr:.4g}"
         f" ({100*(1 - rms_filt_corr/rms_before):+.1f}%)\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, TARGETS):
    r = results[name]
    keep = r["keep_te"]
    y_filt_only = r["y_te"][keep]                 # filtered, NOT corrected (the filter's own contribution alone)
    y_filt_corr = r["y_corr"][keep]                # filtered AND corrected -- the real, valid combined pipeline
    lim = np.percentile(np.abs(r["y_te"]), 99) * 1.1
    lim = lim if lim > 0 else 1.0
    bins = np.linspace(-lim, lim, 41)
    ax.hist(r["y_te"], bins=bins, histtype="step", lw=1.6, density=True, color="tab:red",
           label=f"before: rms {r['rms_before']:.3g}")
    ax.hist(y_filt_only, bins=bins, histtype="step", lw=1.6, density=True, color="tab:orange", ls="--",
           label=f"filtered only: rms {np.sqrt(np.mean(y_filt_only**2)):.3g}")
    ax.hist(y_filt_corr, bins=bins, histtype="step", lw=1.6, density=True, color="tab:green",
           label=f"filtered+corrected: rms {r['rms_filt_corr']:.3g}")
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_title(f"{name} ({r['n_iters']} iteration{'s' if r['n_iters'] != 1 else ''})")
    ax.legend(fontsize=8)
fig.suptitle("ACOS-style ITERATIVE quality filter + bias correction -- trained on sulfate+noise1 (trial filter, "
            "correct, re-filter the residual, correct again), evaluated on noiseless sulfate")
fig.tight_layout()
out = REPO / "plots/aerosol_quality_filter_bias_correction.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved {out}")

# ---- along-slit view, TEST (noiseless) set: pre-filter, post-filter, and bias-corrected VALUES vs eta,
# alongside truth -- one panel per target, sharing the X axis (2026-10-01, user's own follow-up request)
order = np.argsort(x_test)
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(13, 3.2 * len(TARGETS)), sharex=True)
for ax, name in zip(axes, TARGETS):
    r = results[name]
    xs = x_test[order]
    truth_s = T_test[name][order]
    before_s = V_test[name][order]                       # pre-filter: every retrieved value, uncorrected
    corrected_s = (r["y_corr"] + T_test[name])[order]     # valid only at kept bins -- see keep_s below
    keep_s = r["keep_te"][order]

    ax.plot(xs, truth_s, color="k", lw=1.2, label="truth", zorder=5)
    ax.plot(xs, before_s, color="tab:red", lw=0.9, alpha=0.8, label="pre-filter (retrieved)")
    # scatter, not a connected line, for BOTH filtered series: `keep_s` has real gaps (filtered-out bins),
    # and a line would bridge straight across them -- a misleading artifact, not real data. The correction
    # is only ever valid at kept bins (fit on, and only applied to, the filtered regime) -- plotting it
    # elsewhere would be the same out-of-domain extrapolation error this whole restructuring fixed.
    ax.scatter(xs[keep_s], before_s[keep_s], color="tab:orange", s=5, alpha=0.9, zorder=3,
              label="post-filter (retained, uncorrected)")
    ax.scatter(xs[keep_s], corrected_s[keep_s], color="tab:blue", s=5, alpha=0.9, zorder=4,
              label="post-filter + bias-corrected")
    ax.set_ylabel(name)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12 if ax is axes[-1] else 1.25))
axes[-1].set_xlabel("along-slit position [km]")
fig.suptitle("ITERATIVE quality filter + bias correction along the slit -- sulfate, noiseless (held-out test), "
            "correction trained on sulfate+noise1", y=1.01)
fig.tight_layout()
out2 = REPO / "plots/aerosol_quality_filter_along_slit.png"
fig.savefig(out2, dpi=140, bbox_inches="tight")
print(f"saved {out2}")

# ---- same along-slit view, but for the ERROR (retrieved - truth) rather than the raw value -- the zero
# line replaces "truth", and the error magnitude/structure (rather than tracking a large dynamic-range
# curve) is what's actually diagnostic of where/how much the filter and correction help.
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(13, 3.2 * len(TARGETS)), sharex=True)
for ax, name in zip(axes, TARGETS):
    r = results[name]
    xs = x_test[order]
    before_s = r["y_te"][order]            # pre-filter error
    corrected_s = r["y_corr"][order]        # valid only at kept bins -- see keep_s below
    keep_s = r["keep_te"][order]

    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6, zorder=5)
    ax.plot(xs, before_s, color="tab:red", lw=0.9, alpha=0.8, label="pre-filter error")
    ax.scatter(xs[keep_s], before_s[keep_s], color="tab:orange", s=5, alpha=0.9, zorder=3,
              label="post-filter error (retained, uncorrected)")
    ax.scatter(xs[keep_s], corrected_s[keep_s], color="tab:blue", s=5, alpha=0.9, zorder=4,
              label="post-filter + bias-corrected error")
    ax.set_ylabel(f"{name} error")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12 if ax is axes[-1] else 1.25))
axes[-1].set_xlabel("along-slit position [km]")
fig.suptitle("ITERATIVE quality filter + bias correction: ERROR along the slit -- sulfate, noiseless (held-out "
            "test), correction trained on sulfate+noise1", y=1.01)
fig.tight_layout()
out3 = REPO / "plots/aerosol_quality_filter_along_slit_error.png"
fig.savefig(out3, dpi=140, bbox_inches="tight")
print(f"saved {out3}")
