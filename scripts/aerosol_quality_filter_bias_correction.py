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
    all_cov_names = ["dp", "d_amp", "d_height", "chi2_red"] + [n for n, _ in ALBEDO_ROWS]
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


# ---- quality filter FIRST: thresholds on |dp| and chi2_red fit on TRAIN. Computed before the regression
# (reordered 2026-10-01, user: "did you train the bias correction only on the filtered data?" -- it hadn't
# been; ACOS itself fits its own bias correction only on quality-filtered soundings, precisely so outlier/
# bad-quality cases don't skew the least-squares fit via high leverage).
#
# **Filter design criterion (user, 2026-10-01): "the filtering should be done to restrict the relationships
# between the error and covariates to a linear regime"** -- NOT simply whichever threshold minimizes train
# RMS (the earlier version). A linear bias correction is only a valid model where the true error-vs-covariate
# relationship actually IS close to linear; applying it outside that regime is model misspecification, not
# just "a bit more uncertain" -- exactly what made the out-of-domain full-set correction above 17.2% WORSE
# for CO2. So the threshold search below picks the MOST INCLUSIVE (dp_cut, chi2_cut) pair for which a linear
# fit's own residuals show no remaining trend against dp or dp^2 within the retained range (|corr(resid, dp)|
# and |corr(resid, dp^2)| both below `LIN_TOL`) -- a direct linearity diagnostic, not an RMS-outcome proxy for
# one. Falls back to the narrowest (most conservative) grid point if no cut passes the diagnostic.
LIN_TOL = 0.15             # |corr(residual, dp-or-dp^2)| must stay below this within the retained regime
print(f"{'target':10s} {'dp cut':>10s} {'train retained':>14s} {'test retained':>14s}  (linear-regime filter)")
dp_tr, chi2_tr = Xf_train["dp"], Xf_train["chi2_red"]
dp_te, chi2_te = Xf_test["dp"], Xf_test["chi2_red"]
filter_info = {}
for name in TARGETS:
    y_tr = Y_train[name]
    best = None          # most inclusive (highest retained fraction) cut that still passes the linearity check
    fallback = None       # narrowest cut tried, in case nothing passes
    for dp_q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for c2_q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            dp_cut = np.quantile(np.abs(dp_tr), dp_q)
            c2_cut = np.quantile(chi2_tr, c2_q)
            keep_tr = (np.abs(dp_tr) <= dp_cut) & (chi2_tr <= c2_cut)
            if keep_tr.sum() < 10:
                continue
            dp_k, y_k = dp_tr[keep_tr], y_tr[keep_tr]
            A = np.column_stack([dp_k, np.ones(dp_k.size)])
            coef, *_ = np.linalg.lstsq(A, y_k, rcond=None)
            resid = y_k - A @ coef
            c1 = abs(np.corrcoef(resid, dp_k)[0, 1]) if np.std(dp_k) > 0 else 0.0
            c2v = abs(np.corrcoef(resid, dp_k ** 2)[0, 1]) if np.std(dp_k) > 0 else 0.0
            linear_ok = (c1 < LIN_TOL) and (c2v < LIN_TOL)
            if fallback is None or keep_tr.mean() < fallback[0]:
                fallback = (keep_tr.mean(), dp_cut, c2_cut)
            if linear_ok and (best is None or keep_tr.mean() > best[0]):
                best = (keep_tr.mean(), dp_cut, c2_cut)
    _, dp_cut, c2_cut = best if best is not None else fallback
    keep_tr = (np.abs(dp_tr) <= dp_cut) & (chi2_tr <= c2_cut)
    keep_te = (np.abs(dp_te) <= dp_cut) & (chi2_te <= c2_cut)
    filter_info[name] = dict(dp_cut=dp_cut, c2_cut=c2_cut, keep_tr=keep_tr, keep_te=keep_te)
    print(f"{name:10s} {dp_cut:10.3g} {keep_tr.mean()*100:13.1f}% {keep_te.mean()*100:13.1f}%"
         f"  {'(linear OK)' if best is not None else '(fallback -- no cut passed)'}")

# ---- bias correction: fit ONLY on the filtered (quality-passing) TRAIN bins. Real pipeline order (both
# here and in ACOS itself): FILTER FIRST, then correct only the survivors -- the correction is fit on, and
# therefore only valid within, the "good" (small-dp) covariate regime the filter defines. Applying it to a
# bin the filter would have rejected is extrapolation outside where it was ever fit, and (checked directly,
# 2026-10-01, user: "did you train the bias correction only on the filtered data?") makes CO2 actively WORSE
# (-17.2% applied to the full unfiltered test set) even though the same regression, applied only to the
# filtered test bins it's actually valid for, still helps.
print(f"\n{'target':10s} {'train CV R2':>12s} {'apply?':>7s} coefficients ({', '.join(COV_NAMES)}, intercept)")
results = {}
CV_R2_GATE = 0.05        # below this, the regression is not meaningfully predictive on TRAIN itself -- don't apply it
for name in TARGETS:
    keep_tr = filter_info[name]["keep_tr"]
    y_tr_all, y_te = Y_train[name], Y_test[name]
    X_tr_f, y_tr_f = X_train1[keep_tr], y_tr_all[keep_tr]     # filtered training subset only
    r2 = cv_r2(X_tr_f, y_tr_f)
    coef, *_ = np.linalg.lstsq(X_tr_f, y_tr_f, rcond=None)
    apply_corr = r2 >= CV_R2_GATE
    results[name] = dict(y_te=y_te, coef=coef, cv_r2=r2, applied=apply_corr)
    coef_s = ", ".join(f"{c:+.3g}" for c in coef)
    print(f"{name:10s} {r2:12.3f} {'yes' if apply_corr else 'NO':>7s}  [{coef_s}]")

print(f"\n{'target':10s} {'dp cut':>10s} {'retained':>10s} {'rms before':>12s} {'rms filtered':>13s} {'%improved':>10s}")
for name in TARGETS:
    y_te = Y_test[name]
    keep_te = filter_info[name]["keep_te"]
    rms_before = float(np.sqrt(np.mean(y_te ** 2)))
    rms_after = float(np.sqrt(np.mean(y_te[keep_te] ** 2))) if keep_te.any() else float("nan")
    pct = 100.0 * (1.0 - rms_after / rms_before)
    print(f"{name:10s} {filter_info[name]['dp_cut']:10.3g} {keep_te.mean()*100:9.1f}% {rms_before:12.4g} {rms_after:13.4g} {pct:9.1f}%")
    results[name]["keep_te"] = keep_te
    results[name]["rms_before"] = rms_before

# ---- real combined pipeline: filter TEST, then correct ONLY the survivors. `y_corr` is a full-length array
# with the correction applied where valid (retained bins) and left AS-IS (uncorrected) where not -- the
# filter already drops those from any RMS/plot that only looks at retained bins; keeping them as raw rather
# than (invalidly) corrected values if a caller ever looks at the unfiltered array by mistake.
print(f"\n{'target':10s} {'rms before':>12s} {'rms filt+corr':>14s} {'%improved':>10s} {'retained':>10s}  (out-of-domain full-set correction, NOT the real pipeline, for comparison: %improved)")
for name in TARGETS:
    r = results[name]
    y_te, keep = r["y_te"], r["keep_te"]
    y_corr = y_te.copy()
    if r["applied"]:
        y_corr[keep] = y_te[keep] - X_test1[keep] @ r["coef"]
    r["y_corr"] = y_corr
    rms_after = float(np.sqrt(np.mean(y_corr[keep] ** 2))) if keep.any() else float("nan")
    pct = 100.0 * (1.0 - rms_after / r["rms_before"])
    # out-of-domain comparison number only (never applied/plotted): the full-set correction, for the record
    pred_full = X_test1 @ r["coef"] if r["applied"] else np.zeros_like(y_te)
    rms_full_corr = float(np.sqrt(np.mean((y_te - pred_full) ** 2)))
    pct_full = 100.0 * (1.0 - rms_full_corr / r["rms_before"])
    print(f"{name:10s} {r['rms_before']:12.4g} {rms_after:14.4g} {pct:9.1f}% {keep.mean()*100:9.1f}%  ({pct_full:+.1f}%)")
    r["rms_filt_corr"] = rms_after

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
    ax.set_title(name)
    ax.legend(fontsize=8)
fig.suptitle("ACOS-style quality filter + bias correction -- trained on sulfate+noise1 (filter THEN correct, "
            "both fit on filtered train only), evaluated on noiseless sulfate")
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
fig.suptitle("Quality filter + bias correction along the slit -- sulfate, noiseless (held-out test), "
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
fig.suptitle("Quality filter + bias correction: ERROR along the slit -- sulfate, noiseless (held-out test), "
            "correction trained on sulfate+noise1", y=1.01)
fig.tight_layout()
out3 = REPO / "plots/aerosol_quality_filter_along_slit_error.png"
fig.savefig(out3, dpi=140, bbox_inches="tight")
print(f"saved {out3}")
