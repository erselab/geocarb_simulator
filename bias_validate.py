"""Phase-1 validation: smile(+keystone) truth via gert dispersion, retrieve with
dispersion in the state, compare nonlinear vs linear bias for XCO2/XCH4/XCO."""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/scrowell/Library/CloudStorage/GoogleDrive-seanmrcrowell@gmail.com/My Drive/research/geocarb_simulator")
GERT = Path("/Users/scrowell/Library/CloudStorage/GoogleDrive-seanmrcrowell@gmail.com/My Drive/gert")

import geosat_geometry as gg
from geocarb_gert import (build_geocarb_instrument, sample_geometries,
                          reference_atmosphere, albedo_for)
import gert
from gert.forward_model import ForwardModel
from gert.rt_solver import SingleScatterSolver
from gert.retrieval import StateVector, GERTRetrieval
from gert.instrument import FlatSNR

# ---- scene / instrument / geometry ----
block = gg.geocarb_demo(verbose=False)['blocks'][0]
row, col, geo = sample_geometries(block, n=1, seed=0)[0]
inst  = build_geocarb_instrument()
atm   = reference_atmosphere()
absco = gert.ABSCOTable.load_all(str(GERT/'input/absco/absco.h5'))
solar = gert.SolarSpectrum.load(str(GERT/'input/solar/solar.h5'))
nb    = len(inst.windows)
albedo = albedo_for(inst, 'desert')          # per-band Lambertian albedo
GASES = ['co2', 'ch4', 'co', 'h2o']
KEY_FRAC = 20.0 / 1024.0                      # 20-px slit-image growth on 1024-px slit
SMILE_PX = 2.0                                # per-band smile (placeholder, TBD)

fm = ForwardModel(atm, absco, inst, geo, solver=SingleScatterSolver(), solar_spectrum=solar)

def smile_disp_coeffs(eta, keystone_on, fit_order=4):
    """Per-band dispersion coeffs [a0..] (cm-1) for the smile(+keystone) shift at eta."""
    disp = {}
    for b, w in enumerate(inst.windows):
        nu   = np.asarray(w.wn_instrument, float)
        span = w.wn_max - w.wn_min
        f    = (nu - w.wn_min) / span
        dnu_det = span / 1023.0               # detector pixel = band / 1024 px
        stretch2 = (1.0 + KEY_FRAC * f) ** 2 if keystone_on else 1.0
        dv   = -(SMILE_PX * dnu_det) * f * eta**2 / stretch2      # cm-1 per channel
        u    = w.dispersion_unit_coord()
        disp[b] = np.polyfit(u, dv, fit_order)[::-1]              # -> [a0,a1,...]
    return disp

# nominal (undistorted) truth + noise model
res0   = fm.run(albedo=list(albedo), albedo_slope=[0.0]*nb)
y_nom  = res0.y
sigma  = np.maximum(FlatSNR(300.).sigma(res0.R_band, inst.windows), 1e-30)
Sy_inv = np.diag(1.0 / sigma**2)
xtrue  = {g: atm.column_xgas(g) * (1e6 if g == 'co2' else 1e9) for g in ('co2','ch4','co')}
print("truth Xgas:", {g: round(v,2) for g,v in xtrue.items()}, "| n_y =", y_nom.size)

def bias_case(eta, keystone_on, order):
    disp   = smile_disp_coeffs(eta, keystone_on)
    y_dist = fm.run(albedo=list(albedo), albedo_slope=[0.0]*nb, dispersion=disp).y
    sv = StateVector.gas_scaling(prior_albedo=albedo, prior_albedo_slope=np.zeros(nb),
                                 gases=GASES, include_dispersion=True,
                                 dispersion_order=order, dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_dist, Sy_inv, sv, prior_albedo=albedo,
                        prior_albedo_slope=np.zeros(nb), analytical_jacobians=True,
                        max_iter=12, verbose=False)
    assert ret.analytical_jacobians, "expected analytical jacobians for SingleScatter"
    names = sv.names
    def xgas_bias(scale_vec):
        return {g: xtrue[g] * (scale_vec[names.index(g+'_scale')] - 1.0) for g in ('co2','ch4','co')}
    # nonlinear
    res = ret.run()
    nl = xgas_bias(res.x_ret); nl['_conv'] = res.converged; nl['_chi2'] = float(res.chisq_reduced)
    # linear: dx = G dy at prior
    x0 = sv.prior; y0 = ret._forward(x0); K, _ = ret._jacobian_mixed(x0, y0)
    mask = sv.active_mask; Sa = sv.Sa
    Ka = K[:, mask]; Sai = np.linalg.inv(Sa[np.ix_(mask, mask)])
    KtSi = Ka.T @ Sy_inv; Shat = np.linalg.inv(KtSi @ Ka + Sai)
    dx = np.zeros(len(sv.names)); dx[mask] = Shat @ (KtSi @ (y_dist - y0))
    lin = xgas_bias(x0 + dx)
    return nl, lin

t=time.time()
nl, lin = bias_case(0.9, True, 2)
print("\n[validation] eta=0.9, smile+keystone, order-2  (%.1fs)"%(time.time()-t))
print("  nonlinear bias:", {g: round(nl[g],4) for g in ('co2','ch4','co')}, "conv=%s chi2=%.3f"%(nl['_conv'],nl['_chi2']))
print("  linear    bias:", {g: round(lin[g],4) for g in ('co2','ch4','co')})
