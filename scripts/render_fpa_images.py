"""Render one FPA's full 1024x1024 detector images for the realistic-prior scene and the truth scene
(2026-09-25), saved to results/fpa_images/fpa<n>_{truth,prior,noise}.npy (results/ is gitignored;
plot_fpa_images.py makes the figures). No aerosol, single_scatter, vary_albedo, n_lookup_samples=5600 --
exactly the sweeps' configuration, so the truth is served from the truth cache when it exists.
The noisy truth is A + N(0, sigma(A)) with sigma^2 = N0^2 + N1|A| (the band's geocarb_noise_model),
seed 1, drawn on the noiseless truth image (no re-render).

    python scripts/render_fpa_images.py --fpa 2 [--workers 16]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import gd_per_row_retrieve as gpr  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import GEOCARB_BANDS, along_slit_scene as als, sample_geometries, gert_root  # noqa: E402
from geocarb_gert.radiometry import geocarb_noise_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fpa", type=int, required=True)
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--noise-seed", type=int, default=1)
a = ap.parse_args()
fpa = a.fpa
out = REPO / "results/fpa_images"
out.mkdir(parents=True, exist_ok=True)

block = gg.geocarb_demo(verbose=False)["blocks"][0]
_, _, geo = sample_geometries(block, n=1, seed=0)[0]
root = gert_root()
absco = gert.ABSCOTable.load_all(str(root / "input/absco/absco.h5"))
solar = gert.SolarSpectrum.load(str(root / "input/solar/solar.h5"))
atm = als.atmosphere_at(0.0)
gpr._G.update(dict(atm=atm, absco=absco, geo=geo, solar=solar))
snr = gpr.DEFAULT_SNR_BY_FPA[fpa]
label = GEOCARB_BANDS[fpa][0]

print(f"FPA{fpa} truth (cache-aware)...", flush=True)
truth = gpr._band_setup_cached(fpa, atm, absco, geo, solar, snr, 5600, a.workers, False, False, 10, False, 0,
                               False, vary_albedo=True, with_aerosol=False)
A = np.asarray(truth["A"])
np.save(out / f"fpa{fpa}_truth.npy", A)

# noisy truth: shot noise on the noiseless image
nm = geocarb_noise_model(fpa)
sigma = np.sqrt(nm.N0 ** 2 + nm.N1 * np.abs(A))
np.save(out / f"fpa{fpa}_noise.npy", A + np.random.default_rng([a.noise_seed, fpa]).normal(0.0, sigma))

# realistic-prior scene: same atmosphere/surface names as the truth, prior fields in place of truth fields
f_truth, s_truth = als.build_scene_fields(uniform=False, barcode=False, realistic_barcode=False, vary_albedo=True,
                                          band_labels=[label], constant_albedo=0.0)
prior_atm = {n: als.PRIOR_FIELD_SETS["realistic"][n] for n in f_truth}
prior_alb = als.SURFACE_PRIOR_FIELD_SETS["realistic"]["albedo"]
prior_surf = {label: (lambda x, _l=label: prior_alb(x, _l))}
print(f"FPA{fpa} prior scene ({sorted(prior_atm)})...", flush=True)
prior = gpr._band_setup(fpa, atm, absco, geo, solar, snr, 5600, a.workers, False, False, 10, False, 0, False,
                        vary_albedo=True, fields=prior_atm, surface_fields=prior_surf, with_aerosol=False)
np.save(out / f"fpa{fpa}_prior.npy", np.asarray(prior["A"]))
print("saved", [p.name for p in sorted(out.glob(f"fpa{fpa}_*.npy"))], flush=True)
