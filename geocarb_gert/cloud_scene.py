"""A single liquid-water cloud with gradually thinning edges (2026-09-25).

A cloud is the same object as the aerosol layer in the forward model: a Gaussian layer in pressure (amplitude, centre
height, thickness) of type `cloud_water` (aerosol_mie.py: droplets r_eff ~9.5 um, per-band properties, extinction almost
flat across 0.76-2.3 um). What is new is the ALONG-SLIT shape of its optical depth: a plateau with smoothly tapering
edges, so the cloud fades into clear sky over ~100 km instead of ending at a step. Pixels at the edge are therefore only
weakly perturbed, which is the regime that is hard to detect.

    tau(x) = tau0                                       |x - x0| <= w
           = tau0 * 0.5 * (1 + cos(pi (|x - x0| - w) / e))      w < |x - x0| < w + e
           = 0                                          beyond

`amplitude_aerosol` (per Pa) = tau(x) / (sigma_p * sqrt(2 pi)), referenced to O2-A like every aerosol type; the per-band
scale from the Mie table is applied inside the forward model. Returns surface-field callables with the
`SURFACE_FIELDS` signature `fn(x_km, label=None)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: cloud-top region centre [hPa] for three altitudes; the surface is ~1000 hPa away from the mountain
ALTITUDES_HPA = {"low": 850.0, "mid": 600.0, "high": 350.0}
#: cloud optical depth at the plateau (O2-A reference): thin, moderate, thick
OPTICAL_DEPTHS = {"thin": 0.5, "moderate": 2.0, "thick": 8.0}


@dataclass(frozen=True)
class Cloud:
    tau0: float
    p_centre_hpa: float
    x0_km: float = 700.0          # away from the mountain (-250 km) and the water patch (430-545 km)
    plateau_km: float = 30.0      # half-width of the full-strength core
    edge_km: float = 150.0        # taper width: tau falls from tau0 to 0 over this distance
    sigma_hpa: float = 20.0       # Gaussian layer thickness (pressure sigma)

    @property
    def name(self) -> str:
        return f"tau{self.tau0:g}_p{self.p_centre_hpa:g}"

    def tau(self, x_km):
        d = np.abs(np.asarray(x_km, dtype=float) - self.x0_km)
        t = np.where(d <= self.plateau_km, 1.0, 0.0)
        edge = (d > self.plateau_km) & (d < self.plateau_km + self.edge_km)
        t = np.where(edge, 0.5 * (1.0 + np.cos(np.pi * (d - self.plateau_km) / self.edge_km)), t)
        return self.tau0 * t

    def extent_km(self, frac=1e-3):
        """(lo, hi) of the region where tau > frac * tau0."""
        return self.x0_km - self.plateau_km - self.edge_km, self.x0_km + self.plateau_km + self.edge_km

    def surface_fields(self):
        """Overrides for `als.SURFACE_FIELDS`: the aerosol rows now describe this cloud."""
        sigma_pa = self.sigma_hpa * 100.0
        return {
            "amplitude_aerosol": lambda x, label=None: np.maximum(self.tau(x), 1e-9) / (sigma_pa * np.sqrt(2.0 * np.pi)),
            "height_aerosol": lambda x, label=None: np.full_like(np.asarray(x, dtype=float), self.p_centre_hpa * 100.0),
            "thickness_aerosol": lambda x, label=None: np.full_like(np.asarray(x, dtype=float), sigma_pa),
        }


def scenarios():
    """The 3 altitudes x 3 optical depths grid: {name: Cloud}."""
    return {f"{a}_{t}": Cloud(tau0=tau, p_centre_hpa=p) for a, p in ALTITUDES_HPA.items() for t, tau in OPTICAL_DEPTHS.items()}
