"""Multi-band joint-block state: ONE :class:`~geocarb_gert.joint_state.StateSpec`
shared by several bands (FPAs), with independent per-band albedo rows.

See ``docs/MULTIBAND_PLAN.md`` (Sec.4.1/4.2). Design in one paragraph: the
single-band machinery (``render_at_anchors`` / ``build_forward_state`` /
``jacobians.linearize``) is left untouched. Each band b sees a VIEW of the joint
state -- the shared rows (gases, p, T, h2o, and later aerosol) plus its OWN
albedo row presented under the ordinary single-band name ``"albedo"`` -- and the
joint forward model / Jacobian are the per-band ones stacked, with each band's
Jacobian columns embedded into the joint packed vector. The Gauss-Newton solver,
priors, sigmas, dx/sigma convergence and trial clamping all operate on the joint
``StateSpec`` and are unchanged.

Decisions this encodes (user, 2026-09-20): albedo rows are INDEPENDENT per band
(own positions/sigma/correlation length, each band's single-band values);
atmosphere (and, when present, aerosol) rows are shared; the shared coordinate is
the slit-image ``eta`` (Sec.33 of ``PROJECT_STATUS.md``).

Views must be built AFTER the free/frozen flags are final: a view shares the
joint spec's shared-row ``ParamSpec`` objects but takes a COPY of the band's
albedo row, so freezing the joint albedo row later does not reach an existing
view. Build views (and forward models from them) once the flags are set.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .joint_state import StateSpec, state_spec_from_scene

#: The name every single-band forward model / Jacobian expects for the surface
#: albedo row (``build_forward_state`` etc. read ``surface["albedo"]``).
SINGLE_BAND_ALBEDO_ROW = "albedo"


@dataclass(frozen=True)
class BandRef:
    """One band of a multi-band retrieval."""
    fpa: int
    label: str                  # `SpectralWindow.label`, e.g. "O2_A" -- selects the albedo prior
    albedo_row: str = ""        # this band's row name in the JOINT state; default albedo_<label>

    @property
    def row(self) -> str:
        return self.albedo_row or f"{SINGLE_BAND_ALBEDO_ROW}_{self.label}"


class MultiBandState:
    """A joint :class:`StateSpec` plus the band bookkeeping needed to view it
    per band and to stack per-band forward models / Jacobians."""

    def __init__(self, joint: StateSpec, bands: Sequence[BandRef]):
        self.joint = joint
        self.bands = list(bands)
        names = [p.name for p in joint.params]
        rows = [b.row for b in self.bands]
        if len(set(rows)) != len(rows):
            raise ValueError(f"duplicate per-band albedo row names: {rows}")
        missing = [r for r in rows if r not in names]
        if missing:
            raise ValueError(f"joint state has no albedo row(s) {missing}; rows are {names}")
        self._other_rows = {b.row: {o.row for o in self.bands if o is not b} for b in self.bands}

    # -- views ------------------------------------------------------------
    def view(self, b: int) -> StateSpec:
        """Band `b`'s single-band-shaped :class:`StateSpec`: every joint row
        except the OTHER bands' albedo rows, with band b's albedo row renamed
        ``"albedo"``. Shared rows are the SAME objects as the joint spec's."""
        band = self.bands[b]
        skip = self._other_rows[band.row]
        params = []
        for p in self.joint.params:
            if p.name in skip:
                continue
            if p.name == band.row:
                q = copy.copy(p)
                q.name = SINGLE_BAND_ALBEDO_ROW
                params.append(q)
            else:
                params.append(p)
        return StateSpec(params)

    def _joint_name(self, b: int, view_name: str) -> str:
        return self.bands[b].row if view_name == SINGLE_BAND_ALBEDO_ROW else view_name

    def index_map(self, b: int) -> np.ndarray:
        """Indices into the JOINT packed vector of band b's packed (free-only)
        view vector: ``x_view = x_joint[index_map(b)]``, in the view's own order."""
        vsl = self.view(b).slices()
        jsl = self.joint.slices()
        idx = []
        for vname, vs in vsl.items():          # view order (free rows only)
            jname = self._joint_name(b, vname)
            if jname not in jsl:
                raise ValueError(f"row {vname!r} is free in band {b}'s view but frozen in the "
                                 f"joint state ({jname!r}); build views after setting free flags")
            js = jsl[jname]
            if (js.stop - js.start) != (vs.stop - vs.start):
                raise ValueError(f"row {vname!r}: view/joint sizes differ")
            idx.append(np.arange(js.start, js.stop))
        return np.concatenate(idx) if idx else np.zeros(0, dtype=int)

    def x_view(self, x, b: int) -> np.ndarray:
        return np.asarray(x, dtype=float)[self.index_map(b)]

    def embed_columns(self, b: int, J_b: np.ndarray) -> np.ndarray:
        """Band b's Jacobian (n_data_b x n_view_free) -> (n_data_b x n_joint_free),
        zeros in the columns of the other bands' albedo rows."""
        idx = self.index_map(b)
        J_b = np.asarray(J_b, dtype=float)
        if J_b.shape[1] != idx.size:
            raise ValueError(f"band {b}: Jacobian has {J_b.shape[1]} columns, view has {idx.size} free elements")
        out = np.zeros((J_b.shape[0], self.joint.n_free))
        out[:, idx] = J_b
        return out

    # -- stacking ---------------------------------------------------------
    def stack_forward(self, forwards: Sequence[Callable]) -> Callable:
        """``forward_joint(x) = concat_b forwards[b](x_view_b(x))``."""
        idxs = [self.index_map(b) for b in range(len(self.bands))]

        def forward_joint(x):
            x = np.asarray(x, dtype=float)
            return np.concatenate([np.asarray(f(x[i]), dtype=float).ravel()
                                   for f, i in zip(forwards, idxs)])
        return forward_joint

    def stack_jacobian(self, jacobian_fns: Sequence[Callable]) -> Callable:
        """``J_joint(x) = vstack_b embed(J_b(x_view_b(x)))``."""
        idxs = [self.index_map(b) for b in range(len(self.bands))]

        def jacobian_joint(x):
            x = np.asarray(x, dtype=float)
            return np.vstack([self.embed_columns(b, jf(x[i]))
                              for b, (jf, i) in enumerate(zip(jacobian_fns, idxs))])
        return jacobian_joint

    def stack_linearize(self, lin_fns: Sequence[Callable]) -> Callable:
        """The ``jacobian_fn`` form ``gauss_newton_state`` consumes: each band's
        ``lin_b(x_b) -> (y_b, K_b, K_g_b)`` (what ``jacobians.linearize`` returns)
        becomes ``(concat y, embedded-and-stacked K, {})``. Sub-bin anomaly
        sensitivities (``K_g``) are not supported across bands yet: a non-empty
        ``K_g`` from any band raises rather than being silently dropped."""
        idxs = [self.index_map(b) for b in range(len(self.bands))]

        def linearize_joint(x):
            x = np.asarray(x, dtype=float)
            ys, Ks = [], []
            for b, (lf, i) in enumerate(zip(lin_fns, idxs)):
                y_b, K_b, Kg_b = lf(x[i])
                if Kg_b:
                    raise NotImplementedError("sub-bin anomaly (K_g) is not supported in multi-band solves yet")
                ys.append(np.asarray(y_b, dtype=float).ravel())
                Ks.append(self.embed_columns(b, K_b))
            return np.concatenate(ys), np.vstack(Ks), {}
        return linearize_joint

    @staticmethod
    def stack_Sy_inv_diag(Sy_inv_diags: Sequence[np.ndarray]) -> np.ndarray:
        """Block-diagonal inverse noise covariance (diagonal): no cross-band noise
        correlation is assumed (Sec.4.5 of the plan)."""
        return np.concatenate([np.asarray(s, dtype=float).ravel() for s in Sy_inv_diags])


def joint_spec_from_scene(bin_centers, bands: Sequence[BandRef], band_kwargs: dict | None = None,
                          **kw) -> MultiBandState:
    """Build the joint :class:`MultiBandState` from the truth scene.

    `kw` is forwarded to :func:`state_spec_from_scene` for EVERY band (``free``,
    ``sigmas``, ``kinds``, ``fields``, ``prior_form``, ...); `band_kwargs` maps a
    band ``label`` to per-band overrides (e.g. that band's own ``surface_positions``).
    Atmosphere rows come from the first band's spec (they are identical across
    bands by construction -- checked); each band contributes its own albedo row
    under its :attr:`BandRef.row` name, built with that band's own label so it
    carries that band's own prior, sigma and correlation length. Any OTHER
    surface-target rows (aerosol) are shared and taken from the first band.
    """
    if not bands:
        raise ValueError("need at least one band")
    band_kwargs = band_kwargs or {}
    specs = [state_spec_from_scene(bin_centers, band_label=b.label, **{**kw, **band_kwargs.get(b.label, {})})
             for b in bands]
    base = specs[0]
    for sp, b in zip(specs[1:], bands[1:]):      # shared rows must agree across bands
        for p in base.params:
            if p.name == SINGLE_BAND_ALBEDO_ROW or p.target == "surface":
                continue
            q = sp[p.name]
            if not (np.array_equal(p.positions, q.positions) and np.array_equal(p.prior, q.prior)
                    and p.free == q.free):
                raise ValueError(f"shared row {p.name!r} differs between band {bands[0].label!r} "
                                 f"and {b.label!r}; only the albedo row may be per-band")
    out = []
    for p in base.params:
        if p.name == SINGLE_BAND_ALBEDO_ROW:
            for b, sp in zip(bands, specs):
                q = copy.copy(sp[SINGLE_BAND_ALBEDO_ROW])
                q.name = b.row
                out.append(q)
        else:
            out.append(p)
    return MultiBandState(StateSpec(out), bands)
