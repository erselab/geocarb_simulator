"""Focal-plane image simulator for a parameterized slit spectrometer.

This sits **above** the per-column radiative transfer: given hi-res radiance
spectra *along the slit*, it maps them onto a discrete 2-D detector array
(spatial × spectral) with parameterized optical distortions — **keystone**,
**smile** (spectral line curvature) and **clocking** (detector rotation).  It
reuses `gert`'s ILS for the spectral convolution and owns only the focal-plane
geometry, so it stays in the mission-side adapter layer (not `gert` core).

Conventions
-----------
* Detector array ``A[i, j]`` — spatial row ``i`` (along-slit), spectral column
  ``j`` (dispersion).  ``A.shape == (n_spatial, n_spectral)``.
* Normalized slit coordinate ``η ∈ [-1, +1]`` in **object** space.
* In-band spectral fraction ``f_j = (ν_j − ν_min)/(ν_max − ν_min) ∈ [0, 1]``;
  ``f = 0`` is the short-wave (blue) end, ``f = 1`` the long-wave (red) end.

Distortions — the three standard spectral-spatial focal-plane errors, orthogonal
in the slit coordinate ``η``: keystone is a spatial *scale* vs wavelength, smile a
*quadratic* (η²) spectral curvature, clocking a *linear* (η¹) rotation/shear.

* **Keystone** — spatial magnification varies with wavelength,
  ``M(ν) = M₀·(1 + keystone_frac·f)``.  A detector row therefore samples object
  coordinate ``η_obj = η / (1 + keystone_frac·f)`` — a magnification **symmetric
  about the slit centre**, so the slit image lengthens by an equal pixel amount at
  each end.  The total lengthening at the red end is ``keystone_frac·n_spatial``
  px; specify it directly in pixels via ``keystone_px`` if preferred.  It only
  moves radiance where the scene varies *along the slit*.
* **Smile** — a monochromatic line images as a parabola whose arms bow toward the
  **long-wave (red)** end at the slit edges; the column shift is
  ``+smile_px_edge·f·η²`` [px], growing from 0 at the blue end to
  ``smile_px_edge`` spectral pixels at the red **edge** of the slit (η = ±1).
* **Clocking** — a rigid rotation of the detector about the boresight by
  ``clocking_deg``.  Being a rotation it appears on **both** axes at once and is
  *linear* in the offset: a spectral column tilt ``−θ·η·(n_spatial/2)`` [px]
  (constant across the band, unlike smile) and a spatial shear
  ``+θ·(j − n_spectral/2)`` [px] along dispersion, with ``θ = radians(clocking_deg)``.
  Square detector pixels are assumed, so the rotation is isotropic in pixel space.

The combined per-pixel sample is ``ν_eff(i, j) = ν_j − dλ_px·[column shift]`` on the
spectral axis (handled exactly, per row, via ILS-center shifts) and a per-column
spatial resample (keystone scale + clocking shear); the two axes are applied as a
separable first-order factorization, exact to ``O(distortion²)``.

Instrument blur is separated from scene structure: the **spectral** blur is the ILS
and the **spatial (N/S)** blur is a Gaussian PSF (``spatial_psf_fwhm_px``, GeoCarb
≈ 1.5 px) applied along the slit after rendering.  Scenes are therefore *sharp
truth* — a 25 % albedo step between neighbouring surfaces images with the true
~1.5-px PSF ramp, not an arbitrary blend width.

Terminology note (2026-08-13): "scene," in the physical sense, means the true
*atmospheric state* along the slit — ``state(η) -> AtmosphericProfile`` (see
:func:`geocarb_gert.along_slit_scene.atmosphere_at`). Radiance is a *derived*
quantity, ``radiance(η) = RT(state(η))``, computed by running that state
through ``gert.forward_model.ForwardModel`` (e.g. the ``spectrum_for`` pattern
in ``scripts/gd_joint_block_retrieve.py`` and ``scripts/gd_test.py``'s own
``_band_setup``). The ``η ↔ x_km`` mapping is a fixed bijection
(``x_km = η · SLIT_HALF_KM``), so state can equivalently be indexed by either.

The "scene" *functions* in **this** module (:func:`edge_scene`,
:func:`uniform_scene`, :func:`barcode_scene`, :func:`nearest_bin_scene`,
:func:`random_scene`) do not operate at the state level — they take
already-RT-computed spectra as input and return ``radiance(eta)`` callables, a
computational shortcut for building simple synthetic test truths without
re-running RT per query. They are radiance-space scene *assemblers*, not
state-space scene *generators*. Where one selects rather than blends among
precomputed spectra per query η (:func:`nearest_bin_scene`'s hard nearest-bin
assignment), that stays within the state-space-only rule the joint bin
retrieval depends on (JOINT_ROW_INVERSION_PLAN.md §2,
docs/JOINT_BIN_RETRIEVAL_ATBD.html §4) — each returned spectrum is still
exactly one atmosphere's own RT output. Where one blends
(:func:`edge_scene`/:func:`barcode_scene` with ``softness > 0``), that is a
deliberate synthetic-scene construction choice for testing, not a claim that
blending real spectra is generally valid; the retrieval forward model itself
never does this; it always re-runs ``ForwardModel`` on a bin's own live state
(see ``gd_joint_block_retrieve.make_spectrum_fn``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from gert.instrument import ILS


# ── scene helpers ───────────────────────────────────────────────────────────
# A "scene" is a shared hi-res grid ``wn_hires`` plus a callable
# ``radiance(η) -> S_hires`` giving the spectrum seen at object slit position η.
# ``η`` may be a scalar or an ndarray of any shape; the return shape is always
# ``η.shape + S_hires.shape`` (scalar η -> plain ``(n_hires,)``, as before).
# Array-η support is what lets :mod:`geocarb_gert.gd_render` evaluate a whole
# row's worth of per-pixel true slit positions in one vectorized call instead
# of one Python call per pixel.

def uniform_scene(S_hires: np.ndarray) -> Callable[[float], np.ndarray]:
    """A slit-homogeneous scene: the same spectrum at every slit position."""
    S = np.asarray(S_hires, dtype=float)
    return lambda eta: np.broadcast_to(S, np.shape(eta) + S.shape).copy()

def _segment_blend(boundaries: np.ndarray, seg_matrix: np.ndarray,
                   softness: float = 0.0) -> Callable[[float], np.ndarray]:
    """Build a ``radiance(eta)`` that blends across ``len(seg_matrix)`` segments.

    Shared, efficient engine behind :func:`random_scene` and (with explicit,
    non-random boundaries/segments) any other piecewise along-slit scene:
    evaluates the telescoping-``tanh``-sum construction as one small
    per-segment weight array (shape ``eta.shape + (n_seg,)``) followed by a
    single matrix multiply against ``seg_matrix``, instead of one full
    ``eta.shape + (n_hires,)`` array add per boundary.

    Parameters
    ----------
    boundaries : ndarray, shape (n_seg - 1,)
        Ascending η boundary positions.
    seg_matrix : ndarray, shape (n_seg, n_hires)
        Stacked hi-res spectrum for each segment, in slit order.
    softness : float
        Transition half-width in η. Default ``0`` gives sharp truth edges.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    n_seg = seg_matrix.shape[0]

    def radiance(eta: float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        if boundaries.size == 0:
            return np.broadcast_to(seg_matrix[0], eta.shape + seg_matrix.shape[1:]).copy()
        w = 0.5 * (1.0 + np.tanh((eta[..., None] - boundaries) / max(softness, 1e-9)))
        ones = np.ones(eta.shape + (1,))
        zeros = np.zeros(eta.shape + (1,))
        W = np.concatenate([ones, w, zeros], axis=-1)      # (*eta.shape, n_seg+1)
        beta = W[..., :-1] - W[..., 1:]                     # (*eta.shape, n_seg)
        return beta @ seg_matrix                             # (*eta.shape, n_hires)

    return radiance


def random_scene(scenes, n_segments: int = 4, softness: float = 0.0,
                 seed: Optional[int] = None) -> Callable[[float], np.ndarray]:
    """A patchwork slit scene — ``edge_scene`` generalized to several transitions.

    The slit is split into ``n_segments`` regions at random η boundaries, each
    assigned one of the base ``scenes`` (adjacent segments are forced to differ so
    every boundary is a real edge), with smooth ``tanh`` transitions of half-width
    ``softness``.  One image then shows *multiple* along-slit transitions, exercising
    the keystone/clocking spatial signatures at several edges at once.

    Mathematically a telescoping sum of ``tanh`` steps, exactly like
    :func:`edge_scene` (which is the ``n_segments == 2`` case)::

        S(η) = S₀ + Σ_k ½[1 + tanh((η − b_k)/softness)] · (S_{k+1} − S_k)

    but *evaluated* as one small blend-weight computation (shape
    ``eta.shape + (n_segments,)``) followed by a single matrix multiply
    against the stacked segment spectra, rather than one full
    ``eta.shape + (n_hires,)`` array add per boundary — the latter makes
    cost scale with the number of segments, which is expensive at
    :mod:`geocarb_gert.gd_render` scale (~1e6 pixel evaluations per image).

    Parameters
    ----------
    scenes : sequence of ndarray
        The few base hi-res spectra to draw from, each shape ``(n_hires,)`` on the
        shared ``wn_hires`` grid (e.g. desert/water/forest spectra).
    n_segments : int
        Number of piecewise regions along the slit (``n_segments − 1`` transitions).
    softness : float
        Transition half-width in η.
    seed : int, optional
        RNG seed for reproducible boundaries and segment assignment.

    Returns
    -------
    callable
        ``radiance(η) -> S_hires``.  Carries ``.boundaries`` (the η edge positions)
        and ``.segment_scenes`` (the chosen base-scene index per segment) for
        overlaying the transitions on a plot.
    """
    scenes = [np.asarray(s, dtype=float) for s in scenes]
    if not scenes:
        raise ValueError("need at least one base scene")
    m = max(int(n_segments), 1)
    rng = np.random.default_rng(seed)

    boundaries = np.sort(rng.uniform(-1.0, 1.0, size=m - 1))
    idx = [int(rng.integers(len(scenes)))]
    for _ in range(m - 1):
        choices = [c for c in range(len(scenes)) if c != idx[-1]] or [idx[-1]]
        idx.append(int(rng.choice(choices)))
    seg_matrix = np.stack([scenes[i] for i in idx])   # (n_segments, n_hires)

    radiance = _segment_blend(boundaries, seg_matrix, softness)

    radiance.boundaries = boundaries
    radiance.segment_scenes = idx
    return radiance


def barcode_scene(S: np.ndarray, brightness, widths=None,
                  softness: float = 0.0) -> Callable[[float], np.ndarray]:
    """A diffuser-style barcode illumination: same spectral shape everywhere,
    only the brightness (a scalar gain) varies along the slit.

    Physically: sunlight reflected off a diffuser with alternating
    light/dark reflectivity bars, e.g. for a ground-test illumination
    pattern — the atmospheric column (and hence the spectral *shape*) is
    identical at every slit position, since it's the same column of sky;
    only the reflected brightness changes. This is the high-spatial-frequency
    companion to the single-gradient along-slit albedo test in
    ``KEYSTONE_SMILE_BIAS_PLAN.md`` Sec. 9a — chosen to probe keystone/PSF
    row-mixing and pixel-grid aliasing where the scene varies fastest
    (Sec. 9b), with bar widths and brightness levels independently
    specifiable (a plain alternating high/low pattern is the ``n``-bar,
    2-level special case).

    Same telescoping-``tanh``-sum construction as :func:`random_scene`,
    applied to the scalar brightness rather than the full spectrum.

    Parameters
    ----------
    S : ndarray, shape (n_hires,)
        The single hi-res spectrum, identical at every slit position.
    brightness : sequence of float
        Scalar gain for each bar, e.g. ``[1.0, 0.2] * 5`` for 10 alternating
        bright/dark bars.
    widths : sequence of float, optional
        Bar widths in η units, same length as ``brightness``, must sum to
        ``2.0`` (spans ``η ∈ [-1, 1]``). Default: equal-width bars.
    softness : float
        Transition half-width in η, same convention as :func:`edge_scene`.
        Default ``0`` gives sharp truth bars.
    """
    S = np.asarray(S, dtype=float)
    brightness = np.asarray(brightness, dtype=float)
    n_bars = len(brightness)
    if n_bars < 1:
        raise ValueError("brightness must have at least one entry")
    if widths is None:
        widths = np.full(n_bars, 2.0 / n_bars)
    else:
        widths = np.asarray(widths, dtype=float)
        if len(widths) != n_bars:
            raise ValueError("widths and brightness must be the same length")
        if not np.isclose(widths.sum(), 2.0):
            raise ValueError(f"widths must sum to 2.0 (η spans [-1, 1]), got {widths.sum()}")
    boundaries = -1.0 + np.cumsum(widths)[:-1]

    def radiance(eta: float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        g = np.broadcast_to(brightness[0], eta.shape).copy()
        for k, b in enumerate(boundaries):
            w = 0.5 * (1.0 + np.tanh((eta - b) / max(softness, 1e-9)))
            g = g + w * (brightness[k + 1] - brightness[k])
        return g[..., None] * S

    radiance.boundaries = boundaries
    radiance.brightness = brightness
    return radiance


def edge_scene(S_left: np.ndarray, S_right: np.ndarray,
               edge: float = 0.0, softness: float = 0.0) -> Callable[[float], np.ndarray]:
    """A bright/dark boundary along the slit — the scene that reveals keystone.

    Blends ``S_left`` (η < edge) into ``S_right`` (η > edge).  ``softness`` is the
    transition half-width in η; the default ``0`` gives a **sharp truth edge** —
    the physical scene is a step, and the realistic ~1.5-px spatial ramp comes from
    the instrument's ``spatial_psf_fwhm_px``, not from this blend.
    """
    S_left  = np.asarray(S_left,  dtype=float)
    S_right = np.asarray(S_right, dtype=float)

    def radiance(eta: float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        w = 0.5 * (1.0 + np.tanh((eta - edge) / max(softness, 1e-9)))
        return (1.0 - w[..., None]) * S_left + w[..., None] * S_right

    return radiance


def nearest_bin_scene(bin_centers: np.ndarray,
                      spectra: list) -> Callable[[float], np.ndarray]:
    """``G``-way generalization of :func:`edge_scene` for the joint
    multi-atmosphere block (`JOINT_ROW_INVERSION_PLAN.md` §2/§4).

    Each pixel's ``η`` is assigned to its single nearest bin center and
    returns that bin's own hi-res spectrum unchanged -- **hard assignment,
    no interpolation of spectra**. This is the "simpler and safer default"
    §2 calls for: every pixel's prediction comes from exactly one bin's own
    RT output, never a blend of two bins' already-computed spectra (that
    blend is exactly the operation §9l/§9m found to be the dominant bias
    source in `rectify()`). If a smoother forward map is ever wanted, the
    correct place to interpolate is the *state* (CO2 ppm) between bins,
    followed by a fresh RT run -- not this function.

    Parameters
    ----------
    bin_centers : ndarray, shape (G,)
        Sorted ``η`` bin centers.
    spectra : sequence of ndarray, length ``G``
        Each entry the hi-res spectrum (shape ``(n_hires,)``) for that bin.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    spectra_arr = np.asarray(spectra, dtype=float)   # (G, n_hires)
    # midpoints between consecutive sorted centers -> searchsorted gives
    # the nearest-center index directly (ties go to the right bin).
    edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])

    def radiance(eta: float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        idx = np.searchsorted(edges, eta)
        return spectra_arr[idx]

    radiance.bin_centers = bin_centers
    return radiance


def gaussian_blur_rows(A: np.ndarray, fwhm_px: float) -> np.ndarray:
    """Blur ``A`` along axis 0 (rows) by a normalized Gaussian of FWHM ``fwhm_px``.

    Edge-extended so the array ends are not darkened. Returns ``A`` unchanged
    when ``fwhm_px <= 0``. Shared by :meth:`FocalPlaneModel.apply_spatial_psf`
    and :mod:`geocarb_gert.gd_render` — the same along-slit N/S PSF blur,
    independent of which geometric-distortion model rendered the rows.
    """
    fwhm = float(fwhm_px)
    if fwhm <= 0.0:
        return A
    sigma = fwhm / 2.3548
    half  = max(1, int(np.ceil(4.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    n = A.shape[0]
    Ap = np.pad(A, ((half, half), (0, 0)), mode="edge")
    out = np.zeros_like(A)
    for t, w in enumerate(k):                          # shift-and-add convolution
        out += w * Ap[t:t + n]
    return out


# ── focal-plane model ───────────────────────────────────────────────────────
@dataclass
class FocalPlaneModel:
    """Parameterized slit-spectrometer focal plane, in physical optical units.

    Parameters
    ----------
    slit_length_mm : float
        Physical slit length in object space [mm].
    magnification : float
        Optical magnification (image / object).
    detector_pitch_um : float
        Detector pixel pitch [µm].  ``n_spatial`` follows from
        ``slit_length_mm · magnification / pitch``.
    wn_min, wn_max : float
        Band limits [cm⁻¹].
    n_spectral : int
        Number of spectral (dispersion) pixels — the wavelength sampling.
    fwhm_nm : float
        Spectral resolution as an ILS FWHM [nm] at band centre.
    keystone_frac : float
        Fractional spatial-magnification increase from the blue to the red end
        of the band (e.g. ``1e-3`` for +0.1 %).  Symmetric about the slit centre.
    keystone_px : float, optional
        Alternative spec: **total** slit-image lengthening at the red end, in
        pixels (each end extends by ``keystone_px/2``).  When given, it overrides
        ``keystone_frac`` via ``keystone_frac = keystone_px / n_spatial``.
    smile_px_edge : float
        Smile amplitude in **spectral pixels** at the slit edge (η = ±1) and the
        red end of the band; grows linearly to 0 at the blue end.
    clocking_deg : float
        Detector rotation about the boresight [degrees].  A rigid rotation, so it
        tilts spectral lines linearly across the slit *and* shears the slit image
        along dispersion (see the module docstring).
    spatial_psf_fwhm_px : float
        Along-slit (N/S) point-spread-function FWHM [detector pixels].  Applied as
        a normalized Gaussian blur along the spatial axis — the spatial analog of
        the spectral ILS — so a sharp scene edge images with the instrument's true
        spatial response (GeoCarb ≈ 1.5 px).  Set ``0`` to disable.
    """
    slit_length_mm:    float
    magnification:     float
    detector_pitch_um: float
    wn_min:            float
    wn_max:            float
    n_spectral:        int
    fwhm_nm:           float
    keystone_frac:     float = 1e-3
    keystone_px:       Optional[float] = None
    smile_px_edge:     float = 0.0
    clocking_deg:      float = 0.0
    spatial_psf_fwhm_px: float = 1.5

    # derived (built in __post_init__)
    n_spatial: int        = field(init=False)
    eta:       np.ndarray = field(init=False, repr=False)   # (n_spatial,)  ∈ [-1, 1]
    nu:        np.ndarray = field(init=False, repr=False)   # (n_spectral,) nominal centres [cm⁻¹]
    dnu:       float      = field(init=False)               # dispersion [cm⁻¹/pixel]
    _theta:    float      = field(init=False, repr=False)   # clocking angle [rad]
    ils:       ILS        = field(init=False, repr=False)

    def __post_init__(self):
        if self.wn_max <= self.wn_min:
            raise ValueError("wn_max must exceed wn_min")
        if self.n_spectral < 2:
            raise ValueError("n_spectral must be >= 2")
        pitch_mm = self.detector_pitch_um * 1e-3
        self.n_spatial = int(round(self.slit_length_mm * self.magnification / pitch_mm))
        if self.n_spatial < 2:
            raise ValueError("derived n_spatial < 2; check slit/mag/pitch")
        if self.keystone_px is not None:
            # pixel spec -> fractional magnification (total growth / slit length)
            self.keystone_frac = self.keystone_px / self.n_spatial
        self.eta = np.linspace(-1.0, 1.0, self.n_spatial)
        self.nu  = np.linspace(self.wn_min, self.wn_max, self.n_spectral)
        self.dnu = (self.wn_max - self.wn_min) / (self.n_spectral - 1)
        self._theta = np.radians(self.clocking_deg)
        self.ils = ILS.from_fwhm_nm(self.fwhm_nm, 0.5 * (self.wn_min + self.wn_max))

    # -- geometry ----------------------------------------------------------
    @property
    def frac(self) -> np.ndarray:
        """In-band spectral fraction ``f_j ∈ [0, 1]`` (0 = blue, 1 = red)."""
        return (self.nu - self.wn_min) / (self.wn_max - self.wn_min)

    def column_shift(self, eta_i: float) -> np.ndarray:
        """Spectral column shift [px] for slit position ``eta_i`` (+ = toward red).

        Smile (``+smile_px_edge·f·η²``, grows toward red) plus the clocking line
        tilt (``−θ·η·(n_spatial/2)``, constant across the band).  A line images at
        ``nominal_column + column_shift``.
        """
        smile = self.smile_px_edge * self.frac * (eta_i ** 2)
        clock = -self._theta * (self.n_spatial / 2.0) * eta_i
        return smile + clock

    def spectral_shift(self, eta_i: float) -> np.ndarray:
        """Signed wavenumber shift ``δν = ν_eff − ν_j`` [cm⁻¹] for slit position ``eta_i``."""
        return -self.dnu * self.column_shift(eta_i)

    def channel_centers(self, eta_i: float) -> np.ndarray:
        """Distortion-shifted channel centres [cm⁻¹] for row at slit position ``eta_i``."""
        return self.nu + self.spectral_shift(eta_i)

    def wavelength_grid(self) -> np.ndarray:
        """Per-pixel effective wavenumber ``ν_eff[i, j]`` [cm⁻¹] — the spectral distortion map.

        This is the calibration data product a downstream retrieval would need:
        the wavenumber each detector pixel actually samples (smile + clocking).
        """
        cs = self.smile_px_edge * self.frac[None, :] * (self.eta[:, None] ** 2) \
            - self._theta * (self.n_spatial / 2.0) * self.eta[:, None]
        return self.nu[None, :] - self.dnu * cs

    def line_locus(self, wn0: float) -> np.ndarray:
        """Detector column (fractional) where monochromatic ``wn0`` images, per row.

        Solves ``ν_eff(i, j) = wn0`` for the column index — the smile parabola plus
        the clocking tilt.  Handy for overlaying the curvature on an ``imshow``.
        """
        span = self.wn_max - self.wn_min
        a = self.smile_px_edge * (self.eta ** 2)                    # smile px coeff (× f_j)
        c = -self._theta * (self.n_spatial / 2.0) * self.eta        # clocking px (const in j)
        # ν_j − dnu·(a·f_j + c) = wn0, with f_j = (ν_j − wn_min)/span.
        A = self.dnu * a / span
        nu_j = (wn0 + self.dnu * c - A * self.wn_min) / (1.0 - A)
        return (nu_j - self.wn_min) / self.dnu

    def keystone_stretch(self) -> np.ndarray:
        """Per-column spatial magnification factor ``1 + keystone_frac·f_j``."""
        return 1.0 + self.keystone_frac * self.frac

    def clocking_shear(self) -> np.ndarray:
        """Per-column spatial offset [η units] from the clocking rotation.

        The slit image shears along dispersion: ``δη(j) = θ·(j − n_spectral/2)``
        pixels, converted to the normalized slit coordinate.  Zero at band centre.
        """
        jidx = np.arange(self.n_spectral)
        return self._theta * (jidx - self.n_spectral / 2.0) * (2.0 / self.n_spatial)

    def pixel_bandwidth_um(self) -> np.ndarray:
        """Per-pixel spectral bandwidth ``Δλ_eff[i, j]`` [µm] from the distorted grid.

        The local dispersion width each pixel integrates.  Smile compresses/stretches
        it across the array; a rigid clocking shift leaves it unchanged.  This is the
        weight that makes the spectral integral energy-conserving.
        """
        lam = 1.0e4 / self.wavelength_grid()               # µm per pixel (smile+clocking)
        return np.abs(np.gradient(lam, axis=1))

    def apply_spatial_psf(self, A: np.ndarray) -> np.ndarray:
        """Blur ``A`` along the spatial (row) axis by the N/S PSF.

        A normalized Gaussian of FWHM ``spatial_psf_fwhm_px`` pixels, with edge
        extension so the slit ends are not darkened.  Returns ``A`` unchanged when
        ``spatial_psf_fwhm_px <= 0``.
        """
        return gaussian_blur_rows(A, self.spatial_psf_fwhm_px)

    # -- rendering ---------------------------------------------------------
    def image(self, wn_hires: np.ndarray,
              radiance: Callable[[float], np.ndarray],
              units: str = "radiance") -> np.ndarray:
        """Render the discrete detector image ``A[i, j]``.

        Parameters
        ----------
        wn_hires : ndarray, shape (n_hires,)
            Hi-res wavenumber grid [cm⁻¹], monotonically increasing, covering the
            band with ILS wings.
        radiance : callable
            ``radiance(η) -> S_hires`` — the hi-res spectrum at object slit
            position ``η ∈ [-1, 1]`` (see :func:`uniform_scene`, :func:`edge_scene`).
        units : {'radiance', 'energy', 'electrons'}
            ``'radiance'`` (default) returns the resampled radiance field
            ``L[i, j]`` — the ray-invariant, correct for a lossless remap but *not*
            flux-conserving when summed with uniform bins.  ``'energy'`` (alias
            ``'electrons'``) returns the **flux-conserving** signal
            ``L·Δλ_eff·/stretch``: radiance weighted by the smile-distorted local
            bandwidth (:meth:`pixel_bandwidth_um`) and the keystone plate-scale
            factor ``1/stretch`` (clocking is a rotation, so area-preserving — no
            factor).  It is proportional to collected energy up to the instrument
            throughput (étendue·τ·QE·t·photon-energy), which the radiometry layer
            (:class:`gert.radiometry.RadiometricNoise`) applies — so it is *not*
            literal electron counts, but it is what that layer should integrate.

        Returns
        -------
        ndarray, shape (n_spatial, n_spectral)

        Notes
        -----
        A residual non-conservation remains at the **band edges**, where the
        normalized ILS convolution loses area to kernel truncation, and where the
        keystone/clocking resample clamps at the slit ends.  These are edge effects,
        not present in the array interior.
        """
        if units not in ("radiance", "energy", "electrons"):
            raise ValueError("units must be 'radiance', 'energy' or 'electrons'")
        wn_hires = np.asarray(wn_hires, dtype=float)

        # Step 1 — spectral distortion (smile + clocking tilt) + convolution,
        # per row, at its object η, via shifted ILS centres.
        A0 = np.empty((self.n_spatial, self.n_spectral), dtype=float)
        for i, eta_i in enumerate(self.eta):
            S_i = np.asarray(radiance(float(eta_i)), dtype=float)
            A0[i] = self.ils.convolve(wn_hires, S_i, self.channel_centers(eta_i),
                                      exact_center=True)

        # Step 2 — spatial resample: keystone magnification + clocking shear.
        # Detector row i samples object coordinate  η_obj = η_i / stretch_j − shear_j.
        if self.keystone_frac == 0.0 and self.clocking_deg == 0.0:
            A = A0
        else:
            stretch = self.keystone_stretch()
            shear   = self.clocking_shear()
            A = np.empty_like(A0)
            for j in range(self.n_spectral):
                A[:, j] = np.interp(self.eta / stretch[j] - shear[j], self.eta, A0[:, j])

        # Step 3 — along-slit (N/S) PSF blur: the spatial analog of the ILS.
        A = self.apply_spatial_psf(A)

        if units == "radiance":
            return A
        # energy / electrons: weight to conserve flux (spectral bandwidth + plate scale)
        return A * self.pixel_bandwidth_um() / self.keystone_stretch()[None, :]
