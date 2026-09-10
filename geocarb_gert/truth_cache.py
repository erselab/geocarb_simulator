"""Cache for the deterministic (noise-free) rendered truth detector image
that ``scripts/gd_per_row_retrieve.py::_band_setup`` produces -- "having an observation
saved on disk" (user request, 2026-08-20), so a config-matrix sweep that
varies only retrieval-side knobs (``g_ratio``, ``anchor_density``, free
rows, ...) doesn't re-render the same scene from scratch for every config.
The render (``als.build_lookup_radiance`` + ``gd_render.image``) is the
dominant per-run cost for any non-``exact`` ``--prior-fields`` config
(``n_lookup_samples=5600`` instead of 400) and is identical across every
config in a matrix sweep that shares a scene.

Does **not** cache ``radiance`` (an unpicklable closure over
``build_lookup_radiance``'s own local ``spectra`` array -- see that
function's implementation) or ``noise_arr`` (a random per-call
realization, only meaningful under ``noise=True``, which no cached caller
uses). Every current consumer of this cache
(``gd_joint_block_retrieve.py``) only reads ``A``/``wn_hires``/
``ils`` from the returned dict; a future caller that needs ``radiance``
must bypass the cache -- a bare ``KeyError`` on the returned dict makes
that failure loud rather than silently serving a wrong/incomplete band.

**Cache key**: a hash of every input that actually determines the render
(see :func:`cache_key`'s callers), plus :data:`TRUTH_CACHE_VERSION` --
bump that constant whenever a code change touches the deterministic
rendering path itself (``gd_render.image``, ``build_lookup_radiance``, the
GD polynomial CSV/loader, the spatial PSF convolution, ...), so stale
caches from before the change are never silently served. Forgetting to
bump it is the one failure mode this module cannot protect against by
construction -- there is no way to detect "the code changed" from the
data alone.

**Concurrency**: no locking. Parallel SLURM array tasks racing to render
the *same* scene on a cold cache will each render it independently (no
speedup for that first concurrent batch) and then race to write -- safe
because :func:`save` writes to a temp file and renames atomically (POSIX
guarantee), so a reader never sees a partial file; the loser's write is
simply overwritten by (or overwrites) an equally-valid cache entry, never
a corrupt one. :func:`load` treats any read failure as a cache miss, not
an error, for the same reason -- the fallback is always just "re-render
it," so a corrupt or half-written file already skipped by `save`'s
atomicity can never break a caller.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Optional

from .paths import REPO_ROOT

#: Bump on ANY change to the deterministic rendering path (gd_render.py,
#: along_slit_scene.build_lookup_radiance, the GD polynomial CSV/loader,
#: focalplane.py's spatial PSF convolution, ...) -- this is the only guard
#: against silently serving a stale cached render after such a change.
TRUTH_CACHE_VERSION = 3  # 2026-09-09: (a) Sec.11 bug 4 + Sec.12 both changed the
                         # deterministic truth-render path (aerosol threading, the
                         # simulate_spectrum consolidation) without bumping this;
                         # (b) _band_setup's aerosol pull is now opt-in via
                         # `with_aerosol` (Sec.14 follow-up) -- the default render
                         # is aerosol-free again, so every version-2 entry (all
                         # rendered WITH background aerosol) must be invalidated.
                         # 2026-08-28 (v2): ALBEDO_CORR_KM 10km -> 0.5km, ALBEDO_COV's
                         # renormalization now grid-dependent (along_slit_scene.
                         # _correlated_field).

_CACHE_ROOT_ENV = "GEOCARB_TRUTH_CACHE_ROOT"


def cache_root(explicit: Optional[str] = None) -> Path:
    """Resolve the truth-cache directory. Same search-then-fallback shape
    as ``paths.gert_root()``/``paths.config_root()``: explicit argument,
    then ``$GEOCARB_TRUTH_CACHE_ROOT``, then ``<repo_root>/results/
    truth_cache`` (``results/`` is already gitignored wholesale, so this
    needs no separate ignore entry)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(_CACHE_ROOT_ENV)
    if env:
        return Path(env)
    return REPO_ROOT / "results" / "truth_cache"


def cache_key(**fields) -> str:
    """Stable hash over every keyword field given. Callers pass exactly
    the inputs that determine the render (see this module's docstring for
    what that means for ``_band_setup``); order-independent (sorted
    items), so callers don't need to worry about keyword order.
    :data:`TRUTH_CACHE_VERSION` is folded in automatically."""
    payload = repr(sorted(fields.items())).encode("utf-8")
    h = hashlib.sha256(payload)
    h.update(str(TRUTH_CACHE_VERSION).encode("utf-8"))
    return h.hexdigest()[:24]


def load(key: str, root: Optional[Path] = None) -> Optional[dict]:
    """The cached dict, or ``None`` on a cache miss -- no file, or a
    corrupt/unreadable one (treated as a miss, not an error: the caller's
    fallback is always just 're-render it')."""
    path = (root or cache_root()) / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save(key: str, data: dict, root: Optional[Path] = None) -> Path:
    """Write ``data`` under ``key``, atomically (temp file + rename) so a
    concurrent reader never observes a partial file."""
    root = root or cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pkl"
    tmp = path.with_suffix(f".pkl.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path
