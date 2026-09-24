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

**Metadata and the exact-match requirement** (2026-09-24, v8): every entry is stored as
``{"_meta": {cache_version, inputs, provenance}, "data": <band dict>}``. ``inputs`` is the exact
set of render inputs (including the RT solver the render used); ``provenance`` (timestamp, git
head, SLURM job/array task, host, argv) is for tracing only. :func:`load` serves an entry ONLY if
it has ``_meta``, its version equals :data:`TRUTH_CACHE_VERSION`, and its recorded ``inputs`` equal
the requested ones exactly -- otherwise it is a miss and the caller must re-render (the reason is
printed). Query with :func:`describe` or ``scripts/list_truth_cache.py``.

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
TRUTH_CACHE_VERSION = 8  # 2026-09-24: entries now carry `_meta` (exact render inputs + provenance) and are
                         # served ONLY on an exact inputs match; the render solver is part of the inputs. v7 and
                         # older entries (no metadata) are never served.
                         # (v7) 2026-09-21 (later): terrain-following aerosol height (p_surface - offset) and an urban
                         # AOD bump at the -500 km CO2 plume -- both change every aerosol render; v5 entries are stale.
                         # (v5) 2026-09-21: the truth aerosol height is now capped >= 100 m above the surface
                         # (along_slit_scene.height_aerosol), so every aerosol render changes; v4 entries are stale.
                         # (v4) 2026-09-20: eta convention changed (gd_polynomials.eta_of_s,
                         # slit-image centred/scaled; was s/s_max) -- every rendered
                         # image places the scene at different detector rows, so every
                         # version-3 entry is stale and must not be served.
                         # (v3) 2026-09-09: (a) Sec.11 bug 4 + Sec.12 both changed the
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


def _normalize(inputs: dict) -> dict:
    """`{name: repr(value)}`, sorted -- JSON-friendly, human-readable, and comparable with `==`
    regardless of how the caller built the dict."""
    return {k: repr(v) for k, v in sorted(inputs.items())}


def _provenance() -> dict:
    """Who/what/when produced an entry. NOT part of the exact-match requirement (a different
    commit or job legitimately reuses a valid render) -- recorded so an entry can be traced."""
    import datetime
    import socket
    import subprocess
    import sys
    git_head = None
    try:
        git_head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=20).stdout.strip() or None
    except Exception:
        pass                                   # git may not be on PATH (no `module load git` in a job)
    return dict(created=datetime.datetime.now().isoformat(timespec="seconds"), git_head=git_head,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                host=socket.gethostname(), user=os.environ.get("USER"), argv=list(sys.argv))


def load(key: str, inputs: dict, root: Optional[Path] = None, verbose: bool = True) -> Optional[dict]:
    """The cached band dict, or ``None`` (a miss -- the caller re-renders).

    HARD REQUIREMENT (2026-09-24, user: "induce a hard requirement that the experiment build a
    new truth cache without exact matches to metadata"): an entry is served only if it carries
    ``_meta`` AND its recorded ``cache_version`` equals :data:`TRUTH_CACHE_VERSION` AND its
    recorded render ``inputs`` equal ``inputs`` exactly. Anything else -- no metadata (every
    pre-v8 entry), a version or inputs mismatch, an unreadable file -- is a miss, with the reason
    printed so a re-render is never a mystery. The key hash alone is deliberately not trusted.
    """
    path = (root or cache_root()) / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except Exception as e:
        if verbose:
            print(f"  truth cache entry {key} unreadable ({type(e).__name__}) -- treating as a miss", flush=True)
        return None
    meta = blob.get("_meta") if isinstance(blob, dict) else None
    if meta is None or "data" not in blob:
        if verbose:
            print(f"  truth cache entry {key} has NO metadata (pre-v8 format) -- refusing to serve it", flush=True)
        return None
    if meta.get("cache_version") != TRUTH_CACHE_VERSION:
        if verbose:
            print(f"  truth cache entry {key}: version {meta.get('cache_version')} != {TRUTH_CACHE_VERSION} -- miss", flush=True)
        return None
    want = _normalize(inputs)
    have = meta.get("inputs", {})
    if have != want:
        diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        if verbose:
            print(f"  truth cache entry {key}: inputs differ on {diff} -- miss", flush=True)
        return None
    return blob["data"]


def save(key: str, data: dict, inputs: dict, root: Optional[Path] = None) -> Path:
    """Write ``data`` under ``key`` together with its ``_meta`` (exact render ``inputs``, version,
    provenance), atomically (temp file + rename) so a concurrent reader never observes a partial
    file."""
    root = root or cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pkl"
    tmp = path.with_suffix(f".pkl.tmp{os.getpid()}")
    meta = dict(cache_version=TRUTH_CACHE_VERSION, inputs=_normalize(inputs), provenance=_provenance())
    with open(tmp, "wb") as f:
        pickle.dump(dict(_meta=meta, data=data), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def describe(key: str, root: Optional[Path] = None) -> Optional[dict]:
    """The stored ``_meta`` of entry ``key`` (inputs, version, provenance), or ``None`` if the
    entry is missing/unreadable/pre-v8. Loads the whole pickle -- fine at ~10 MB per entry."""
    path = (root or cache_root()) / f"{key}.pkl"
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return blob.get("_meta") if isinstance(blob, dict) else None
    except Exception:
        return None
