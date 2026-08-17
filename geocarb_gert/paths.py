"""Where this project's external inputs live, resolved rather than hardcoded.

Every driver in ``scripts/`` needs ``gert``'s ``input/`` tree (``absco.h5``,
``solar.h5``). Until 2026-08-17 each one carried its own
``GERT_ROOT = Path("/scratch/scrowel3_lab/gert")`` -- correct on the HPC,
silently wrong anywhere else, and duplicated across seven files so fixing it
in one place fixed it in only one place. Worse, the failure mode is a
confusing ``No such file`` deep inside an ABSCO load rather than an obvious
"I cannot find gert".

:func:`gert_root` centralises the lookup so a relocation is a no-op. It is
deliberately a *search* over candidate layouts rather than a single guess:
an earlier attempt hardcoded ``../gert``, which is one level too high (the
repo sits at ``<parent>/research/geocarb_simulator`` and gert at
``<parent>/gert``), and fell through to the HPC path without complaint.
Each candidate is validated by an actual ``input/`` directory, since that is
the thing callers actually need -- a bare directory named ``gert`` is not
enough.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Layouts tried, in order, relative to the repository root.
#: ``../../gert`` is the documented relationship (``geocarb_gert`` is an
#: adapter over a gert checkout two levels up); ``../gert`` covers a flat
#: side-by-side layout.
_CANDIDATES = ("../../gert", "../gert")

#: Final fallback: the HPC scratch checkout every batch script and every
#: pre-2026-08-17 run used. Kept last so cluster behaviour is unchanged.
_HPC_FALLBACK = "/scratch/scrowel3_lab/gert"

REPO_ROOT = Path(__file__).resolve().parent.parent


def gert_root(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the gert checkout that owns ``input/absco/absco.h5``.

    Order: `explicit` argument, then ``$GERT_ROOT``, then each entry of
    :data:`_CANDIDATES` relative to the repo root, then the HPC fallback.
    The first three are validated by the presence of an ``input/``
    directory; the fallback is returned unvalidated so the error surfaces at
    the actual load site with a real path in the message, exactly as before.

    Parameters
    ----------
    explicit : path-like, optional
        A caller-supplied override (e.g. an ``--gert-root`` CLI argument).
        Returned as-is when given, so an explicit choice is never
        second-guessed.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("GERT_ROOT")
    if env:
        return Path(env)
    for rel in _CANDIDATES:
        cand = (REPO_ROOT / rel).resolve()
        if (cand / "input").is_dir():
            return cand
    return Path(_HPC_FALLBACK)


def describe_gert_root() -> str:
    """One-line provenance string for logs -- which path was chosen and why.

    Worth printing at the top of a long run: a sweep that silently picked
    the wrong checkout otherwise only reveals it minutes later, inside an
    ABSCO load.
    """
    root = gert_root()
    if os.environ.get("GERT_ROOT"):
        why = "$GERT_ROOT"
    elif str(root) == _HPC_FALLBACK:
        why = "HPC fallback (no local checkout found)"
    else:
        why = "auto-detected sibling checkout"
    ok = "ok" if (root / "input" / "absco" / "absco.h5").exists() else "MISSING absco.h5"
    return f"gert_root={root}  [{why}; {ok}]"
