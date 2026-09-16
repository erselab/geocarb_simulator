"""Docs-build-only stub for the (still-private, separate) `gert` package
-- see this file's own directory README-equivalent note in
docs/conf.py's own comment at the sys.path.insert call site.

NOT the real library. Every name geocarb_simulator's own modules import
FROM gert (checked directly, 2026-09-16: `grep -rn "^from gert\.\|^import gert"`
across geocarb_gert/*.py and the three documented scripts) gets a plain
placeholder class/function/constant here -- just enough for Sphinx
autodoc to import geocarb_simulator's own modules for docstring
extraction. Real, physically-correct values ONLY where a module does
real arithmetic on them at IMPORT TIME (gert.levels's two constants,
consumed by geocarb_gert/levels.py at module scope) -- everything else
is a bare placeholder since it's only ever referenced in type hints or
instantiated inside function bodies (never called during THIS stub's own
import), which autodoc never executes.
"""
