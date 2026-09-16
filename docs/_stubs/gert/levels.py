"""See gert/__init__.py's own docstring. Real values -- consumed by
geocarb_gert/levels.py at MODULE IMPORT TIME, so a placeholder class
wouldn't do; GERT_P_SFC_STD is exact (gert/levels.py's own real value,
sea-level standard pressure), GERT_P_LEVELS is placeholder-shaped
(real gert_levels() also returns ndarray (20,)) but not physically
gert's own real sigma levels -- fine for docs, not for anything else.
"""
import numpy as np

GERT_P_SFC_STD: float = 101325.0
GERT_P_LEVELS: np.ndarray = np.linspace(100.0, GERT_P_SFC_STD, 20)
