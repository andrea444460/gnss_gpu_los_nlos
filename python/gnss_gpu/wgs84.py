"""WGS84 ellipsoid (EPSG:4979; NIMA TR8350.2 defining parameters).

The semi-major axis ``a`` is an exact defining constant in the standard (6378137 m),
not a fitted value, so it has no extra fractional metres beyond ``6378137.0``.

``1/f`` is the defining reciprocal flattening; ``e²``, ``b``, and mean radius are
derived in IEEE-754 double precision from ``a`` and ``1/f``.
"""

from __future__ import annotations

# --- Defining ---
WGS84_SEMI_MAJOR_AXIS_M: float = 6378137.0
"""Semi-major (equatorial) axis [m]; exact WGS84 defining constant."""

WGS84_INV_FLATTENING: float = 298.257223563
"""Reciprocal flattening :math:`1/f` (defining)."""

# --- Derived in float64 from defining constants ---
WGS84_FLATTENING: float = 1.0 / WGS84_INV_FLATTENING
WGS84_FIRST_ECCENTRICITY_SQ: float = (
    2.0 * WGS84_FLATTENING - WGS84_FLATTENING * WGS84_FLATTENING
)
"""First eccentricity squared :math:`e^2 = f(2-f)`."""

WGS84_SEMI_MINOR_AXIS_M: float = WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
"""Semi-minor (polar) axis :math:`b = a(1-f)` [m]."""

WGS84_ARITHMETIC_MEAN_RADIUS_M: float = (
    (2.0 * WGS84_SEMI_MAJOR_AXIS_M + WGS84_SEMI_MINOR_AXIS_M) / 3.0
)
"""IUGG spherical mean radius :math:`R_1 = (2a+b)/3` [m] from this ellipsoid."""
