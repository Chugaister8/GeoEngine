"""
GeoEngine — dem пакет
DEM завантаження, обробка та аналіз.
"""

from .loader import DEMTile, DEMLoader, DEMLoadError, NODATA_DEFAULT
from .sources import (
    DEMSourceID,
    DEMSourceManager,
    DEMSourceAPIError,
    SourceConfig,
    SOURCES,
)
from .processor import merge_tiles, fill_gaps, smooth
from .analysis import (
    SlopeResult,
    AspectResult,
    HillshadeResult,
    ContourResult,
    ProfileResult,
    compute_slope,
    compute_aspect,
    compute_hillshade,
    compute_contours,
    compute_profile,
)

__all__ = [
    # Loader
    "DEMTile",
    "DEMLoader",
    "DEMLoadError",
    "NODATA_DEFAULT",
    # Sources
    "DEMSourceID",
    "DEMSourceManager",
    "DEMSourceAPIError",
    "SourceConfig",
    "SOURCES",
    # Processor
    "merge_tiles",
    "fill_gaps",
    "smooth",
    # Analysis
    "SlopeResult",
    "AspectResult",
    "HillshadeResult",
    "ContourResult",
    "ProfileResult",
    "compute_slope",
    "compute_aspect",
    "compute_hillshade",
    "compute_contours",
    "compute_profile",
]
