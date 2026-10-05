"""Segmentation, anomaly detection and correlation analysis (Phase 3d)."""

from .engine import SegmentResult, SegmentSpec, build_segments
from .units import build_units, feature_options, unit_options

__all__ = [
    "SegmentResult",
    "SegmentSpec",
    "build_segments",
    "build_units",
    "feature_options",
    "unit_options",
]
