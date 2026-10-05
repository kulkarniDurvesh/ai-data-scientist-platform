"""KPI definitions layer: optional domain files with roles and KPI formulas (Phase 4b)."""

from .config import ConfigError, DomainConfig, KpiDef, load_domain, parse_domain
from .evaluate import KpiReport, evaluate_domain, format_value
from .starter import DEFAULT_FOLDER, discover, starter_yaml

__all__ = [
    "DEFAULT_FOLDER",
    "ConfigError",
    "DomainConfig",
    "KpiDef",
    "KpiReport",
    "discover",
    "evaluate_domain",
    "format_value",
    "load_domain",
    "parse_domain",
    "starter_yaml",
]
