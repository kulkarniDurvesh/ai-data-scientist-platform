"""
Domain configuration: roles and KPI definitions in a YAML file.

This is the only place where domain words (rep, customer, territory, a
status value) may appear. The platform works without any domain file.

    name: Sales field force
    tables:                      # optional default table per role lookup
    roles:
      rep: RepCode               # same column in every table
      territory:                 # or a column per table
        Contacts: Rep Region
        Accounts: Region
      date:
        Contacts: ContactedOn
    kpis:
      - id: win_rate
        label: Win rate
        table: Contacts
        numerator:   {count: rows, where: {outcome: Deal Won}}
        denominator: {count: rows}
        format: percent           # percent | number | currency
        direction: up             # up = higher is better, down = lower is better
        description: Contacts that ended in a won deal.

Aggregations: count (rows), sum, mean, min, max, distinct (column or role).
Filters (where): role or column -> value | [values] | {gt|ge|lt|le|ne: v}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

OPERATIONS = {"count", "sum", "mean", "min", "max", "distinct"}
COMPARATORS = {"gt", "ge", "lt", "le", "ne", "eq"}
FORMATS = {"percent", "number", "currency"}
DIRECTIONS = {"up", "down"}


class ConfigError(ValueError):
    """A domain file that can't be used, with a readable reason."""


@dataclass
class Aggregate:
    op: str                          # count | sum | mean | min | max | distinct
    target: str | None               # column or role (None for count of rows)
    where: dict[str, Any] = field(default_factory=dict)
    table: str | None = None         # overrides the KPI's table

    def describe(self) -> str:
        what = "rows" if self.op == "count" else f"{self.op} of {self.target}"
        if self.op == "distinct":
            what = f"distinct {self.target}"
        if self.where:
            conditions = []
            for key, value in self.where.items():
                if isinstance(value, dict):
                    conditions += [f"{key} {op} {v}" for op, v in value.items()]
                elif isinstance(value, list):
                    conditions.append(f"{key} in {', '.join(map(str, value))}")
                else:
                    conditions.append(f"{key} = {value}")
            what += " where " + " and ".join(conditions)
        if self.table:
            what += f" (in {self.table})"
        return what


@dataclass
class KpiDef:
    id: str
    label: str
    numerator: Aggregate
    denominator: Aggregate | None = None
    table: str | None = None
    format: str = "number"
    direction: str = "up"
    description: str = ""

    def definition(self) -> str:
        if self.denominator is None:
            return self.numerator.describe()
        return f"{self.numerator.describe()} ÷ {self.denominator.describe()}"


@dataclass
class DomainConfig:
    name: str
    roles: dict[str, str | dict[str, str]]
    kpis: list[KpiDef]
    description: str = ""
    path: str | None = None

    def column(self, role_or_column: str, table: str | None) -> str:
        """Column for a role in a table; anything that isn't a role is a column name."""

        mapping = self.roles.get(role_or_column)
        if mapping is None:
            return role_or_column
        if isinstance(mapping, str):
            return mapping
        if table in mapping:
            return mapping[table]
        if "*" in mapping:
            return mapping["*"]
        raise ConfigError(f"Role '{role_or_column}' has no column for table '{table}'.")

    def has_role(self, role: str, table: str | None) -> bool:
        mapping = self.roles.get(role)
        if mapping is None:
            return False
        return isinstance(mapping, str) or table in mapping or "*" in mapping

    def group_roles(self) -> list[str]:
        """Roles that can be used to break KPIs down (everything except the date)."""
        return [role for role in self.roles if role != "date"]


def load_domain(path: str | Path) -> DomainConfig:
    text = Path(path).read_text(encoding="utf-8")
    config = parse_domain(yaml.safe_load(text) or {})
    config.path = str(path)
    return config


def parse_domain(data: dict[str, Any]) -> DomainConfig:
    if not isinstance(data, dict):
        raise ConfigError("A domain file must be a mapping with 'name', 'roles' and 'kpis'.")

    roles = data.get("roles") or {}
    if not isinstance(roles, dict):
        raise ConfigError("'roles' must map role names to columns.")
    for role, mapping in roles.items():
        if not isinstance(mapping, (str, dict)):
            raise ConfigError(f"Role '{role}' must be a column name or a table -> column mapping.")

    kpis = []
    seen = set()
    for index, raw in enumerate(data.get("kpis") or [], start=1):
        if not isinstance(raw, dict) or "id" not in raw:
            raise ConfigError(f"KPI #{index} needs an 'id'.")
        kpi_id = str(raw["id"])
        if kpi_id in seen:
            raise ConfigError(f"KPI id '{kpi_id}' is used twice.")
        seen.add(kpi_id)

        if "value" in raw:
            numerator, denominator = _aggregate(raw["value"], kpi_id), None
        elif "numerator" in raw:
            numerator = _aggregate(raw["numerator"], kpi_id)
            denominator = _aggregate(raw["denominator"], kpi_id) if raw.get("denominator") else None
        else:
            raise ConfigError(f"KPI '{kpi_id}' needs 'value' or 'numerator'.")

        fmt = str(raw.get("format", "percent" if denominator else "number"))
        direction = str(raw.get("direction", "up"))
        if fmt not in FORMATS:
            raise ConfigError(f"KPI '{kpi_id}': format must be one of {sorted(FORMATS)}.")
        if direction not in DIRECTIONS:
            raise ConfigError(f"KPI '{kpi_id}': direction must be 'up' or 'down'.")

        kpis.append(KpiDef(
            id=kpi_id,
            label=str(raw.get("label", kpi_id.replace("_", " ").title())),
            numerator=numerator,
            denominator=denominator,
            table=raw.get("table"),
            format=fmt,
            direction=direction,
            description=str(raw.get("description", "")),
        ))

    if not kpis:
        raise ConfigError("The domain file defines no KPIs.")

    return DomainConfig(
        name=str(data.get("name", "Domain")),
        roles={str(k): v for k, v in roles.items()},
        kpis=kpis,
        description=str(data.get("description", "")),
    )


def _aggregate(raw: Any, kpi_id: str) -> Aggregate:
    if not isinstance(raw, dict):
        raise ConfigError(f"KPI '{kpi_id}': each aggregate must be a mapping like {{count: rows}}.")

    operations = [key for key in raw if key in OPERATIONS]
    if len(operations) != 1:
        raise ConfigError(f"KPI '{kpi_id}': use exactly one of {sorted(OPERATIONS)} in each aggregate.")
    op = operations[0]
    target = raw[op]
    if op == "count":
        target = None if target in (None, "rows") else str(target)
    elif not target:
        raise ConfigError(f"KPI '{kpi_id}': '{op}' needs a column or role.")

    where = raw.get("where") or {}
    if not isinstance(where, dict):
        raise ConfigError(f"KPI '{kpi_id}': 'where' must map columns or roles to values.")
    for key, value in where.items():
        if isinstance(value, dict) and not set(value) <= COMPARATORS:
            raise ConfigError(f"KPI '{kpi_id}': unknown comparison in where.{key}; use {sorted(COMPARATORS)}.")

    return Aggregate(op=op, target=None if target is None else str(target), where=where, table=raw.get("table"))
