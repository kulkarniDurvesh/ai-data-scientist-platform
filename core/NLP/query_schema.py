from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QueryFilter:
    column: str
    operator: str
    value: Any


@dataclass
class ParsedQuery:
    intent: str
    entity: Optional[str] = None
    filters: list[QueryFilter] = field(default_factory=list)
    subject: Optional[str] = None
    limit: int | None = None
    # Used for grouped analytical queries such as:
    # "average TotalSales by Region"
    group_by: Optional[str] = None

    # Used for grouped aggregation:
    # AVERAGE, SUM, COUNT, MIN, MAX
    aggregation: Optional[str] = None

    # Requested chart type for "pie chart of X by Y" questions.
    chart_type: Optional[str] = None

    # Calendar grain when grouping by a date unit ("by month").
    time_grain: Optional[str] = None

    # Remarks shown with the answer (e.g. "March spans 2024 and 2025").
    notes: list[str] = field(default_factory=list)

    # ID-like values from the question ("MR01") found in no column.
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "intent": self.intent,
            "entity": self.entity,
            "filters": [
                {
                    "column": f.column,
                    "operator": f.operator,
                    "value": f.value
                }
                for f in self.filters
            ],
            "subject": self.subject,
            "group_by": self.group_by,
            "aggregation": self.aggregation,
            "limit": self.limit,
            "chart_type": self.chart_type,
            "time_grain": self.time_grain,
            "notes": self.notes,
            "unresolved": self.unresolved,
        }



