INTENT_KEYWORDS = {
    "COUNT": [
        "how many",
        "count",
        "number of"
    ],

    "AVERAGE": [
        "average",
        "mean"
    ],

    "SUM": [
        "sum",
        "total"
    ],

    "MIN": [
        "minimum",
        "lowest",
        "smallest"
    ],

    "MAX": [
        "maximum",
        "highest",
        "largest"
    ],
    
    "TOP_N": [
        "top"
    ],

    "BOTTOM_N": [
        "bottom"
    ],
    "FILTER": [
        "show",
        "find",
        "list",
        "get",
        "give",
        "display"
    ]
}


# Words that make a question a list request when no other intent
# matched: "Which doctors are in Pune North?"
FALLBACK_LIST_WORDS = ["which", "who", "what are"]


# A chart is requested when one of these words appears...
CHART_WORDS = r"\b(?:chart|graph|plot|visuali[sz]e|visuali[sz]ation|diagram)\b"

# ...and its type comes from the first matching pattern:
# (chart type, pattern, enough on its own without CHART_WORDS)
CHART_TYPE_PATTERNS = [
    ("PIE", r"\b(?:pie|donut|doughnut)\b", True),
    ("HISTOGRAM", r"\bhistogram\b", True),
    ("SCATTER", r"\bscatter\b", True),
    ("BOX", r"\bbox\s*(?:plot|chart)\b", False),
    ("BAR", r"\bbar\b", False),
    ("LINE", r"\b(?:line|trend)\b", False),
]


# Calendar units a chart can be grouped by: "visits by month".
TIME_GRAIN_WORDS = {
    "day": "D",
    "daily": "D",
    "week": "W",
    "weekly": "W",
    "month": "M",
    "monthly": "M",
    "quarter": "Q",
    "quarterly": "Q",
    "year": "Y",
    "yearly": "Y",
}


OPERATORS = {
    "greater than": ">",
    "more than": ">",
    "above": ">",
    "less than": "<",
    "below": "<",
    "under": "<",
    "at least": ">=",
    "greater than or equal to": ">=",
    "at most": "<=",
    "less than or equal to": "<=",
    "equal to": "=",
    "equals": "=",
    "is": "=",
    "not equal to": "!="
}

