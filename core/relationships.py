
def generate_candidates(profile):
    """
    Generate analytical candidates from the dataset profile.

    Candidate types:

    1. Dimension → Measure
    2. Measure ↔ Measure
    3. Time → Measure

    The function is fully generic and does not depend
    on specific column names.
    """

    columns = profile["column_profiles"]

    dimensions = [
        column["column"]
        for column in columns
        if column["semantic_role"] == "dimension"
    ]

    measures = [
        column["column"]
        for column in columns
        if column["semantic_role"] == "measure"
    ]

    # Trends are analysed on the primary time axis only; secondary
    # timestamps (e.g. "last activity") produce noisy, duplicate trends.
    primary_time = profile.get("primary_time")

    time_columns = (
        [primary_time]
        if primary_time
        else [
            column["column"]
            for column in columns
            if column["semantic_role"] == "time"
        ]
    )

    candidates = []

    # --------------------------------------------------
    # 1. Dimension → Measure
    # --------------------------------------------------

    for dimension in dimensions:

        for measure in measures:

            candidates.append({
                "type": "dimension_measure",
                "dimension": dimension,
                "measure": measure
            })

    # --------------------------------------------------
    # 2. Measure ↔ Measure
    # --------------------------------------------------

    for i in range(len(measures)):

        for j in range(i + 1, len(measures)):

            candidates.append({
                "type": "measure_measure",
                "x": measures[i],
                "y": measures[j]
            })

    # --------------------------------------------------
    # 3. Time → Measure
    # --------------------------------------------------

    for time_column in time_columns:

        for measure in measures:

            candidates.append({
                "type": "time_measure",
                "time": time_column,
                "measure": measure
            })

    return candidates



