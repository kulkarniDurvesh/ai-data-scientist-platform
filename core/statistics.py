import pandas as pd


def calculate_dimension_measure_stats(
    df,
    dimension,
    measure
):
    grouped = (
        df.groupby(dimension)[measure]
        .agg(
            count="count",
            sum="sum",
            mean="mean",
            median="median",
            min="min",
            max="max",
            std="std"
        )
        .reset_index()
    )

    return grouped


def calculate_measure_measure_stats(
    df,
    x,
    y
):
    data = df[[x, y]].dropna()

    if len(data) < 2:
        return {
            "count": len(data),
            "pearson": None,
            "spearman": None
        }

    pearson = data[x].corr(
        data[y],
        method="pearson"
    )

    spearman = data[x].corr(
        data[y],
        method="spearman"
    )

    return {
        "count": len(data),
        "pearson": pearson,
        "spearman": spearman
    }


def calculate_time_measure_stats(
    df,
    time_column,
    measure,
    frequency="month"
):

    data = df[
        [time_column, measure]
    ].dropna().copy()

    data[time_column] = pd.to_datetime(
        data[time_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[time_column]
    )

    if frequency == "month":

        data["period"] = (
            data[time_column]
            .dt.to_period("M")
        )

    elif frequency == "week":

        data["period"] = (
            data[time_column]
            .dt.to_period("W")
        )

    elif frequency == "day":

        data["period"] = (
            data[time_column]
            .dt.to_period("D")
        )

    else:
        raise ValueError(
            "Unsupported frequency"
        )

    result = (
        data.groupby("period")[measure]
        .agg(
            count="count",
            sum="sum",
            mean="mean"
        )
        .reset_index()
    )

    return result

def calculate_statistics(
    df,
    candidate
):

    candidate_type = candidate["type"]

    if candidate_type == "dimension_measure":

        return calculate_dimension_measure_stats(
            df,
            candidate["dimension"],
            candidate["measure"]
        )

    elif candidate_type == "measure_measure":

        return calculate_measure_measure_stats(
            df,
            candidate["x"],
            candidate["y"]
        )

    elif candidate_type == "time_measure":

        return calculate_time_measure_stats(
            df,
            candidate["time"],
            candidate["measure"]
        )

    else:

        raise ValueError(
            f"Unsupported candidate type: {candidate_type}"
        )