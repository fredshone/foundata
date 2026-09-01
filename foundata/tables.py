from typing import Iterable, Optional

import polars as pl


def render_markdown_table(
    headers: Iterable[str], rows: Iterable[Iterable[str]]
) -> str:
    """Render a markdown pipe table with columns padded to their widest cell.

    Plain markdown pipe tables don't require aligned column widths to
    render correctly in a markdown viewer, but without padding, columns
    printed straight to a terminal (as this pipeline does) end up ragged —
    each column's width is fixed to its own cell, not the widest cell in
    that column. Padding every cell (header included) to the column's max
    width keeps both terminal output and rendered markdown aligned.
    """
    headers = [str(h) for h in headers]
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
            + " |"
        )

    lines = [
        _fmt_row(headers),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def group_null_pct(
    df: pl.DataFrame,
    group_cols: Iterable[str],
    ignore: Optional[Iterable[str]] = None,
    return_per_column: bool = False,
    return_overall: bool = True,
) -> pl.DataFrame:
    """
    Compute null percentages per group.
    - return_per_column=True: adds one column per input column with % nulls
    - return_overall=True: adds a single 'overall_null_pct' across all kept columns
    """
    ignore = list(ignore) if ignore else []
    kept_cols = df.select(pl.all().exclude(ignore)).columns
    k = len(kept_cols)
    if k == 0 and return_overall:
        raise ValueError(
            "No columns left after excluding; overall % null is undefined."
        )

    agg_exprs = []

    if return_per_column:
        agg_exprs.append((pl.all().exclude(ignore).is_null().mean() * 100))

    if return_overall:
        total_nulls_expr = pl.fold(
            acc=pl.lit(0),
            function=lambda acc, s: acc + s,
            exprs=[pl.col(c).is_null().sum() for c in kept_cols],
        )
        agg_exprs.append(
            (total_nulls_expr / (pl.len() * pl.lit(k)) * 100).alias(
                "overall_null_pct"
            )
        )

    summary = df.group_by(list(group_cols)).agg(agg_exprs)
    summary = summary.with_columns(
        pl.col(col).list.first()
        for col in summary.columns
        if summary[col].dtype == pl.List
    )
    return summary


def summary_table(
    attributes: pl.DataFrame, trips: pl.DataFrame, markdown: bool = False
) -> pl.DataFrame | str:
    """Produce a per-source summary table (persons, nulls %, trips, kms).

    If `markdown` is True, return a markdown-formatted string table (as
    used in the README) instead of a DataFrame.
    """
    # treat "unknown" as null for null-pct calculation
    attributes = attributes.with_columns(
        pl.when(pl.col(col) == "unknown")
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in attributes.columns
        if attributes[col].dtype == pl.String
    )

    attribute_counts = attributes.group_by("source").agg(n_attributes=pl.len())

    null_counts = group_null_pct(
        attributes,
        group_cols=["source"],
        ignore=["hid", "pid", "source"],
        return_per_column=False,
        return_overall=True,
    )

    trip_counts = (
        trips.join(attributes.select("pid", "source"), on="pid", how="left")
        .group_by("source")
        .agg(n_trips=pl.len())
    )

    distance_counts = (
        trips.join(attributes.select("pid", "source"), on="pid", how="left")
        .group_by("source")
        .agg(total_distance=pl.col("distance").sum() / 1000000)
    )

    attributes_summary = (
        attribute_counts.join(
            null_counts.select("source", "overall_null_pct"),
            on="source",
            how="left",
        )
        .join(trip_counts, on="source", how="left")
        .join(distance_counts, on="source", how="left")
        .fill_null(0)
        .rename(
            {
                "n_attributes": "persons",
                "n_trips": "trips",
                "overall_null_pct": "nulls",
                "total_distance": "kms (millions)",
            }
        )
        .sort("persons", descending=True)
    )

    total_nulls = sum(
        attributes[col].is_null().sum()
        for col in attributes.select(
            pl.all().exclude("hid", "pid", "source")
        ).columns
    )

    total_row = pl.DataFrame(
        {
            "source": ["total"],
            "persons": [attributes.height],
            "trips": [trips.height],
            "nulls": [
                total_nulls / (attributes.height * (attributes.width - 3)) * 100
            ],
            "kms (millions)": [
                trips.select(pl.col("distance").sum() / 1000000).item()
            ],
        },
        schema={
            "source": pl.String,
            "persons": pl.UInt32,
            "trips": pl.UInt32,
            "nulls": attributes_summary["nulls"].dtype,
            "kms (millions)": attributes_summary["kms (millions)"].dtype,
        },
    )

    table = pl.concat([attributes_summary, total_row], how="diagonal")

    if markdown:
        return _summary_table_to_markdown(table)

    return table


def _summary_table_to_markdown(table: pl.DataFrame) -> str:
    headers = [
        "Source",
        "Plans",
        "Missing attributes",
        "Trips",
        "Trip kms (millions)",
    ]
    rows = []
    for row in table.iter_rows(named=True):
        is_total = row["source"] == "total"
        cells = [
            row["source"],
            f"{row['persons']:,}",
            f"{row['nulls']:.0f}%",
            f"{row['trips']:,}",
            f"{row['kms (millions)']:.1f}",
        ]
        if is_total:
            cells = [f"**{cell}**" for cell in cells]
        rows.append(cells)
    return render_markdown_table(headers, rows)


def attribute_availability(
    attributes: pl.DataFrame, markdown: bool = False
) -> pl.DataFrame | str:
    """Per-attribute % available (not null or "unknown"), one column per source.

    Returns a table with one row per attribute and one column per source
    (plus an "all" column pooling across all sources), cell values are the
    % of records with that attribute available (i.e. not null or
    "unknown") in that source. Rows are sorted by the "all" column,
    highest availability first.
    """
    attributes = attributes.with_columns(
        pl.when(pl.col(col) == "unknown")
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in attributes.columns
        if attributes[col].dtype == pl.String
    )

    per_source = group_null_pct(
        attributes,
        group_cols=["source"],
        ignore=["hid", "pid", "source"],
        return_per_column=True,
        return_overall=False,
    )
    overall = group_null_pct(
        attributes.with_columns(pl.lit("all").alias("source")),
        group_cols=["source"],
        ignore=["hid", "pid", "source"],
        return_per_column=True,
        return_overall=False,
    )
    per_source = per_source.sort("source")
    per_source = pl.concat([per_source, overall], how="vertical")
    per_source = per_source.with_columns(
        (pl.lit(100) - pl.col(col)).alias(col)
        for col in per_source.columns
        if col != "source"
    )

    table = per_source.transpose(
        include_header=True, header_name="attribute", column_names="source"
    ).sort("all", descending=True)

    if markdown:
        return _attribute_availability_to_markdown(table)

    return table


def _attribute_availability_to_markdown(table: pl.DataFrame) -> str:
    headers = ["Attribute"] + table.columns[1:]
    rows = []
    for row in table.iter_rows():
        attribute, *pcts = row
        rows.append([attribute] + [f"{pct:.0f}%" for pct in pcts])
    return render_markdown_table(headers, rows)


def activity_summary_table(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
    on: str = "source",
    markdown: bool = False,
) -> pl.DataFrame | str:
    """Per-source, per-activity-type participation and typical duration.

    `activities` should come from `post_process.trips_to_activities`.
    Participation probability is the share of persons in each source group
    with at least one activity of that type (P(count >= 1)); rate is the
    expected number of that activity per person (mean count, including
    zeros). Duration stats are computed over all activities of that type
    (not per-person).
    """
    activities = activities.join(
        attributes.select("pid", on), on="pid", how="left"
    ).with_columns(duration=(pl.col("end") - pl.col("start")).cast(pl.Float64))

    n_persons = attributes.group_by(on).agg(n_persons=pl.len())

    summary = (
        activities.group_by([on, "act"])
        .agg(
            n_activities=pl.len(),
            n_participants=pl.col("pid").n_unique(),
            median_duration_min=pl.col("duration").median(),
            mean_duration_min=pl.col("duration").mean(),
        )
        .join(n_persons, on=on, how="left")
        .with_columns(
            (pl.col("n_participants") / pl.col("n_persons") * 100).alias(
                "participation_prob_pct"
            ),
            (pl.col("n_activities") / pl.col("n_persons") * 100).alias(
                "participation_rate_pct"
            ),
        )
        .drop("n_persons")
        .sort(["act", on])
    )

    if markdown:
        return _activity_summary_table_to_markdown(summary, on)
    return summary


def _activity_summary_table_to_markdown(table: pl.DataFrame, on: str) -> str:
    headers = [
        on,
        "participation prob %",
        "rate %",
        "median dur (min)",
        "mean dur (min)",
    ]

    blocks = []
    for act in table["act"].unique(maintain_order=False).sort().to_list():
        rows = []
        for row in table.filter(pl.col("act") == act).iter_rows(named=True):
            rows.append(
                [
                    str(row[on]),
                    f"{row['participation_prob_pct']:.1f}%",
                    f"{row['participation_rate_pct']:.1f}%",
                    (
                        f"{row['median_duration_min']:.1f}"
                        if row["median_duration_min"] is not None
                        else "n/a"
                    ),
                    (
                        f"{row['mean_duration_min']:.1f}"
                        if row["mean_duration_min"] is not None
                        else "n/a"
                    ),
                ]
            )
        blocks.append(f"**{act}**\n\n" + render_markdown_table(headers, rows))

    return "\n\n".join(blocks)
