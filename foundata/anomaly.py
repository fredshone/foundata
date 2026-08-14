import math
from typing import Optional

import numpy as np
import polars as pl

from foundata import post_process, tables


def _cramers_v(categories: np.ndarray, binary: np.ndarray) -> float:
    """Cramér's V between a categorical array and a binary (0/1) array.

    0 = no association, 1 = perfect association. NaN if there are fewer
    than 2 categories, no rows, or the binary array has no variation.
    """
    n = len(categories)
    if n == 0:
        return float("nan")
    cats = np.unique(categories)
    if len(cats) < 2 or len(np.unique(binary)) < 2:
        return float("nan")
    table = np.array(
        [
            [
                np.sum((categories == c) & (binary == 0)),
                np.sum((categories == c) & (binary == 1)),
            ]
            for c in cats
        ],
        dtype=float,
    )
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(
            np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
        )
    # df denominator min(rows-1, cols-1); cols is always 2 (binary), and the
    # `len(cats) < 2` guard above ensures rows-1 >= 1, so this is always 1 —
    # kept explicit for readability rather than hardcoding 1.
    k = min(len(cats) - 1, 1)
    return float(math.sqrt(chi2 / (n * k)))


def _as_group_cols(on: str | list[str]) -> list[str]:
    return [on] if isinstance(on, str) else list(on)


def _jensen_shannon(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence (log base 2, bounded [0, 1]) between two
    categorical distributions given as {category: probability} dicts.

    Categories present in one distribution but not the other are treated
    as probability 0 in the missing one rather than being dropped — a
    category a group never uses is itself part of how its distribution
    differs from the reference.
    """
    keys = set(p) | set(q)
    p_arr = np.array([p.get(k, 0.0) for k in keys])
    q_arr = np.array([q.get(k, 0.0) for k in keys])
    m_arr = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p_arr, m_arr) + 0.5 * _kl(q_arr, m_arr)


def _category_distribution(df: pl.DataFrame, col: str) -> tuple[dict, int]:
    """{category: share} distribution of non-null values in `col`, plus the
    non-null row count it was computed over."""
    non_null = df.select(col).drop_nulls()
    n = non_null.height
    if n == 0:
        return {}, 0
    counts = non_null.group_by(col).agg(pl.len().alias("n"))
    return dict(zip(counts[col].to_list(), (counts["n"] / n).to_list())), n


def _top_deviations(
    group_p: dict, reference_p: dict
) -> tuple[str, float, str, float]:
    """The category most over- and under-represented in `group_p` relative
    to `reference_p`, as (category, group_p - reference_p) pairs."""
    keys = set(group_p) | set(reference_p)
    deltas = {k: group_p.get(k, 0.0) - reference_p.get(k, 0.0) for k in keys}
    over_cat = max(deltas, key=lambda k: deltas[k])
    under_cat = min(deltas, key=lambda k: deltas[k])
    return over_cat, deltas[over_cat], under_cat, deltas[under_cat]


# Attributes checked by `conditionality_matrix` by default: every
# person/household attribute that plausibly has *some* real relationship to
# activity participation, i.e. excluding ids (pid/hid/source/country),
# purely administrative fields (year), weather, and fields derived directly
# from the trips themselves (weight, avg_speed, access_egress_distance) —
# correlating those against activity participation would be close to
# tautological rather than a useful conditionality check.
DEFAULT_CONDITIONALITY_ATTRS = [
    "age",
    "sex",
    "employment",
    "employed_type",
    "education",
    "occupation",
    "disability",
    "can_wfh",
    "race",
    "relationship",
    "dwelling",
    "ownership",
    "hh_zone",
    "has_licence",
    "day",
    "hh_size",
    "hh_income",
    "vehicles",
]


def conditionality_matrix(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
    on: str | list[str] = "source",
    attribute_cols: Optional[list[str]] = None,
    act_types: Optional[list[str]] = None,
    n_bins: int = 5,
    min_group_n: int = 30,
) -> pl.DataFrame:
    """Cramér's V between every candidate attribute and every activity-type
    participation indicator (has >=1 activity of that type), per group.

    Generalises the old two-pair check (employment-vs-work,
    age-vs-education) to every attribute in `attribute_cols` (default
    `DEFAULT_CONDITIONALITY_ATTRS`) against every activity type present in
    the data — the number of genuinely-plausible conditional relationships
    in a household travel survey is much larger than two, and a
    source-specific bug can hide in any of them (e.g. `day` vs `work`
    should show weekday >> weekend for most sources).

    `on` is usually "source", but can also be a list of columns (e.g.
    `["source", "year"]`) to check at a finer granularity — a common
    real-world failure is a per-year mapping change (documented in a
    source's config, applied via `utils.config_for_year`) that only
    miscodes one year of an otherwise-good source; grouping by source alone
    would average that year away against its own source's other years.

    Numeric attributes are binned before computing association: `age` uses
    the fixed bands from `post_process.add_age_band` (see its docstring for
    why fixed bands matter); other numeric columns (`hh_size`, `hh_income`,
    `vehicles`) use `post_process.discretise_numeric` with quantile bins
    computed globally across all sources at once (not per source) — so, as
    with age, every source is measured against the same cut points rather
    than being invisibly re-smoothed by its own bins.

    Returns a long-format table: one row per (`*on`, attribute, act_type)
    with the group size (`n`, non-null rows the score was computed on) and
    `cramers_v`. Groups with fewer than `min_group_n` non-null rows are
    skipped. Feed this into `flag_conditionality_outliers` to surface the
    pairs where one group's score is unusual relative to its peers, rather
    than eyeballing the full matrix.
    """
    group_cols = _as_group_cols(on)

    if attribute_cols is None:
        attribute_cols = DEFAULT_CONDITIONALITY_ATTRS
    attribute_cols = [c for c in attribute_cols if c in attributes.columns]

    numeric_cols = [
        c
        for c in attribute_cols
        if c != "age" and attributes[c].dtype in pl.NUMERIC_DTYPES
    ]

    binned = attributes
    if "age" in attribute_cols:
        binned = post_process.add_age_band(binned, age_col="age", out_col="age")
    if numeric_cols:
        binned = post_process.discretise_numeric(
            binned, n_bins=n_bins, method="quantile", cols=numeric_cols
        )

    if act_types is None:
        act_types = sorted(
            set(activities.select("act").drop_nulls().unique().to_series())
            - {"home", "unknown"}
        )

    join_cols = [c for c in group_cols if c not in attribute_cols]
    counts = post_process.activity_counts_per_person(
        attributes, activities, act_types
    )
    counts = counts.join(
        binned.select("pid", *join_cols, *attribute_cols), on="pid", how="left"
    )
    counts = counts.with_columns(
        [(pl.col(t) > 0).cast(pl.Int8).alias(f"has_{t}") for t in act_types]
    )

    groups = (
        counts.select(group_cols).drop_nulls().unique().sort(group_cols).rows()
    )

    rows = []
    for g in groups:
        filter_expr = pl.all_horizontal(
            [pl.col(c) == v for c, v in zip(group_cols, g)]
        )
        sub = counts.filter(filter_expr)
        act_arrays = {t: sub[f"has_{t}"].to_numpy() for t in act_types}
        for attr in attribute_cols:
            valid_mask = sub[attr].is_not_null().to_numpy()
            n = int(valid_mask.sum())
            if n < min_group_n:
                continue
            attr_arr = sub[attr].to_numpy()[valid_mask]
            for t in act_types:
                v = _cramers_v(attr_arr, act_arrays[t][valid_mask])
                rows.append(
                    {
                        **dict(zip(group_cols, g)),
                        "attribute": attr,
                        "act_type": t,
                        "n": n,
                        "cramers_v": v,
                    }
                )

    return pl.DataFrame(rows)


def flag_conditionality_outliers(
    matrix: pl.DataFrame,
    on: str | list[str] = "source",
    z_threshold: float = 1.5,
    min_peers: int = 3,
    top_n: int = 20,
    markdown: bool = False,
) -> pl.DataFrame | str:
    """Flag (`*on`, attribute, act_type) cells whose Cramér's V is unusual
    relative to its peers' scores for the same (attribute, act_type) pair.

    For each (attribute, act_type) pair, computes a robust z-score for each
    group's `cramers_v` against its peers' scores for that same pair
    (median and MAD-based scale, robust to one bad group dragging the
    mean/std). Rows with |z| >= `z_threshold` are kept, sorted by |z|
    descending (most anomalous first) and capped to the `top_n` most
    extreme cells — an unusually *weak* relationship relative to peers
    (the "my model can't learn this" case) and an unusually *strong* one
    (a possible leakage/miscoding bug) are both "strange" and can appear
    anywhere in the ranking. Pairs with fewer than `min_peers` groups
    scored are skipped — not enough peers to judge "unusual" against.

    `on` must match whatever `on` was passed to `conditionality_matrix`
    (e.g. `["source", "year"]` to flag individual source-years against all
    other source-years, catching a per-year mapping bug that a
    source-level-only check would average away).
    """
    group_cols = _as_group_cols(on)

    valid = matrix.filter(
        pl.col("cramers_v").is_not_null() & pl.col("cramers_v").is_finite()
    )

    valid = valid.with_columns(
        peer_median=pl.col("cramers_v")
        .median()
        .over(["attribute", "act_type"]),
        n_sources=pl.len().over(["attribute", "act_type"]),
    )
    valid = valid.with_columns(
        peer_mad=(pl.col("cramers_v") - pl.col("peer_median"))
        .abs()
        .median()
        .over(["attribute", "act_type"])
    )
    valid = valid.with_columns(
        z=pl.when(pl.col("peer_mad") > 0)
        .then(
            (pl.col("cramers_v") - pl.col("peer_median"))
            / (1.4826 * pl.col("peer_mad"))
        )
        .otherwise(None)
    )

    flagged = (
        valid.filter(pl.col("n_sources") >= min_peers)
        .filter(pl.col("z").is_not_null())
        .filter(pl.col("z").abs() >= z_threshold)
        .select(
            *group_cols,
            "attribute",
            "act_type",
            "n",
            "cramers_v",
            "peer_median",
            "z",
        )
        .sort(pl.col("z").abs(), descending=True)
        .head(top_n)
    )

    if markdown:
        return _conditionality_outliers_to_markdown(flagged, group_cols)
    return flagged


def _conditionality_outliers_to_markdown(
    table: pl.DataFrame, on: str | list[str]
) -> str:
    group_cols = _as_group_cols(on)
    if table.is_empty():
        return "No anomalies flagged."

    headers = [
        *group_cols,
        "attribute",
        "activity",
        "n",
        "V",
        "peer median V",
        "z",
    ]
    rows = []
    for row in table.iter_rows(named=True):
        rows.append(
            [
                *[str(row[c]) for c in group_cols],
                row["attribute"],
                row["act_type"],
                f"{row['n']:,}",
                f"{row['cramers_v']:.2f}",
                f"{row['peer_median']:.2f}",
                f"{row['z']:+.1f}",
            ]
        )
    return tables.render_markdown_table(headers, rows)


def attribute_distribution_shift_matrix(
    attributes: pl.DataFrame,
    on: str | list[str] = "source",
    attribute_cols: Optional[list[str]] = None,
    n_bins: int = 5,
    min_group_n: int = 30,
) -> pl.DataFrame:
    """Jensen-Shannon divergence between each group's attribute distribution
    and the overall distribution of that attribute (all rows pooled, i.e.
    every source and year at once).

    Complements `conditionality_matrix`: that checks whether an attribute
    still *relates* to activity participation the way it should; this
    checks whether the attribute's own marginal distribution — e.g. the mix
    of `employment` categories, or the shape of the `age` histogram — looks
    like the pooled average, or whether one group has quietly drifted (a
    recoded category, a sampling skew, a unit change).

    `on` is usually `["source", "year"]` so a single bad year within an
    otherwise-normal source is caught rather than averaged into that
    source's other years; pass `"source"` alone to compare whole sources
    instead.

    Numeric attributes are binned the same way as in `conditionality_matrix`
    (see its docstring): `age` uses the fixed bands from
    `post_process.add_age_band`; other numeric columns use
    `post_process.discretise_numeric` with bins computed globally across all
    rows, so every group is compared on the same categories rather than its
    own re-binned scale.

    Returns a long-format table: one row per (`*on`, attribute) with the
    group size (`n`, non-null rows the distribution was computed over),
    `jsd` (Jensen-Shannon divergence, log base 2; 0 = identical to the
    pooled distribution, 1 = disjoint support), and the single most
    over-/under-represented category in that group relative to the pooled
    distribution (`top_over_category`/`top_over_delta_pct`,
    `top_under_category`/`top_under_delta_pct` — deltas are percentage
    points of share). Groups with fewer than `min_group_n` non-null values
    are skipped. Feed this into `flag_distribution_shift_outliers` to
    surface the most unusual groups rather than eyeballing the full matrix.
    """
    group_cols = _as_group_cols(on)

    if attribute_cols is None:
        attribute_cols = DEFAULT_CONDITIONALITY_ATTRS
    attribute_cols = [c for c in attribute_cols if c in attributes.columns]

    numeric_cols = [
        c
        for c in attribute_cols
        if c != "age" and attributes[c].dtype in pl.NUMERIC_DTYPES
    ]

    binned = attributes
    if "age" in attribute_cols:
        binned = post_process.add_age_band(binned, age_col="age", out_col="age")
    if numeric_cols:
        binned = post_process.discretise_numeric(
            binned, n_bins=n_bins, method="quantile", cols=numeric_cols
        )

    rows = []
    for attr in attribute_cols:
        overall_p, n_overall = _category_distribution(binned, attr)
        if n_overall == 0:
            continue

        groups = (
            binned.select(*group_cols, attr)
            .drop_nulls()
            .select(group_cols)
            .unique()
            .sort(group_cols)
            .rows()
        )
        for g in groups:
            filter_expr = pl.all_horizontal(
                [pl.col(c) == v for c, v in zip(group_cols, g)]
            )
            sub = binned.filter(filter_expr)
            group_p, n = _category_distribution(sub, attr)
            if n < min_group_n:
                continue
            jsd = _jensen_shannon(group_p, overall_p)
            over_cat, over_delta, under_cat, under_delta = _top_deviations(
                group_p, overall_p
            )
            rows.append(
                {
                    **dict(zip(group_cols, g)),
                    "attribute": attr,
                    "n": n,
                    "jsd": jsd,
                    "top_over_category": over_cat,
                    "top_over_delta_pct": over_delta * 100,
                    "top_under_category": under_cat,
                    "top_under_delta_pct": under_delta * 100,
                }
            )

    return pl.DataFrame(rows)


def activity_distribution_shift_matrix(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
    on: str | list[str] = "source",
    min_group_n: int = 30,
) -> pl.DataFrame:
    """Jensen-Shannon divergence between each group's activity-purpose mix
    and the overall mix (all activities pooled, i.e. every source and year
    at once).

    `distribution_shift_matrix` checks whether a *person attribute*'s
    marginal distribution looks like the pooled average; this does the same
    thing for `activities` (from `post_process.trips_to_activities`) — the
    share of `work` vs `education` vs `shop` vs ... activities a group's
    persons report. A source-year whose activity mix has drifted (a purpose
    miscoded, an activity type dropped or over-generated, a sampling skew)
    shows up as high JSD here, replacing the need to eyeball a raw
    participation-by-source table.

    `on` is usually `["source", "year"]` so a single bad year within an
    otherwise-normal source is caught rather than averaged into that
    source's other years; pass `"source"` alone to compare whole sources
    instead.

    Returns a long-format table shaped like `distribution_shift_matrix`'s
    output — one row per `*on` group, `attribute` fixed to `"act"`, `n`
    (non-null activities the distribution was computed over), `jsd`, and
    the single most over-/under-represented activity type in that group
    relative to the pooled mix — so it can be fed into
    `flag_distribution_shift_outliers` unchanged. Groups with fewer than
    `min_group_n` activities are skipped.
    """
    group_cols = _as_group_cols(on)
    activities = activities.join(
        attributes.select("pid", *group_cols), on="pid", how="left"
    )

    overall_p, n_overall = _category_distribution(activities, "act")
    if n_overall == 0:
        return pl.DataFrame([])

    groups = (
        activities.select(*group_cols, "act")
        .drop_nulls()
        .select(group_cols)
        .unique()
        .sort(group_cols)
        .rows()
    )

    rows = []
    for g in groups:
        filter_expr = pl.all_horizontal(
            [pl.col(c) == v for c, v in zip(group_cols, g)]
        )
        sub = activities.filter(filter_expr)
        group_p, n = _category_distribution(sub, "act")
        if n < min_group_n:
            continue
        jsd = _jensen_shannon(group_p, overall_p)
        over_cat, over_delta, under_cat, under_delta = _top_deviations(
            group_p, overall_p
        )
        rows.append(
            {
                **dict(zip(group_cols, g)),
                "attribute": "act",
                "n": n,
                "jsd": jsd,
                "top_over_category": over_cat,
                "top_over_delta_pct": over_delta * 100,
                "top_under_category": under_cat,
                "top_under_delta_pct": under_delta * 100,
            }
        )

    return pl.DataFrame(rows)


def flag_distribution_shift_outliers(
    matrix: pl.DataFrame,
    on: str | list[str] = "source",
    top_n: int = 10,
    markdown: bool = False,
) -> pl.DataFrame | str:
    """The `top_n` (`*on`, attribute) rows with the highest Jensen-Shannon
    divergence from the pooled distribution — the attribute distributions
    that look least like the rest of the data.

    `on` must match whatever `on` was passed to `distribution_shift_matrix`.
    """
    group_cols = _as_group_cols(on)

    valid = matrix.filter(
        pl.col("jsd").is_not_null() & pl.col("jsd").is_finite()
    )
    top = valid.sort("jsd", descending=True).head(top_n)

    if markdown:
        return _distribution_shift_outliers_to_markdown(top, group_cols)
    return top


def _distribution_shift_outliers_to_markdown(
    table: pl.DataFrame, on: str | list[str]
) -> str:
    group_cols = _as_group_cols(on)
    if table.is_empty():
        return "No anomalies flagged."

    headers = [
        *group_cols,
        "attribute",
        "n",
        "JSD",
        "over-represented",
        "under-represented",
    ]
    rows = []
    for row in table.iter_rows(named=True):
        rows.append(
            [
                *[str(row[c]) for c in group_cols],
                row["attribute"],
                f"{row['n']:,}",
                f"{row['jsd']:.2f}",
                f"{row['top_over_category']} ({row['top_over_delta_pct']:+.0f}pp)",
                f"{row['top_under_category']} ({row['top_under_delta_pct']:+.0f}pp)",
            ]
        )
    return tables.render_markdown_table(headers, rows)


def time_quality_summary_table(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    max_plausible_speed: float = 150.0,
    markdown: bool = False,
) -> pl.DataFrame | str:
    """Per-source trip-time quality diagnostics.

    Flags non-positive-duration trips (tst >= tet), day-wrap trips
    (tst/tet > 1440), and implausibly fast trips (implied speed above
    `max_plausible_speed` km/h) — the signatures of a per-source
    time-encoding bug rather than an exhaustive quality check.
    """
    trips = trips.join(
        attributes.select("pid", on), on="pid", how="left"
    ).with_columns(duration=(pl.col("tet") - pl.col("tst")).cast(pl.Float64))

    speed_expr = (
        pl.when((pl.col("duration") > 0) & pl.col("distance").is_not_null())
        .then(pl.col("distance") / (pl.col("duration") / 60))
        .otherwise(None)
    )

    summary = (
        trips.with_columns(speed=speed_expr)
        .group_by(on)
        .agg(
            n_trips=pl.len(),
            non_positive_duration_pct=(pl.col("duration") <= 0).mean() * 100,
            day_wrap_pct=(
                (pl.col("tst") > 1440) | (pl.col("tet") > 1440)
            ).mean()
            * 100,
            median_duration_min=pl.col("duration")
            .filter(pl.col("duration") > 0)
            .median(),
            implausible_speed_pct=(pl.col("speed") > max_plausible_speed).mean()
            * 100,
            median_speed_kmh=pl.col("speed").median(),
        )
        .sort(on)
    )

    if markdown:
        return _time_quality_table_to_markdown(summary)
    return summary


def _time_quality_table_to_markdown(table: pl.DataFrame) -> str:
    headers = [
        "source",
        "trips",
        "non-positive dur %",
        "day-wrap %",
        "median dur (min)",
        "implausible speed %",
        "median speed (km/h)",
    ]
    rows = []
    for row in table.iter_rows(named=True):
        rows.append(
            [
                str(row[table.columns[0]]),
                f"{row['n_trips']:,}",
                f"{row['non_positive_duration_pct']:.1f}%",
                f"{row['day_wrap_pct']:.1f}%",
                (
                    f"{row['median_duration_min']:.1f}"
                    if row["median_duration_min"] is not None
                    else "n/a"
                ),
                f"{row['implausible_speed_pct']:.1f}%",
                (
                    f"{row['median_speed_kmh']:.1f}"
                    if row["median_speed_kmh"] is not None
                    else "n/a"
                ),
            ]
        )
    return tables.render_markdown_table(headers, rows)
