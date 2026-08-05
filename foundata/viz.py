import math
from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from foundata import post_process


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
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" * len(h) for h in headers) + "|",
    ]
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
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_numeric_hist_grid(
    df: pl.DataFrame,
    on: str = "source",
    n_cols: int = 3,
    max_sample: int = 10_000,
    bins="auto",
    density: bool = True,
    cmap_name: str = "tab20",
    linewidth: float = 3,
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    ignore_cols: list | set | tuple = (),
    min_unique: int = 5,
    min_group_rows: int = 10,
    tail_handling: str | None = "ignore",
    tail_ratio_threshold: float = 4.0,
    outlier_share_max: float = 0.05,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    verbose: bool = True,
    save_path: str | Path | None = None,
):
    ignore_cols = set(ignore_cols) if ignore_cols else set()
    num_cols = [
        c
        for c, dt in zip(df.columns, df.dtypes)
        if dt in pl.NUMERIC_DTYPES and c != on and c not in ignore_cols
    ]

    if len(num_cols) == 0:
        raise ValueError(
            "No numeric columns to plot after applying ignore_cols."
        )

    if df.height > max_sample:
        df_plot = df.sample(n=max_sample, shuffle=True)
    else:
        df_plot = df

    groups = (
        df_plot.select(pl.col(on)).drop_nulls().unique().to_series().to_list()
    )
    groups = sorted(groups)
    if len(groups) == 0:
        raise ValueError(f"No non-null groups found in '{on}'.")

    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        palette = list(cmap.colors)
    else:
        palette = [cmap(i / 20) for i in range(20)]
    color_cycle = cycle(palette)
    color_map = {g: next(color_cycle) for g in groups}

    def _finite_np(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        return values[np.isfinite(values)]

    def _percentiles(vals: np.ndarray, ps=(1, 5, 25, 50, 75, 95, 99)):
        return np.percentile(vals, ps)

    def _assess_long_tail(vals: np.ndarray):
        p1, p5, p25, p50, p75, p95, p99 = _percentiles(vals)
        iqr = p75 - p25
        eps = 1e-12
        if iqr < eps:
            return {
                "flag": True,
                "reason": "near-constant (IQR≈0)",
                "upper_tail_ratio": np.nan,
                "lower_tail_ratio": np.nan,
                "outlier_share": np.nan,
                "iqr": iqr,
                "clip_bounds": (p1, p99),
            }

        upper_tail_ratio = (p99 - p95) / (iqr + eps)
        lower_tail_ratio = (p5 - p1) / (iqr + eps)

        upper_fence = p75 + 3 * iqr
        lower_fence = p25 - 3 * iqr
        outlier_share = np.mean((vals > upper_fence) | (vals < lower_fence))

        flag = (
            max(upper_tail_ratio, lower_tail_ratio) >= tail_ratio_threshold
            and outlier_share <= outlier_share_max
        )

        return {
            "flag": flag,
            "reason": (
                f"long-tail (ratios: up={upper_tail_ratio:.2f}, "
                f"low={lower_tail_ratio:.2f}, outlier_share={outlier_share:.3f})"
            ),
            "upper_tail_ratio": upper_tail_ratio,
            "lower_tail_ratio": lower_tail_ratio,
            "outlier_share": outlier_share,
            "iqr": iqr,
            "clip_bounds": (p1, p99),
        }

    to_plot = []
    skip_reasons = {}
    col_tail_info = {}

    for col in num_cols:
        all_vals = (
            df_plot.select(pl.col(col)).drop_nulls().to_series().to_numpy()
        )
        all_vals = _finite_np(all_vals)

        if len(np.unique(all_vals)) < min_unique:
            skip_reasons[col] = f"insufficient unique values (<{min_unique})"
            continue

        if all_vals.size < 2 or np.nanstd(all_vals) == 0:
            skip_reasons[col] = "insufficient variation"
            continue

        info = _assess_long_tail(all_vals)
        col_tail_info[col] = info

        if tail_handling == "ignore" and info["flag"]:
            skip_reasons[col] = info["reason"]
            continue

        to_plot.append(col)

    if len(to_plot) == 0:
        message = "All candidate numeric columns were filtered out."
        if verbose:
            lines = [message, "Reasons:"]
            for k, v in skip_reasons.items():
                lines.append(f" - {k}: {v}")
            print("\n".join(lines))
        raise ValueError(
            message + " Check thresholds or disable tail_handling."
        )

    n_plots = len(to_plot) + 1  # +1 for legend
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    for idx, col in enumerate(sorted(to_plot)):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        ax.set_facecolor(ax_bg)

        all_vals = (
            df_plot.select(pl.col(col)).drop_nulls().to_series().to_numpy()
        )
        all_vals = _finite_np(all_vals)

        if tail_handling == "clip" and col_tail_info.get(col, {}).get(
            "flag", False
        ):
            lo_p, hi_p = clip_percentiles
            lo, hi = np.percentile(all_vals, [lo_p, hi_p])
            all_vals = np.clip(all_vals, lo, hi)

        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

        for g in groups:
            sub = df_plot.filter(pl.col(on) == g)
            vals = sub.select(pl.col(col)).drop_nulls().to_series().to_numpy()
            vals = _finite_np(vals)

            if vals.size < min_group_rows:
                continue

            if tail_handling == "clip" and col_tail_info.get(col, {}).get(
                "flag", False
            ):
                lo_p, hi_p = clip_percentiles
                lo, hi = np.percentile(all_vals, [lo_p, hi_p])
                vals = np.clip(vals, lo, hi)

            ax.hist(
                vals,
                bins=bin_edges,
                histtype="step",
                density=density,
                color=color_map[g],
                linewidth=linewidth,
                label=str(g),
            )

        ax.set_title(col.title(), fontsize="large")
        ax.fontsize = "large"

    handles = [
        Line2D([0], [0], color=color_map[g], lw=linewidth, label=str(g))
        for g in groups
    ]

    # repeat for empty legend plot
    idx = len(to_plot)
    r = idx // n_cols
    c = idx % n_cols
    ax = axes[r][c]
    ax.set_facecolor(ax_bg)

    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title=None,
        fontsize="large",
    )

    for i in range(n_plots - 1, n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        axes[r][c].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _build_color_map_for_column(
    df: pl.DataFrame, col: str, cmap_name: str = "tab20"
):
    uniques = df.select(pl.col(col)).drop_nulls().unique().to_series().to_list()
    non_unknown = sorted(v for v in uniques if str(v).lower() != "unknown")

    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        palette = list(cmap.colors)
    else:
        palette = [cmap(i / 20) for i in range(20)]

    color_cycle = cycle(palette)
    return {v: next(color_cycle) for v in non_unknown}


def _plot_categorical_column(ax, df: pl.DataFrame, col: str, on: str):
    groups = df.select(pl.col(on)).drop_nulls().unique().to_series().to_list()
    groups = sorted(groups)

    color_map = _build_color_map_for_column(df, col)

    for i, g in enumerate(groups):
        sub = df.filter(pl.col(on) == g)
        proportions = (
            sub[col]
            .value_counts(normalize=True)
            .sort("proportion", descending=True)
        )

        unknown = proportions.filter(
            pl.col(col).cast(pl.Utf8).str.to_lowercase() == "unknown"
        )
        other = proportions.filter(
            pl.col(col).cast(pl.Utf8).str.to_lowercase() != "unknown"
        )

        proportions = other.vstack(unknown)

        left = 0.0
        for val, prop in proportions.select(col, "proportion").iter_rows():
            if val is None:
                continue

            color = color_map.get(val, "#999999")

            ax.barh(
                y=i,
                width=float(prop),
                left=left,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            left += float(prop)

    ax.set_title(col.title())
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xticks([])
    ax.set_xlim(0, 1)

    legend_items = []
    for i, (cat, color) in enumerate(color_map.items()):
        label = str(cat)
        label_length = min(12, len(label))
        legend_items.append(
            Patch(
                facecolor=color, edgecolor="black", label=label[:label_length]
            )
        )
        if i >= 13:
            break

    ax.legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        fontsize="x-small",
        facecolor="none",
        edgecolor="none",
    )

    ax.set_facecolor("lightgray")


def plot_summary_trends(
    df: pl.DataFrame,
    on: str = "source",
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    save_path: str | Path | None = None,
):
    MONTH_LABELS = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    groups = sorted(
        df.select(pl.col(on)).drop_nulls().unique().to_series().to_list()
    )
    cmap = plt.get_cmap(cmap_name)
    palette = (
        list(cmap.colors)
        if hasattr(cmap, "colors") and cmap.colors is not None
        else [cmap(i / 20) for i in range(20)]
    )
    color_map = {g: c for g, c in zip(groups, cycle(palette))}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
    fig.patch.set_facecolor(fig_bg)
    for ax in axes:
        ax.set_facecolor(ax_bg)

    # Panel 1 — avg max_temp by month
    ax = axes[0]
    for g in groups:
        sub = df.filter(pl.col(on) == g).filter(
            pl.col("month").is_not_null() & pl.col("max_temp_c").is_not_null()
        )
        if sub.is_empty():
            continue
        agg = (
            sub.group_by("month").agg(pl.col("max_temp_c").mean()).sort("month")
        )
        ax.plot(
            agg["month"].to_list(),
            agg["max_temp_c"].to_list(),
            color=color_map[g],
            label=g,
            linewidth=2,
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
    ax.set_title("Avg Max Temp by Month", fontsize="large")
    ax.set_ylabel("max_temp_c")

    # Panel 2 — avg hh_income by race
    ax = axes[1]
    races = sorted(
        df.select(pl.col("race")).drop_nulls().unique().to_series().to_list()
    )
    for g in groups:
        sub = df.filter(pl.col(on) == g).filter(
            pl.col("race").is_not_null() & pl.col("hh_income").is_not_null()
        )
        if sub.is_empty():
            continue
        agg = sub.group_by("race").agg(pl.col("hh_income").mean())
        race_list = agg["race"].to_list()
        x_positions = [races.index(r) for r in race_list if r in races]
        y_values = [v / 1000 for v in agg["hh_income"].to_list()]
        ax.plot(
            x_positions,
            y_values,
            color=color_map[g],
            label=g,
            marker="_",
            markersize=16,
            linewidth=0,
            markeredgewidth=4,
        )
    ax.set_xticks(range(len(races)))
    ax.set_xticklabels(races, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.set_title("Avg HH Income by Race", fontsize="large")
    ax.set_ylabel("hh_income")

    # Panel 3 — avg avg_speed by year
    ax = axes[2]
    for g in groups:
        sub = df.filter(pl.col(on) == g).filter(
            pl.col("year").is_not_null() & pl.col("avg_speed").is_not_null()
        )
        if sub.is_empty():
            continue
        agg = sub.group_by("year").agg(pl.col("avg_speed").mean()).sort("year")
        xs = agg["year"].to_list()
        ys = agg["avg_speed"].to_list()
        if len(xs) == 1:
            ax.scatter(xs, ys, color=color_map[g], label=g, s=80, zorder=3)
        else:
            ax.plot(xs, ys, color=color_map[g], label=g, linewidth=2)
    ax.set_title("Avg Speed by Year", fontsize="large")
    ax.set_ylabel("avg_speed")
    ax.set_xlabel("year")

    handles = [
        Line2D([0], [0], color=color_map[g], lw=2, label=g) for g in groups
    ]
    axes[2].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        frameon=False,
        fontsize="large",
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_categorical_bar_grid(
    df: pl.DataFrame,
    on: str = "source",
    n_cols: int = 3,
    max_sample: int = 10_000,
    cmap_name: str = "tab20",
    ignore_cols: set | None = None,
    save_path: str | Path | None = None,
):
    if ignore_cols is None:
        ignore_cols = {"source", "pid", "hid", "country"}

    cat_cols = {
        col
        for col in df.columns
        if df[col].dtype in [pl.String, pl.Categorical, pl.Boolean]
    } - ignore_cols

    n_plots = len(cat_cols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.0 * n_rows), squeeze=False
    )

    if df.height > max_sample:
        df_plot = df.sample(n=max_sample, shuffle=True)
    else:
        df_plot = df

    fig.patch.set_facecolor("lightgray")

    for idx, col in enumerate(sorted(cat_cols)):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        _plot_categorical_column(ax, df_plot, col, on=on)

    for i in range(n_plots, n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        axes[r][c].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _group_color_map(groups: list, cmap_name: str = "tab20") -> dict:
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        palette = list(cmap.colors)
    else:
        palette = [cmap(i / 20) for i in range(20)]
    color_cycle = cycle(palette)
    return {g: next(color_cycle) for g in groups}


def _non_null_groups(df: pl.DataFrame, on: str) -> list:
    groups = sorted(
        df.select(pl.col(on)).drop_nulls().unique().to_series().to_list()
    )
    if not groups:
        raise ValueError(f"No non-null groups found in '{on}'.")
    return groups


def plot_time_of_day_profile(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    linewidth: float = 2.5,
    save_path: str | Path | None = None,
):
    """Departure/arrival time-of-day density per source.

    tst/tet are wrapped modulo 1440 (minutes/day) onto a 0-24h axis so
    day-crossing trips still land in a sensible hour-of-day bucket. The
    legend annotates each source's share of trips with tst/tet > 1440
    (uncorrected day-wrap) — a source with a much larger share than the
    others is a strong signal of a time-encoding bug rather than genuine
    late-night travel.
    """
    trips = trips.join(attributes.select("pid", on), on="pid", how="left")
    groups = _non_null_groups(trips, on)
    color_map = _group_color_map(groups, cmap_name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(fig_bg)
    for ax in axes:
        ax.set_facecolor(ax_bg)

    bin_edges = np.linspace(0, 24, 49)  # half-hour bins
    wrap_shares = {}

    for panel, (col, title) in enumerate(
        [("tst", "Departure Time of Day"), ("tet", "Arrival Time of Day")]
    ):
        ax = axes[panel]
        for g in groups:
            vals = (
                trips.filter(pl.col(on) == g)
                .select(col)
                .drop_nulls()
                .to_series()
                .to_numpy()
            )
            if vals.size == 0:
                continue
            if panel == 0:
                wrap_shares[g] = float(np.mean(vals > 1440))
            hours = np.mod(vals, 1440) / 60.0
            ax.hist(
                hours,
                bins=bin_edges,
                histtype="step",
                density=True,
                color=color_map[g],
                linewidth=linewidth,
                label=str(g),
            )
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day")
        ax.set_title(title, fontsize="large")

    axes[0].set_ylabel("density")

    handles = [
        Line2D(
            [0],
            [0],
            color=color_map[g],
            lw=linewidth,
            label=f"{g} ({wrap_shares.get(g, 0.0) * 100:.1f}% day-wrap)",
        )
        for g in groups
    ]
    axes[1].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=False,
        fontsize="medium",
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_time_heaping(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    time_col: str = "tst",
    n_cols: int = 3,
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    save_path: str | Path | None = None,
):
    """Minute-within-hour heaping per source.

    Self-reported travel times tend to "heap" at round numbers (:00, :15,
    :30, :45). A source with disproportionate heaping relative to the
    others is a classic marker of imprecise or reconstructed times.
    """
    trips = trips.join(attributes.select("pid", on), on="pid", how="left")
    groups = _non_null_groups(trips, on)
    color_map = _group_color_map(groups, cmap_name)

    n_rows = math.ceil(len(groups) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    for idx, g in enumerate(groups):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        ax.set_facecolor(ax_bg)

        minutes = (
            trips.filter(pl.col(on) == g)
            .select(time_col)
            .drop_nulls()
            .to_series()
            .to_numpy()
        )
        minutes = np.mod(minutes, 60).astype(int)
        if minutes.size == 0:
            ax.axis("off")
            continue

        counts = np.bincount(minutes, minlength=60)
        proportions = counts / counts.sum()

        ax.bar(range(60), proportions, color=color_map[g], width=1.0)
        round_share = proportions[[0, 15, 30, 45]].sum() * 100
        ax.set_title(
            f"{g} ({round_share:.0f}% on :00/:15/:30/:45)", fontsize="medium"
        )
        ax.set_xlim(-0.5, 59.5)
        ax.set_xticks([0, 15, 30, 45])
        ax.set_xlabel("minute of hour")

    for i in range(len(groups), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")

    fig.suptitle(f"Time Heaping ({time_col})", fontsize="large")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_trip_time_diagnostics(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    linewidth: float = 2.5,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    save_path: str | Path | None = None,
):
    """Trip duration, implied speed, and non-positive-duration share per source.

    Implied speed is distance / ((tet - tst) / 60), restricted to trips
    with positive duration and non-null distance (same guard as
    `utils.compute_avg_speed`). A source with speed mass piled up near 0 or
    stretching to implausible values usually means a duration or distance
    unit/encoding bug rather than genuinely different travel behaviour.
    """
    trips = trips.join(
        attributes.select("pid", on), on="pid", how="left"
    ).with_columns(duration=(pl.col("tet") - pl.col("tst")).cast(pl.Float64))
    groups = _non_null_groups(trips, on)
    color_map = _group_color_map(groups, cmap_name)

    valid = trips.filter(pl.col("duration") > 0)
    speeds = valid.filter(pl.col("distance").is_not_null()).with_columns(
        speed=pl.col("distance") / (pl.col("duration") / 60)
    )

    negative_share = trips.group_by(on).agg(
        share=(pl.col("duration") <= 0).mean() * 100
    )
    negative_map = dict(
        zip(negative_share[on].to_list(), negative_share["share"].to_list())
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(fig_bg)
    for ax in axes:
        ax.set_facecolor(ax_bg)

    lo_p, hi_p = clip_percentiles

    # Panel 1: trip duration
    ax = axes[0]
    all_durations = valid.select("duration").to_series().to_numpy()
    hi = (
        float(np.percentile(all_durations, hi_p)) if all_durations.size else 1.0
    )
    bin_edges = np.linspace(0, max(hi, 1.0), 40)
    for g in groups:
        vals = (
            valid.filter(pl.col(on) == g)
            .select("duration")
            .to_series()
            .to_numpy()
        )
        if vals.size == 0:
            continue
        vals = np.clip(vals, 0, hi)
        ax.hist(
            vals,
            bins=bin_edges,
            histtype="step",
            density=True,
            color=color_map[g],
            linewidth=linewidth,
            label=str(g),
        )
    ax.set_title("Trip Duration (min)", fontsize="large")
    ax.set_xlabel("minutes")
    ax.set_ylabel("density")

    # Panel 2: implied speed
    ax = axes[1]
    all_speeds = speeds.select("speed").to_series().to_numpy()
    if all_speeds.size:
        lo_s, hi_s = np.percentile(all_speeds, [lo_p, hi_p])
    else:
        lo_s, hi_s = 0.0, 1.0
    lo_s = max(lo_s, 0.0)
    bin_edges = np.linspace(lo_s, max(hi_s, lo_s + 1.0), 40)
    for g in groups:
        vals = (
            speeds.filter(pl.col(on) == g)
            .select("speed")
            .to_series()
            .to_numpy()
        )
        if vals.size == 0:
            continue
        vals = np.clip(vals, bin_edges[0], bin_edges[-1])
        ax.hist(
            vals,
            bins=bin_edges,
            histtype="step",
            density=True,
            color=color_map[g],
            linewidth=linewidth,
            label=str(g),
        )
    ax.set_title("Implied Trip Speed (km/h)", fontsize="large")
    ax.set_xlabel("km/h")

    # Panel 3: non-positive duration share
    ax = axes[2]
    ax.set_facecolor(ax_bg)
    xs = range(len(groups))
    heights = [negative_map.get(g, 0.0) for g in groups]
    ax.bar(xs, heights, color=[color_map[g] for g in groups])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Non-positive Duration Trips", fontsize="large")

    handles = [
        Line2D([0], [0], color=color_map[g], lw=linewidth, label=str(g))
        for g in groups
    ]
    axes[1].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=False,
        fontsize="medium",
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_activity_duration_by_type(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    n_cols: int = 3,
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    linewidth: float = 2.5,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    min_group_rows: int = 10,
    save_path: str | Path | None = None,
):
    """Activity duration distribution per source, faceted by activity type.

    Derives activities via `post_process.trips_to_activities`. A source
    with an implausible duration shape for a specific purpose (e.g. `work`
    durations clustering near 0 or near a full day) points to a
    purpose-specific time-encoding issue rather than a general one.
    """
    activities = post_process.trips_to_activities(attributes, trips)
    activities = activities.join(
        attributes.select("pid", on), on="pid", how="left"
    ).with_columns(duration=(pl.col("end") - pl.col("start")).cast(pl.Float64))

    groups = _non_null_groups(activities, on)
    act_types = sorted(
        activities.select("act").drop_nulls().unique().to_series().to_list()
    )
    color_map = _group_color_map(groups, cmap_name)

    n_plots = len(act_types) + 1  # +1 for legend
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    lo_p, hi_p = clip_percentiles

    for idx, act in enumerate(act_types):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        ax.set_facecolor(ax_bg)

        sub_act = activities.filter(pl.col("act") == act)
        all_durations = sub_act.select("duration").to_series().to_numpy()
        if all_durations.size == 0:
            ax.axis("off")
            continue
        hi = float(np.percentile(all_durations, hi_p))
        bin_edges = np.linspace(0, max(hi, 1.0), 30)

        for g in groups:
            vals = (
                sub_act.filter(pl.col(on) == g)
                .select("duration")
                .to_series()
                .to_numpy()
            )
            if vals.size < min_group_rows:
                continue
            vals = np.clip(vals, 0, hi)
            ax.hist(
                vals,
                bins=bin_edges,
                histtype="step",
                density=True,
                color=color_map[g],
                linewidth=linewidth,
                label=str(g),
            )

        ax.set_title(act.title(), fontsize="large")
        ax.set_xlabel("minutes")

    # legend cell
    idx = len(act_types)
    r, c = idx // n_cols, idx % n_cols
    ax = axes[r][c]
    ax.set_facecolor(ax_bg)
    handles = [
        Line2D([0], [0], color=color_map[g], lw=linewidth, label=str(g))
        for g in groups
    ]
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize="large",
    )

    for i in range(n_plots - 1, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _activity_counts_per_person(
    attributes: pl.DataFrame, trips: pl.DataFrame, act_types: list[str]
) -> pl.DataFrame:
    """Per-person activity counts, one column per `act_types` entry, zero-filled.

    Every pid in `attributes` gets a row, including persons with no trips
    (all counts 0) — needed so attribute categories with mostly-inactive
    persons (e.g. "retired") aren't silently dropped from the denominator.
    """
    activities = post_process.trips_to_activities(attributes, trips)
    counts = (
        activities.filter(pl.col("act").is_in(act_types))
        .group_by("pid", "act")
        .agg(n=pl.len())
        .pivot(on="act", index="pid", values="n")
    )
    counts = attributes.select("pid").join(counts, on="pid", how="left")
    counts = counts.with_columns(
        (
            pl.col(act).fill_null(0) if act in counts.columns else pl.lit(0)
        ).alias(act)
        for act in act_types
    )
    return counts.select("pid", *act_types)


def plot_activity_count_by_attribute(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    attribute_col: str = "employment",
    act_types: Optional[list[str]] = None,
    on: str = "source",
    n_cols: int = 3,
    cmap_name: str = "tab20",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    bar_width: float = 0.8,
    save_path: str | Path | None = None,
):
    """Mean per-person activity count by attribute category, faceted by activity type.

    Bars are dodged by `on` (source) so cross-survey coding differences are
    visible directly. E.g. `attribute_col="employment"`, `act_types=["work",
    "education"]` should show "student" peaking on education and
    "employed"/"ft-employed" peaking on work — a category that doesn't
    follow the expected pattern (e.g. "unemployed" with a high mean `work`
    count) usually signals a miscoded attribute or activity-purpose field.
    """
    if act_types is None:
        act_types = ["work", "education"]

    counts = _activity_counts_per_person(attributes, trips, act_types)
    counts = counts.join(
        attributes.select("pid", attribute_col, on), on="pid", how="left"
    )

    cats = sorted(
        counts.select(attribute_col).drop_nulls().unique().to_series().to_list()
    )
    if not cats:
        raise ValueError(f"No non-null categories found in '{attribute_col}'.")
    groups = _non_null_groups(counts, on)
    color_map = _group_color_map(groups, cmap_name)

    n_plots = len(act_types) + 1  # +1 for legend
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    x = np.arange(len(cats))
    width = bar_width / max(len(groups), 1)

    for idx, act in enumerate(act_types):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        ax.set_facecolor(ax_bg)

        agg = counts.group_by([attribute_col, on]).agg(
            pl.col(act).mean().alias("mean_count")
        )

        for gi, g in enumerate(groups):
            sub = agg.filter(pl.col(on) == g)
            values = dict(
                zip(sub[attribute_col].to_list(), sub["mean_count"].to_list())
            )
            heights = [values.get(cat, 0.0) for cat in cats]
            offset = (gi - (len(groups) - 1) / 2) * width
            ax.bar(
                x + offset,
                heights,
                width=width,
                color=color_map[g],
                label=str(g),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title(
            f"Mean {act.title()} Activities by {attribute_col.title()}",
            fontsize="large",
        )
        ax.set_ylabel("mean count / person")

    # legend cell
    idx = len(act_types)
    r, c = idx // n_cols, idx % n_cols
    ax = axes[r][c]
    ax.set_facecolor(ax_bg)
    handles = [Patch(facecolor=color_map[g], label=str(g)) for g in groups]
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize="large",
    )

    for i in range(n_plots - 1, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_attribute_activity_heatmap(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    attribute_col: str = "employment",
    on: str = "source",
    cmap_name: str = "YlOrRd",
    n_cols: int = 3,
    fig_bg: str = "lightgray",
    save_path: str | Path | None = None,
):
    """Mean activity-count matrix (attribute category x activity type), per source.

    Cell (row, col) is the mean number of `col`-purpose activities per person
    in attribute category `row`, one heatmap per `on` group with a shared
    colour scale for cross-source comparability. Look for rows that don't
    match the category's expected activity profile — e.g. "student" with
    near-zero `education`, or "retired" with a high `work` mean — which
    usually points to a miscoded attribute or activity-purpose field rather
    than genuine behavioural variation.
    """
    activities = post_process.trips_to_activities(attributes, trips)
    act_types = sorted(
        activities.select("act").drop_nulls().unique().to_series().to_list()
    )

    counts = _activity_counts_per_person(attributes, trips, act_types)
    counts = counts.join(
        attributes.select("pid", attribute_col, on), on="pid", how="left"
    )

    cats = sorted(
        counts.select(attribute_col).drop_nulls().unique().to_series().to_list()
    )
    if not cats:
        raise ValueError(f"No non-null categories found in '{attribute_col}'.")
    groups = _non_null_groups(counts, on)

    def _matrix_for(sub: pl.DataFrame) -> np.ndarray:
        agg = sub.group_by(attribute_col).agg(
            [pl.col(act).mean().alias(act) for act in act_types]
        )
        agg_map = {row[attribute_col]: row for row in agg.iter_rows(named=True)}
        return np.array(
            [
                [agg_map.get(cat, {}).get(act, 0.0) for act in act_types]
                for cat in cats
            ]
        )

    vmax = max(float(_matrix_for(counts).max()), 1e-9)

    n_rows = math.ceil(len(groups) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    for idx, g in enumerate(groups):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]

        matrix = _matrix_for(counts.filter(pl.col(on) == g))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=0, vmax=vmax)

        for i in range(len(cats)):
            for j in range(len(act_types)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    color="black",
                )

        ax.set_xticks(range(len(act_types)))
        ax.set_xticklabels(act_types, rotation=45, ha="right")
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats)
        ax.set_title(str(g), fontsize="large")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(groups), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")

    fig.suptitle(
        f"Mean Activity Count by {attribute_col.title()}", fontsize="large"
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def activity_summary_table(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    on: str = "source",
    markdown: bool = False,
) -> pl.DataFrame | str:
    """Per-source, per-activity-type participation and typical duration.

    Derives activities via `post_process.trips_to_activities`. Participation
    probability is the share of persons in each source group with at least
    one activity of that type (P(count >= 1)); rate is the expected number
    of that activity per person (mean count, including zeros). Duration
    stats are computed over all activities of that type (not per-person).
    """
    activities = post_process.trips_to_activities(attributes, trips)
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
    sep = "|" + "|".join("-" * len(h) for h in headers) + "|"

    blocks = []
    for act in table["act"].unique(maintain_order=False).sort().to_list():
        lines = [f"**{act}**", "", "| " + " | ".join(headers) + " |", sep]
        for row in table.filter(pl.col("act") == act).iter_rows(named=True):
            cells = [
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
            lines.append("| " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


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
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" * len(h) for h in headers) + "|",
    ]
    for row in table.iter_rows(named=True):
        cells = [
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
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
