import math
from itertools import cycle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from foundata import post_process


def numeric_hist_grid(
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


def summary_trends(
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


def _stratified_sample(
    df: pl.DataFrame, on: str, max_per_group: int
) -> pl.DataFrame:
    """Sample up to `max_per_group` rows per `on` group, not globally.

    A single global sample over pooled multi-source data can, purely by
    chance, drop every row of a rare category within a source that's a
    small share of the total — making that category look absent from that
    source when it isn't. Sampling within each group avoids that.
    """
    groups = df.select(pl.col(on)).drop_nulls().unique().to_series().to_list()
    parts = []
    for g in groups:
        sub = df.filter(pl.col(on) == g)
        if sub.height > max_per_group:
            sub = sub.sample(n=max_per_group, shuffle=True)
        parts.append(sub)
    return pl.concat(parts, how="vertical") if parts else df.head(0)


def categorical_bar_grid(
    df: pl.DataFrame,
    on: str = "source",
    n_cols: int = 3,
    max_sample_per_group: int = 1000,
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

    df_plot = _stratified_sample(df, on, max_sample_per_group)

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


def _kde_bandwidth(vals: np.ndarray) -> float:
    """Silverman's rule of thumb, robust to skew via min(std, IQR/1.34).

    A bandwidth built from the raw std alone is inflated by the long right
    tail typical of durations, which over-smooths the bulk of the
    distribution where the anomalies usually show up. Falling back to the
    IQR-based scale when it's smaller keeps the bandwidth sane for skewed
    data without needing a manual bin width.
    """
    n = vals.size
    std = float(np.std(vals))
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = q75 - q25
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    if scale <= 0:
        scale = std if std > 0 else 1.0
    return max(0.9 * scale * n ** (-1 / 5), 1e-6)


def _gaussian_kde(
    centers: np.ndarray, grid: np.ndarray, bandwidth: float, norm_n: int
) -> np.ndarray:
    diffs = (grid[:, None] - centers[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    return density / (norm_n * bandwidth * np.sqrt(2 * np.pi))


def _kde_subsample(
    vals: np.ndarray, max_sample: int, rng_seed: int = 0
) -> np.ndarray:
    if vals.size <= max_sample:
        return vals
    rng = np.random.default_rng(rng_seed)
    return rng.choice(vals, size=max_sample, replace=False)


def _kde_bounded(
    vals: np.ndarray,
    grid: np.ndarray,
    lo: float = 0.0,
    max_sample: int = 20_000,
) -> np.ndarray:
    """Gaussian KDE for a non-negative quantity (duration, etc.), reflected
    at `lo` so density doesn't leak below the natural lower bound instead
    of piling up at it as a clipped histogram would.
    """
    vals = _kde_subsample(vals, max_sample)
    bandwidth = _kde_bandwidth(vals)
    centers = np.concatenate([vals, 2 * lo - vals])
    return _gaussian_kde(centers, grid, bandwidth, vals.size)


def _kde_circular(
    vals: np.ndarray,
    grid: np.ndarray,
    period: float = 24.0,
    max_sample: int = 20_000,
) -> np.ndarray:
    """Gaussian KDE on a circular domain (e.g. hour-of-day), wrapping mass
    across the period boundary instead of losing it there.
    """
    vals = _kde_subsample(vals, max_sample)
    bandwidth = _kde_bandwidth(vals)
    centers = np.concatenate([vals - period, vals, vals + period])
    return _gaussian_kde(centers, grid, bandwidth, vals.size)


def time_of_day_profile(
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
    day-crossing trips still land in a sensible hour-of-day bucket. Density
    is a Gaussian KDE wrapped across the midnight boundary (rather than a
    fixed-width histogram), so smooth diurnal shape and sharp anomalous
    spikes (e.g. heaping at a specific hour) both show up without an
    arbitrary bin width hiding or exaggerating either.
    """
    trips = trips.join(attributes.select("pid", on), on="pid", how="left")
    groups = _non_null_groups(trips, on)
    color_map = _group_color_map(groups, cmap_name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(fig_bg)
    for ax in axes:
        ax.set_facecolor(ax_bg)

    grid = np.linspace(0, 24, 241)  # ~6-minute resolution

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
            hours = np.mod(vals, 1440) / 60.0
            density = _kde_circular(hours, grid)
            ax.plot(
                grid,
                density,
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


def time_heaping(
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


def trip_time_diagnostics(
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
    grid = np.linspace(0, max(hi, 1.0), 200)
    for g in groups:
        vals = (
            valid.filter(pl.col(on) == g)
            .select("duration")
            .to_series()
            .to_numpy()
        )
        if vals.size == 0:
            continue
        density = _kde_bounded(vals, grid, lo=0.0)
        ax.plot(
            grid, density, color=color_map[g], linewidth=linewidth, label=str(g)
        )
    ax.set_xlim(grid[0], grid[-1])
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


def activity_duration_by_type(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
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

    `activities` should come from `post_process.trips_to_activities`. A
    source with an implausible duration shape for a specific purpose (e.g.
    `work` durations clustering near 0 or near a full day) points to a
    purpose-specific time-encoding issue rather than a general one.
    """
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
        grid = np.linspace(0, max(hi, 1.0), 150)

        for g in groups:
            vals = (
                sub_act.filter(pl.col(on) == g)
                .select("duration")
                .to_series()
                .to_numpy()
            )
            if vals.size < min_group_rows:
                continue
            density = _kde_bounded(vals, grid, lo=0.0)
            ax.plot(
                grid,
                density,
                color=color_map[g],
                linewidth=linewidth,
                label=str(g),
            )

        ax.set_xlim(grid[0], grid[-1])
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


def activity_count_by_attribute(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
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

    `activities` should come from `post_process.trips_to_activities`. Bars
    are dodged by `on` (source) so cross-survey coding differences are
    visible directly. E.g. `attribute_col="employment"`, `act_types=["work",
    "education"]` should show "student" peaking on education and
    "employed"/"ft-employed" peaking on work — a category that doesn't
    follow the expected pattern (e.g. "unemployed" with a high mean `work`
    count) usually signals a miscoded attribute or activity-purpose field.
    """
    if act_types is None:
        act_types = ["work", "education"]

    counts = post_process.activity_counts_per_person(
        attributes, activities, act_types
    )
    # Rename count columns before joining: an act_type can share a name
    # with a person attribute (e.g. "education" is both an activity
    # purpose and an attribute), and pl.DataFrame.join silently suffixes
    # the *incoming* column on a name clash rather than raising — so
    # `attribute_col="education"` would silently read the activity count
    # instead of the attribute value below.
    count_cols = {t: f"__count_{t}" for t in act_types}
    counts = counts.rename(count_cols)
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
            pl.col(count_cols[act]).mean().alias("mean_count")
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


def attribute_activity_heatmap(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
    attribute_col: str = "employment",
    act_types: Optional[list[str]] = None,
    on: str = "source",
    n_bins: Optional[int] = None,
    cmap_name: str = "YlOrRd",
    n_cols: int = 3,
    fig_bg: str = "lightgray",
    save_path: str | Path | None = None,
):
    """Mean activity-count matrix (attribute category x source), one subplot
    per activity type.

    `activities` should come from `post_process.trips_to_activities`. Cell
    (row, col) is the mean number of activities of this subplot's type per
    person in attribute category `row` and source `col` — sources share the
    x-axis within each subplot so they can be compared directly, rather
    than needing to flip between separate per-source heatmaps. Each
    subplot's colour scale is normalised to its own activity type (not
    shared across the grid), since activity types differ hugely in typical
    count (e.g. `home` vs `escort`) and a shared scale would wash out the
    low-count ones. Look for rows that don't match the category's expected
    activity profile — e.g. "student" with near-zero `education`, or
    "retired" with a high `work` mean — which usually points to a miscoded
    attribute or activity-purpose field rather than genuine behavioural
    variation. Pass `n_bins` for a continuous `attribute_col` (e.g.
    `hh_income`) to quantile-bin it into that many rows instead of treating
    it as categorical.
    """
    if act_types is None:
        act_types = sorted(
            activities.select("act").drop_nulls().unique().to_series().to_list()
        )

    counts = post_process.activity_counts_per_person(
        attributes, activities, act_types
    )
    # Rename count columns before joining: an act_type can share a name
    # with a person attribute (e.g. "education" is both an activity
    # purpose and an attribute, and this function's own docstring example
    # uses it as `attribute_col`), and pl.DataFrame.join silently suffixes
    # the *incoming* column on a name clash rather than raising — so
    # `attribute_col` would silently resolve to the activity count instead
    # of the attribute value everywhere below.
    count_cols = {t: f"__count_{t}" for t in act_types}
    counts = counts.rename(count_cols)
    counts = counts.join(
        attributes.select("pid", attribute_col, on), on="pid", how="left"
    )
    groups = _non_null_groups(counts, on)

    edges = None
    if n_bins is not None:
        all_vals = (
            counts.select(attribute_col).drop_nulls().to_series().to_numpy()
        )
        if all_vals.size == 0:
            raise ValueError(f"No non-null values found in '{attribute_col}'.")
        edges = _quantile_bin_edges(all_vals, n_bins)
        cats = _bin_edge_labels(edges)
    else:
        cats = sorted(
            counts.select(attribute_col)
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        if not cats:
            raise ValueError(
                f"No non-null categories found in '{attribute_col}'."
            )

    def _matrix_for(act: str) -> np.ndarray:
        """NaN where a category/source combination has no persons at all —
        distinct from a genuine mean of 0 for persons who exist but never
        do this activity.
        """
        if edges is not None:
            n = len(cats)
            matrix = np.full((n, len(groups)), np.nan)
            for gi, g in enumerate(groups):
                sub = (
                    counts.filter(pl.col(on) == g)
                    .select(attribute_col, count_cols[act])
                    .drop_nulls()
                )
                if sub.height == 0:
                    continue
                vals = sub[attribute_col].to_numpy()
                y = sub[count_cols[act]].to_numpy()
                bin_idx = np.clip(
                    np.digitize(vals, edges[1:-1], right=True), 0, n - 1
                )
                for b in range(n):
                    mask = bin_idx == b
                    if mask.any():
                        matrix[b, gi] = y[mask].mean()
            return matrix

        agg = counts.group_by([attribute_col, on]).agg(
            pl.col(count_cols[act]).mean().alias("mean_count")
        )
        agg_map = {
            (row[attribute_col], row[on]): row["mean_count"]
            for row in agg.iter_rows(named=True)
        }
        return np.array(
            [[agg_map.get((cat, g), np.nan) for g in groups] for cat in cats]
        )

    matrices = {act: _matrix_for(act) for act in act_types}

    n_rows = math.ceil(len(act_types) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="lightgrey")

    for idx, act in enumerate(act_types):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]

        matrix = matrices[act]
        finite = matrix[np.isfinite(matrix)]
        vmax = max(float(finite.max()), 1e-9) if finite.size else 1e-9
        im = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )

        for i in range(len(cats)):
            for j in range(len(groups)):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    "n/a" if np.isnan(value) else f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    color="black",
                )

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats)
        ax.set_title(act.title(), fontsize="large")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(act_types), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")

    fig.suptitle(
        f"Mean Activity Count by {attribute_col.title()} and {on.title()}",
        fontsize="large",
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([float(values.min()), float(values.min()) + 1.0])
    return edges


def _plot_bar_cell(
    ax,
    counts: pl.DataFrame,
    act: str,
    attribute_col: str,
    on: str,
    groups: list,
    color_map: dict,
    bar_width: float,
):
    """`act` is the column in `counts` holding the activity count — the
    caller must pass its (possibly renamed) column name, not necessarily
    the literal activity-type string, to avoid colliding with an
    identically-named attribute column (see `activities_attributes_grid`).
    """
    cats = sorted(
        counts.select(attribute_col).drop_nulls().unique().to_series().to_list()
    )
    x = np.arange(len(cats))
    width = bar_width / max(len(groups), 1)

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
        ax.bar(x + offset, heights, width=width, color=color_map[g])

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize="small")


def _bin_edge_labels(edges: np.ndarray) -> list[str]:
    return [f"{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(edges) - 1)]


def _plot_line_cell(
    ax,
    counts: pl.DataFrame,
    act: str,
    attribute_col: str,
    on: str,
    groups: list,
    color_map: dict,
    n_bins: int,
):
    """`act` is the column in `counts` holding the activity count — see the
    note on `_plot_bar_cell`."""
    all_vals = counts.select(attribute_col).drop_nulls().to_series().to_numpy()
    if all_vals.size == 0:
        return
    edges = _quantile_bin_edges(all_vals, n_bins)
    n = len(edges) - 1
    x = np.arange(n)

    for g in groups:
        sub = (
            counts.filter(pl.col(on) == g)
            .select(attribute_col, act)
            .drop_nulls()
        )
        if sub.height == 0:
            continue
        vals = sub[attribute_col].to_numpy()
        y = sub[act].to_numpy()
        bin_idx = np.clip(np.digitize(vals, edges[1:-1], right=True), 0, n - 1)
        means = np.full(n, np.nan)
        for b in range(n):
            mask = bin_idx == b
            if mask.any():
                means[b] = y[mask].mean()
        ax.plot(
            x, means, color=color_map[g], marker="o", markersize=4, linewidth=2
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        _bin_edge_labels(edges), rotation=45, ha="right", fontsize="small"
    )


def activities_attributes_grid(
    attributes: pl.DataFrame,
    activities: pl.DataFrame,
    act_types: Optional[list[str]] = None,
    attribute_cols: Optional[dict[str, str]] = None,
    on: str = "source",
    n_bins: int = 8,
    cmap_name: str = "Dark2",
    fig_bg: str = "lightgray",
    ax_bg: str = "lightgray",
    bar_width: float = 0.8,
    save_path: str | Path | None = None,
):
    """Mean per-person activity count: one row per activity type, one column
    per attribute, faceted by `on` (source).

    `activities` should come from `post_process.trips_to_activities`.
    `attribute_cols` maps attribute column -> "bar" (categorical, dodged bars
    over sorted categories) or "line" (continuous, mean count within
    quantile bins of the attribute, one line per `on` group, plotted at
    regular bin-index intervals with bin-range tick labels rather than at
    the actual bin-midpoint values — so a skewed distribution doesn't bunch
    points together). Defaults to `{"employment": "bar", "hh_income":
    "line", "age": "line"}` — bars for the categorical driver of activity
    participation, lines for the two continuous ones.
    """
    if act_types is None:
        act_types = sorted(
            activities.select("act").drop_nulls().unique().to_series().to_list()
        )
    if attribute_cols is None:
        attribute_cols = {
            "employment": "bar",
            "hh_income": "line",
            "age": "line",
        }

    counts = post_process.activity_counts_per_person(
        attributes, activities, act_types
    )
    # Rename count columns before joining: an act_type can share a name
    # with a person attribute (e.g. "education"), and pl.DataFrame.join
    # silently suffixes the *incoming* column on a name clash rather than
    # raising — so an attribute in `attribute_cols` would silently shadow
    # the identically-named activity count instead of joining in cleanly.
    count_cols = {t: f"__count_{t}" for t in act_types}
    counts = counts.rename(count_cols)
    counts = counts.join(
        attributes.select("pid", on, *attribute_cols.keys()),
        on="pid",
        how="left",
    )
    groups = _non_null_groups(counts, on)
    color_map = _group_color_map(groups, cmap_name)

    n_rows = len(act_types)
    n_cols = len(attribute_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False
    )
    fig.patch.set_facecolor(fig_bg)

    for r, act in enumerate(act_types):
        for c, (col, kind) in enumerate(attribute_cols.items()):
            ax = axes[r][c]
            ax.set_facecolor(ax_bg)

            if kind == "bar":
                _plot_bar_cell(
                    ax,
                    counts,
                    count_cols[act],
                    col,
                    on,
                    groups,
                    color_map,
                    bar_width,
                )
            elif kind == "line":
                _plot_line_cell(
                    ax,
                    counts,
                    count_cols[act],
                    col,
                    on,
                    groups,
                    color_map,
                    n_bins,
                )
                ax.set_xlabel(col.replace("_", " ").title(), fontsize="small")
            else:
                raise ValueError(
                    f"Unknown kind '{kind}' for '{col}': must be 'bar' or 'line'."
                )

            if r == 0:
                ax.set_title(col.replace("_", " ").title(), fontsize="large")
            if c == 0:
                ax.set_ylabel(f"{act.title()}\nmean count / person")

    handles = [Patch(facecolor=color_map[g], label=str(g)) for g in groups]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(groups), 6),
        frameon=False,
        fontsize="large",
    )

    fig_height = 3.2 * n_rows
    legend_frac = min(0.5 / fig_height, 0.15)
    plt.tight_layout(rect=(0, 0, 1, 1 - legend_frac))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
