from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import polars as pl

_NUMERIC_DTYPES = set(pl.NUMERIC_DTYPES)
_CATEGORICAL_DTYPES = {pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean}


@dataclass(frozen=True)
class TableRef:
    name: str
    table: pl.DataFrame


def _dtype_kind(dtype: pl.DataType) -> str:
    if dtype in _NUMERIC_DTYPES:
        return "numeric"
    if dtype in _CATEGORICAL_DTYPES:
        return "categorical"
    return "other"


def _format_values(values: Sequence, max_values: int) -> str:
    preview = list(values)[:max_values]
    suffix = (
        ""
        if len(values) <= max_values
        else f" (+{len(values) - max_values} more)"
    )
    return f"{preview}{suffix}"


def _compare_columns(
    template: pl.DataFrame, other: pl.DataFrame
) -> tuple[list[str], list[str]]:
    template_cols = set(template.columns)
    other_cols = set(other.columns)
    missing = sorted(template_cols - other_cols)
    extra = sorted(other_cols - template_cols)
    return missing, extra


def _compare_kinds(template: pl.DataFrame, other: pl.DataFrame) -> list[str]:
    mismatches: list[str] = []
    shared = set(template.columns) & set(other.columns)
    for col in sorted(shared):
        t_kind = _dtype_kind(template[col].dtype)
        o_kind = _dtype_kind(other[col].dtype)
        if t_kind != o_kind:
            mismatches.append(f"{col} ({t_kind} vs {o_kind})")
    return mismatches


def _compare_categorical_values(
    template: pl.DataFrame, other: pl.DataFrame, max_values: int
) -> list[str]:
    diffs: list[str] = []
    shared = set(template.columns) & set(other.columns)
    for col in sorted(shared):
        if _dtype_kind(template[col].dtype) != "categorical":
            continue
        if _dtype_kind(other[col].dtype) != "categorical":
            continue
        t_values = template[col].drop_nulls().unique().to_list()
        o_values = other[col].drop_nulls().unique().to_list()
        t_set = set(t_values)
        o_set = set(o_values)
        missing = sorted(t_set - o_set)
        extra = sorted(o_set - t_set)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing={_format_values(missing, max_values)}")
            if extra:
                parts.append(f"extra={_format_values(extra, max_values)}")
            diffs.append(f"{col} ({', '.join(parts)})")
    return diffs


def _compare_numeric_values(
    template: pl.DataFrame, other: pl.DataFrame, rel_tol: float, abs_tol: float
) -> list[str]:
    diffs: list[str] = []
    shared = set(template.columns) & set(other.columns)
    for col in sorted(shared):
        if _dtype_kind(template[col].dtype) != "numeric":
            continue
        if _dtype_kind(other[col].dtype) != "numeric":
            continue
        t_series = template[col].drop_nulls()
        o_series = other[col].drop_nulls()
        if t_series.is_empty() or o_series.is_empty():
            continue
        t_stats = t_series.describe()
        o_stats = o_series.describe()
        t_row = t_stats.row(1)
        o_row = o_stats.row(1)
        t_map = dict(zip(t_stats["statistic"], t_row))
        o_map = dict(zip(o_stats["statistic"], o_row))
        for stat in ("min", "max", "mean", "std"):
            t_val = t_map.get(stat)
            o_val = o_map.get(stat)
            if t_val is None or o_val is None:
                continue
            diff = abs(t_val - o_val)
            rel = diff / max(abs(t_val), abs(o_val), abs_tol)
            if diff > abs_tol and rel > rel_tol:
                diffs.append(f"{col}.{stat} ({t_val} vs {o_val})")
                break
    return diffs


def verify_tables(
    template: TableRef,
    others: Iterable[TableRef],
    *,
    max_values: int = 10,
    numeric_rel_tol: float = 0.05,
    numeric_abs_tol: float = 1e-6,
) -> None:
    """Compare tables to a template and log differences to stdout."""
    template_name = template.name
    template_table = template.table

    for other in others:
        missing, extra = _compare_columns(template_table, other.table)
        kind_mismatches = _compare_kinds(template_table, other.table)
        cat_diffs = _compare_categorical_values(
            template_table, other.table, max_values
        )
        num_diffs = _compare_numeric_values(
            template_table, other.table, numeric_rel_tol, numeric_abs_tol
        )

        if not (missing or extra or kind_mismatches or cat_diffs or num_diffs):
            print(f"[verify] {other.name}: OK vs {template_name}")
            continue

        print(f"[verify] {other.name} vs {template_name}")
        if missing:
            print(f"  missing columns: {missing}")
        if extra:
            print(f"  extra columns: {extra}")
        if kind_mismatches:
            print(f"  kind mismatches: {kind_mismatches}")
        if cat_diffs:
            print(f"  categorical diffs: {cat_diffs}")
        if num_diffs:
            print(f"  numeric diffs: {num_diffs}")
