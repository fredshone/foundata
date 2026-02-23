import random
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl
import yaml
from rapidfuzz import fuzz, process


def fuzzy_loader(path: str | Path, target: str, **kwargs) -> pl.DataFrame:
    # look in given path for closest math to target
    candidates = [f.name for f in path.iterdir()]
    if not candidates:
        raise FileNotFoundError(f"No file found in {path}")
    # fuzzy find best candidate
    best_match, score, idx = process.extractOne(
        target, candidates, scorer=fuzz.ratio, score_cutoff=80
    )
    if best_match is None:
        raise FileNotFoundError(
            f"No file found in {path} matching {target} (best match: {best_match} with score {score})"
        )
    print(
        f"Fuzzy match loading {best_match} from {path} (score: {int(score)}%)"
    )
    return pl.read_csv(path / candidates[idx], **kwargs)


def expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def load_yaml_config(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def check_overlap(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    on: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
) -> set:
    on_lhs = set(lhs[on])
    on_rhs = set(rhs[on])
    missing_in_lhs = on_rhs - on_lhs
    missing_in_rhs = on_lhs - on_rhs
    n_lhs = len(missing_in_lhs)
    n_rhs = len(missing_in_rhs)
    perc_lhs = n_lhs / len(on_rhs) * 100
    perc_rhs = n_rhs / len(on_lhs) * 100

    if missing_in_lhs:
        n = min(n_lhs, 3)
        print(
            f"Warning: Missing {n_lhs} ({perc_lhs:.1f}%) of '{on}' keys in '{lhs_name}': {list(missing_in_lhs)[:n]}"
        )
    else:
        print(f"All '{on}' keys in '{rhs_name}' are present in '{lhs_name}'")

    if missing_in_rhs:
        n = min(n_rhs, 3)
        print(
            f"Warning: Missing {n_rhs} ({perc_rhs:.1f}%) of '{on}' keys in '{rhs_name}': {list(missing_in_rhs)[:n]}"
        )
    else:
        print(f"All '{on}' keys in '{lhs_name}' are present in '{rhs_name}'")


def table_joiner(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    on: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    maintain_order: str = "left_right",
) -> pl.DataFrame:
    # check for missing keys
    check_overlap(lhs, rhs, on, lhs_name=lhs_name, rhs_name=rhs_name)

    # check for duplicates
    a_cols = lhs.columns
    b_cols = rhs.columns
    duplicates = set(a_cols) & set(b_cols) - {on}
    if duplicates:
        print(f"Warning: Duplicate columns (other than join key): {duplicates}")

    return lhs.join(rhs, on=on, maintain_order=maintain_order)


def tables_joiner(tables: Mapping[int, pl.DataFrame], on: str) -> pl.DataFrame:
    if not tables:
        raise ValueError("No tables to join")

    result = tables[0]
    for table in tables[1:]:
        result = table_joiner(result, table, on)

    return result


def table_stacker(tables: Iterable[pl.DataFrame]) -> pl.DataFrame:
    all_columns = set()
    for table in tables:
        all_columns.update(table.columns)
    for idx, table in enumerate(tables):
        missing_columns = all_columns - set(table.columns)
        if missing_columns:
            raise UserWarning(
                f"Cannot stack tables: Table {idx} is missing columns: {missing_columns}"
            )
    base_order = tables[0].columns
    return pl.concat([table.select(base_order) for table in tables])


def config_for_year(config: dict, year):
    cnfg = {}
    for key, value in config.items():
        if year in value:
            cnfg[key] = value[year]
        else:
            cnfg[key] = value["default"]
    return cnfg


def sample_int_range(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return pl.lit(None, pl.Int32)
    a, b = bounds
    return random.randint(int(a), int(b))


def sample_us_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return pl.lit(None, pl.Int32)
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 0.85)


def sample_uk_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 1.14)


def sample_aus_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return pl.lit(None, pl.Int32)
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 0.6)


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)
