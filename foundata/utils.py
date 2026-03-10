import random
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl
import yaml
from rapidfuzz import fuzz, process

DTYPE_MAP = {
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int": pl.Int32,
    "integer": pl.Int32,
    "float": pl.Float32,
    "string": pl.String,
    "str": pl.String,
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
}


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


def bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def sample_int_range(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return random.randint(int(a), int(b))


def sample_us_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 0.85)


def sample_uk_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 1.14)


def sample_aus_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 0.6)


def sample_krw_to_euro(bounds: tuple[int, int] | None) -> int | None:
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * 0.00058)


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)


def template() -> Path:
    return Path(__file__).parent.parent / "configs" / "core" / "template.yaml"


def compute_avg_speed(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> pl.DataFrame:
    """Add avg_speed (km/h) column to attributes.

    Computed as total trip distance / total trip duration per person.
    Null for persons with no valid trips or zero total duration.
    """
    # check distances are not null and durations are positive to avoid skewing avg_speed
    if not trips.select(pl.col("distance").is_not_null()).to_series().all():
        print(
            "Warning: Some trips have null distance — these will be excluded from avg_speed calculation"
        )
    if not trips.select(pl.col("tet") > pl.col("tst")).to_series().all():
        print(
            "Warning: Some trips have non-positive duration (tet <= tst) — these will be excluded from avg_speed calculation"
        )

    speed = (
        trips.with_columns(duration=pl.col("tet") - pl.col("tst"))
        .filter(pl.col("duration") > 0)
        .filter(pl.col("distance").is_not_null())
        .group_by("pid")
        .agg(
            total_distance=pl.col("distance").sum(),
            total_duration=pl.col("duration").sum(),
        )
        .with_columns(
            avg_speed=(
                pl.col("total_distance") / (pl.col("total_duration") / 60)
            ).cast(pl.Float32)
        )
        .select("pid", "avg_speed")
    )
    return attributes.join(speed, on="pid", how="left")


def get_template_attributes() -> set[str]:
    # load yaml config and return set of expected columns
    with open(template()) as f:
        config = yaml.safe_load(f)
    return config["attributes"]


def get_template_trips() -> set[str]:
    # load yaml config and return set of expected columns
    with open(template()) as f:
        config = yaml.safe_load(f)
    return config["trips"]


def norm_weights(
    attributes: pl.DataFrame, weight_col: str = "weight"
) -> pl.DataFrame:
    # norm weights to average 1
    if weight_col not in attributes.columns:
        raise ValueError(
            f"Weight column '{weight_col}' not found in attributes"
        )
    if not attributes.select(pl.col(weight_col).is_not_null()).to_series().all():
        print(
            "Warning: Some weights are null — these will be treated as zero in normalization"
        )
    # check for non-positive weights to avoid skewing normalization
    if not attributes.select(pl.col(weight_col).fill_null(0).gt(0)).to_series().all():
        print(
            "Warning: Some weights are non-positive (<= 0) — these will be treated as zero in normalization"
        )
        attributes = attributes.with_columns(
            pl.col(weight_col).fill_null(0).clip(lower_bound=0).alias(weight_col)
        )
    avg_weight = attributes[weight_col].fill_null(0).mean()
    if avg_weight == 0:
        print("Warning: Total weight is zero — returning all weights as 1")
        return attributes.with_columns(pl.lit(1, dtype=pl.Float32).alias(weight_col))
    return attributes.with_columns(
        (pl.col(weight_col).fill_null(0) / avg_weight).cast(pl.Float32).alias(weight_col)
    )
