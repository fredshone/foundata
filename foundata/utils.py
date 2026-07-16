import functools
import random
from pathlib import Path
from typing import Iterable

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


def sample_to_euro(bounds, rate=1.0):
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * rate)


def resolve_activity_chain(
    data: pl.DataFrame, group_cols: list[str]
) -> pl.DataFrame:
    """
    Resolve round-trip destination activities and chain origin activities.

    Some trips have no fixed destination (e.g. ODiN's Doel code 10
    "toeren/wandelen", or NTS's TripPurpTo_B01ID code 15 "Day trip/just
    walk") and are mapped to a null `dact`. These are resolved to the last
    known real activity within `group_cols` (i.e. they return to wherever
    the traveller came from), cascading correctly across consecutive round
    trips. The first trip within a group falls back to its own `oact` if it
    is itself a round trip.

    Each trip's `oact` is then set to the previous trip's (resolved) `dact`,
    chaining activities across the group (e.g. a person's day). If a
    group's first trip is itself a round trip on both ends (no real
    activity anywhere in the group to anchor to — e.g. `oact` shares the
    same round-trip vocabulary as `dact`, unlike ODiN's VertLoc-derived
    `oact`), the remaining nulls are filled with "unknown".
    """
    data = data.sort(group_cols + ["seq"]).with_columns(
        dact=pl.coalesce(
            pl.col("dact"), pl.when(pl.col("seq") == 1).then(pl.col("oact"))
        )
        .forward_fill()
        .over(group_cols)
        .fill_null("unknown")
    )
    return data.with_columns(
        oact=pl.col("dact")
        .shift(1)
        .over(group_cols)
        .fill_null(pl.col("oact"))
        .fill_null("unknown")
    )


def combine_consecutive_acts(
    trips: pl.DataFrame,
    non_consecutive_types: list[str] = ["home", "work", "education"],
    on: str = "pid",
) -> pl.DataFrame:
    """Combine consecutive activities of the same non-consecutive type.

    Where a trip's destination activity is the same as the activity
    immediately before it (either its own origin, for a "there and back"
    trip, or the previous trip's destination, for a chain broken
    elsewhere), that trip is removed rather than filtering out the whole
    plan. Removing it merges the two same-type activities either side of
    it into one, instead of leaving an artificial extra activity in the
    plan.

    Args:
        trips: DataFrame of trips with columns matching `on`, "seq", "oact", "dact".
        non_consecutive_types: List of activity types that are not allowed to appear consecutively (e.g. "work", "education").
        on: Column name to join on (default "pid").

    Returns:
        trips DataFrame with the redundant trips removed.
    """
    n = len(trips)
    redundant = (
        trips.sort(on, "seq")
        .with_columns(prev_dact=pl.col("dact").shift(1).over(on))
        .filter(
            (
                (pl.col("oact") == pl.col("dact"))
                | (pl.col("dact") == pl.col("prev_dact"))
            )
            & pl.col("dact").is_in(non_consecutive_types)
        )
        .select(on, "seq")
    )
    clean_trips = trips.join(
        redundant, on=[on, "seq"], how="anti", maintain_order="left"
    )
    nn = n - len(clean_trips)
    if nn:
        print(
            f"Combined {nn}/{n} trips with consecutive activities of the same non-consecutive types ({100 * nn / n:.1f}%)"
        )
    return clean_trips


def odin_equivalence(num_adults, num_children):
    """Compute equivalence factor for household based on number of adults and children.

    Based on ODIN equivalence scale (https://www.odin.dk/en/odin-equivalence-scale/).
    """
    return (num_adults + (0.8 * num_children)) ** 0.5


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)


def template() -> Path:
    return Path(__file__).parent.parent / "configs" / "core" / "template.yaml"


def compute_avg_speed(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> pl.DataFrame:
    """Add avg_speed (km/h) column to attributes.

    Computed as total trip distance / total trip duration per `on` group
    (default "pid"). Null for groups with no valid trips or zero total
    duration.
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
        .group_by(on)
        .agg(
            total_distance=pl.col("distance").sum(),
            total_duration=pl.col("duration").sum(),
        )
        .with_columns(
            avg_speed=(
                pl.col("total_distance") / (pl.col("total_duration") / 60)
            ).cast(pl.Float32)
        )
        .select(on, "avg_speed")
    )
    attributes = attributes.join(speed, on=on, how="left")
    return attributes


def split_employment_type(attributes: pl.DataFrame) -> pl.DataFrame:
    """Split the ft/pt distinction out of employment into employed_type.

    "ft-employed" and "pt-employed" collapse to "employed", with the split
    preserved in the new "employed_type" column ("ft"/"pt"). Categories the
    split doesn't apply to (bare "employed", "student", "unemployed",
    "retired", "other", "void") get "void". "unknown"/null employment maps
    to "unknown" employed_type.
    """
    attributes = attributes.with_columns(
        employed_type=pl.when(pl.col("employment") == "ft-employed")
        .then(pl.lit("ft"))
        .when(pl.col("employment") == "pt-employed")
        .then(pl.lit("pt"))
        .when(
            (pl.col("employment") == "unknown") | pl.col("employment").is_null()
        )
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void"))
    )
    return attributes.with_columns(
        employment=pl.col("employment").replace(
            {"ft-employed": "employed", "pt-employed": "employed"}
        )
    )


@functools.lru_cache(maxsize=1)
def _load_template() -> dict:
    with open(template()) as f:
        return yaml.safe_load(f)


def get_template_attributes() -> dict:
    return _load_template()["attributes"]


def get_template_trips() -> dict:
    return _load_template()["trips"]


def norm_weights(
    attributes: pl.DataFrame, weight_col: str = "weight"
) -> pl.DataFrame:
    # norm weights to average 1
    if weight_col not in attributes.columns:
        raise ValueError(
            f"Weight column '{weight_col}' not found in attributes"
        )
    if (
        not attributes.select(pl.col(weight_col).is_not_null())
        .to_series()
        .all()
    ):
        print(
            "Warning: Some weights are null — these will be treated as zero in normalization"
        )
    # check for non-positive weights to avoid skewing normalization
    if (
        not attributes.select(pl.col(weight_col).fill_null(0).gt(0))
        .to_series()
        .all()
    ):
        print(
            "Warning: Some weights are non-positive (<= 0) — these will be treated as zero in normalization"
        )
        attributes = attributes.with_columns(
            pl.col(weight_col)
            .fill_null(0)
            .clip(lower_bound=0)
            .alias(weight_col)
        )
    avg_weight = attributes[weight_col].fill_null(0).mean()
    if avg_weight == 0:
        print("Warning: Total weight is zero — returning all weights as 1")
        return attributes.with_columns(
            pl.lit(1, dtype=pl.Float32).alias(weight_col)
        )
    return attributes.with_columns(
        (pl.col(weight_col).fill_null(0) / avg_weight)
        .cast(pl.Float32)
        .alias(weight_col)
    )
