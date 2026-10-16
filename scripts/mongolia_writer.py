"""Write samples of the mongolia source's raw tables (Stata .dta for 2023, SPSS
.sav for 2019 and 2011).

Mongolia's TUS is restricted (registered download, no redistribution
terms), so samples are anonymised like the other restricted sources:

- only the columns the loader reads are kept (no names, addresses or other
  free text);
- households get fresh synthetic ids, consistent across a round's tables
  (person line numbers are kept);
- household and person attribute columns are each shuffled independently
  across the sample, so no real respondent's combination survives (the
  parts of the home location code are shuffled together, so they still
  form a real code);
- diary rows are kept intact apart from the remapped ids, so the diary
  timing and sequencing stay coherent.
"""

from pathlib import Path

import polars as pl
import pyreadstat

# raw id columns of each table; a table's household key is its first
# len(KEYS[year]["hh"]) columns
KEYS = {
    2023: {
        "hh": ["hhid"],
        "person": ["hhid", "pid"],
        "diary": ["hhid", "pid"],
    },
    2019: {
        "hh": ["HH2", "HH3"],
        "person": ["HH2", "HH3", "HL1"],
        "diary": ["HD02", "HD03", "HD05"],
    },
    2011: {
        "hh": ["HH1", "HH2"],
        "person": ["HH1", "HH2", "HL1"],
        "questionnaire": ["HH1", "HH2", "HL1"],
        "diary": ["HH1", "HH2", "HL1"],
    },
}

# synthetic households: 2023 hhid = SYNTHETIC_HHID + i (person ids are
# hhid * 10 + the member digit, as in the real data); older rounds get
# cluster SYNTHETIC_CLUSTER + i, household number 1
SYNTHETIC_HHID = 100000
SYNTHETIC_CLUSTER = 9000


def _columns(year: int, configs: dict) -> dict[str, list[str]]:
    """The raw columns the loader reads from each of a round's tables."""
    from foundata.utils import config_for_year

    hh, person, trip = (
        config_for_year(configs[n], str(year)) for n in ("hh", "person", "trip")
    )
    extras = [
        *(col for rule in person["employment_rules"] for col in rule),
        *person["disability_columns"],
    ]
    cols = {
        "hh": [
            *hh["column_mappings"],
            *hh["vehicle_columns"],
            *(hh["mode_availability"] or {}).values(),
        ],
        "diary": list(trip["column_mappings"]),
    }
    mappings = person["column_mappings"]
    if "persons" in mappings:
        cols["person"] = list(mappings["persons"])
        cols["questionnaire"] = [*mappings["questionnaire"], *extras]
    else:
        cols["person"] = [*mappings, *extras]
    return {
        table: list(dict.fromkeys([*KEYS[year][table], *c]))
        for table, c in cols.items()
    }


def _hh_key(df: pl.DataFrame, cols: list[str]) -> pl.Expr:
    return pl.concat_str(
        [pl.col(c).cast(pl.Int64).cast(pl.String) for c in cols], separator="_"
    )


def _shuffle_non_keys(
    df: pl.DataFrame, keys: list[str], seed: int, together: list[str] = ()
) -> pl.DataFrame:
    """Shuffle each non-key column independently, except the *together*
    columns (the parts of the home location code), which are shuffled as
    one block so they still form a real code."""
    together = [c for c in together if c in df.columns]
    rest = [c for c in df.columns if c not in keys and c not in together]
    df = df.with_columns(
        df[c].sample(fraction=1.0, shuffle=True, seed=seed + i).alias(c)
        for i, c in enumerate(rest)
    )
    if together:
        block = df.select(together).sample(
            fraction=1.0, shuffle=True, seed=seed - 1
        )
        df = df.with_columns(block)
    return df


def _remap(
    df: pl.DataFrame, year: int, table: str, hh_map: dict
) -> pl.DataFrame:
    keys = KEYS[year][table]
    n_hh = len(KEYS[year]["hh"])
    new = _hh_key(df, keys[:n_hh]).replace_strict(hh_map, return_dtype=pl.Int64)
    if year == 2023:
        hhid = (SYNTHETIC_HHID + new).cast(pl.Float64)
        if "pid" in keys:
            member = pl.col("pid") - pl.col("hhid") * 10
            return df.with_columns(pid=hhid * 10 + member, hhid=hhid)
        return df.with_columns(hhid=hhid)
    return df.with_columns(
        (SYNTHETIC_CLUSTER + new).cast(pl.Float64).alias(keys[0]),
        pl.lit(1.0).alias(keys[1]),
    )


def sample_mongolia(
    src_root: Path,
    dst_root: Path,
    n: int,
    seed: int,
    require_road_travel: bool = False,
) -> None:
    """Write `n` anonymised households per round (with their members and
    diaries) to dst_root/<year>/.

    require_road_travel: only sample households with a diary episode on the
    road / in transport, so a tiny sample still yields trips.
    """
    from foundata.mongolia import ROUNDS, read_table
    from foundata.utils import (
        config_for_year,
        get_config_path,
        load_yaml_config,
    )

    config_dir = get_config_path() / "mongolia"
    configs = {
        name: load_yaml_config(config_dir / f"{name}_dictionary.yaml")
        for name in ("hh", "person", "trip")
    }
    for year, round_ in ROUNDS.items():
        src, dst = src_root / round_["dir"], dst_root / round_["dir"]
        columns = _columns(year, configs)
        tables = {
            table: read_table(src / round_["files"][table], cols)
            for table, cols in columns.items()
        }
        n_hh = len(KEYS[year]["hh"])
        hh_keys = {
            table: _hh_key(df, KEYS[year][table][:n_hh])
            for table, df in tables.items()
        }

        diary = tables["diary"].with_columns(key=hh_keys["diary"])
        candidates = diary["key"]
        if require_road_travel:
            trip = config_for_year(configs["trip"], str(year))
            location = next(
                raw
                for raw, new in trip["column_mappings"].items()
                if new == "location"
            )
            candidates = diary.filter(
                pl.col(location).cast(pl.Int64).is_in(trip["road_locations"])
            )["key"]
        sampled = candidates.unique().sort().sample(n=n, seed=seed)
        hh_map = {k: i for i, k in enumerate(sorted(sampled))}

        for i, (table, df) in enumerate(tables.items()):
            df = df.filter(hh_keys[table].is_in(sampled.implode()))
            if table != "diary":
                hh_mapping = config_for_year(configs["hh"], str(year))[
                    "column_mappings"
                ]
                location = [
                    raw
                    for raw, new in hh_mapping.items()
                    if new.startswith("home_")
                ]
                df = _shuffle_non_keys(
                    df, KEYS[year][table], seed + 100 * i, together=location
                )
            df = _remap(df, year, table, hh_map)
            path = dst / round_["files"][table]
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = (
                pyreadstat.write_dta
                if path.suffix == ".dta"
                else pyreadstat.write_sav
            )
            writer(df.to_pandas(), str(path))
            print(f"  {year} {table}: {df.height} rows")
