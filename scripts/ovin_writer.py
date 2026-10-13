"""Write samples of the ovin and om sources' raw SPSS .sav files.

OViN and OM/MON are restricted (DANS registered access, like ODiN), so
samples are anonymised like the other restricted sources:

- only the columns the loader reads are kept, and only respondent rows and
  trip rows (Verpl >= 1), not the extra ride rows;
- respondents (and om households) get fresh synthetic ids;
- attributes are shuffled independently across the sample, so no real
  respondent's combination survives. Columns that must stay consistent
  are shuffled as a block: home gemeente with its Sted (the loaders'
  trip-zone lookup), and month with day of month and weekday. Columns
  that describe a respondent's trips (Weggeweest/Responsc, Vertrekp) or
  the household's structure (om AantPers) stay put;
- trip rows are kept intact apart from the remapped ids, so trip
  sequencing, times and zones stay coherent.
"""

from pathlib import Path

import polars as pl
import pyreadstat

GEMEENTE = ["WoGem", "Sted"]
DATE = ["Maand", "Dag", "Weekdag"]


def loader_columns(source: str) -> tuple[list[str], list[str], list[str]]:
    """Raw hh, person and trip columns the source's loader reads."""
    from foundata.utils import get_config_path, load_yaml_config

    configs = get_config_path() / source
    hh, person, trip = (
        list(
            load_yaml_config(configs / f"{name}_dictionary.yaml")[
                "column_mappings"
            ]
        )
        for name in ("hh", "person", "trip")
    )
    return hh, person, trip


def shuffle(
    df: pl.DataFrame, blocks: list[list[str]], fixed: set[str], seed: int
) -> pl.DataFrame:
    """Shuffle each block of columns, and every other column not in
    `fixed` on its own, independently across rows."""
    blocked = {c for block in blocks for c in block}
    singles = [[c] for c in df.columns if c not in fixed | blocked]
    shuffled = df
    for i, block in enumerate(blocks + singles):
        order = pl.Series(range(len(df))).shuffle(seed=seed + i)
        shuffled = shuffled.with_columns(
            df.select(block)[order.to_list()].get_columns()
        )
    return shuffled


def sample_ovin(
    src_root: Path, dst_root: Path, n: int, seed: int, min_trips: int = 0
) -> None:
    """Write `n` anonymised respondents per OViN year.

    With `min_trips`, only respondents with at least that many regular
    trips are sampled (so a tiny template sample still yields plans).
    """
    from foundata.ovin import DATA_FILES, read_sav

    hh, person, trip = loader_columns("ovin")
    respondent_cols = list(dict.fromkeys(["OP"] + hh + person))
    trip_cols = [c for c in trip if c not in respondent_cols]
    for i, (year, name) in enumerate(DATA_FILES.items()):
        data = read_sav(
            src_root / str(year) / name, respondent_cols + trip_cols
        )
        respondents = data.filter(pl.col("OP") == 1).select(respondent_cols)
        trips = data.filter(pl.col("Verpl") >= 1)

        regular = trips.filter(pl.col("Verpl") == 1).group_by("OPID").len()
        eligible = respondents.join(regular, on="OPID", how="left").filter(
            pl.col("len").fill_null(0) >= min_trips
        )
        ids = eligible["OPID"].sample(n=min(n, len(eligible)), seed=seed + i)

        respondents = shuffle(
            respondents.filter(pl.col("OPID").is_in(ids.implode())),
            blocks=[GEMEENTE, DATE],
            fixed={"OP", "OPID", "Jaar", "Weggeweest", "Vertrekp"},
            seed=seed,
        )
        # trip rows keep their own (unshuffled) trip columns but take the
        # shuffled respondent attributes; OP is cleared so the respondent
        # row is the only OP == 1 row, as the first trip row often is in the
        # raw file
        trips = (
            trips.filter(pl.col("OPID").is_in(ids.implode()))
            .select(["OPID"] + trip_cols)
            .join(respondents.drop("OP"), on="OPID")
            .with_columns(OP=pl.lit(0, pl.Int64))
        )
        respondents = respondents.with_columns(
            pl.lit(None, pl.Int64).alias(c) for c in trip_cols
        )

        new_ids = {old: 900001 + k for k, old in enumerate(sorted(ids))}
        out = (
            pl.concat([respondents, trips.select(respondents.columns)])
            .with_columns(OPID=pl.col("OPID").replace_strict(new_ids))
            .sort("OPID", "OP", "VerplNr", descending=[False, True, False])
        )
        dst = dst_root / str(year) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        pyreadstat.write_sav(out.to_pandas(), str(dst))
        print(f"  {year}: respondents: {len(respondents)}, trips: {len(trips)}")


def sample_om(
    src_root: Path, dst_root: Path, n: int, seed: int, min_trips: int = 0
) -> None:
    """Write `n` anonymised households per OM/MON year.

    With `min_trips`, only households where someone made at least that many
    regular trips are sampled. Household attributes are shuffled across
    households and person attributes across persons; the diary date is
    shuffled as a household attribute, so all members get the same day.
    """
    from foundata.om import DATA_FILES, read_sav

    hh_cols, person_cols, trip_cols = loader_columns("om")
    person_cols = [c for c in person_cols if c not in hh_cols]
    date = ["Jaar"] + DATE
    trip_cols = [c for c in trip_cols if c not in hh_cols + person_cols]
    for i, (year, name) in enumerate(DATA_FILES.items()):
        data = read_sav(
            src_root / str(year) / name,
            ["Huishoud", "Persoon"] + hh_cols + person_cols + trip_cols,
        )
        persons = data.filter(pl.col("Persoon") == 1)
        trips = data.filter(pl.col("Verpl") >= 1)

        regular = trips.filter(pl.col("Verpl") == 1).group_by("PersID").len()
        eligible = (
            persons.join(regular, on="PersID", how="left")
            .group_by("HHID")
            .agg(pl.col("len").fill_null(0).max())
            .filter(pl.col("len") >= min_trips)
        )
        ids = eligible["HHID"].sample(n=min(n, len(eligible)), seed=seed + i)
        persons = persons.filter(pl.col("HHID").is_in(ids.implode()))
        trips = trips.filter(pl.col("HHID").is_in(ids.implode()))

        hhs = shuffle(
            persons.filter(pl.col("Huishoud") == 1).select(hh_cols + date),
            blocks=[GEMEENTE, DATE],
            fixed={"HHID", "AantPers", "Jaar"},
            seed=seed,
        )
        persons = shuffle(
            persons.select(person_cols).drop(date),
            blocks=[],
            fixed={"PersID", "Responsc"},
            seed=seed,
        ).with_columns(
            HHID=pl.col("PersID") // 100,
            Persoon=pl.lit(1, pl.Int64),
        )
        persons = persons.join(hhs, on="HHID").with_columns(
            Huishoud=(
                pl.col("PersID") == pl.col("PersID").min().over("HHID")
            ).cast(pl.Int64)
        )
        # trip rows keep their own (unshuffled) trip columns but take the
        # shuffled person and household attributes; Persoon and Huishoud
        # are cleared so the person row is the only Persoon == 1 row
        trips = (
            trips.select(["PersID"] + trip_cols)
            .join(persons.drop("Persoon", "Huishoud"), on="PersID")
            .with_columns(
                Persoon=pl.lit(0, pl.Int64), Huishoud=pl.lit(0, pl.Int64)
            )
        )
        persons = persons.with_columns(
            pl.lit(None, pl.Int64).alias(c) for c in trip_cols
        )

        new_hh = {old: 900001 + k for k, old in enumerate(sorted(ids))}
        out = (
            pl.concat([persons, trips.select(persons.columns)])
            .with_columns(
                PersID=pl.col("HHID").replace_strict(new_hh) * 100
                + pl.col("PersID") % 100,
                HHID=pl.col("HHID").replace_strict(new_hh),
            )
            .sort(
                "HHID",
                "PersID",
                "Persoon",
                "VerplNr",
                descending=[False, False, True, False],
            )
        )
        dst = dst_root / str(year) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        pyreadstat.write_sav(out.to_pandas(), str(dst))
        print(
            f"  {year}: households: {len(hhs)}, persons: {len(persons)}, "
            f"trips: {len(trips)}"
        )
