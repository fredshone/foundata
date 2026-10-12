"""Write samples of SPSS .sav tables (the arg source's raw format).

The Argentine surveys are open data, so the sampled rows are written with
their real values, all columns, and the original variable and value labels.
"""

import tempfile
from pathlib import Path

import pyreadstat


def read_full(path: Path):
    """All columns and metadata of a .sav file (see arg.read_sav for the
    long-variable-name workaround)."""
    from foundata.arg import ascii_long_names

    try:
        df, meta = pyreadstat.read_sav(str(path))
    except pyreadstat.ReadstatError:
        # CH-CTES_MODOS.sav has latin-1 strings in a file declared UTF-8
        df, meta = pyreadstat.read_sav(str(path), encoding="latin1")
    except UnicodeDecodeError:
        df, meta = None, None
    if df is None or len(df.columns) < len(meta.column_names):
        with tempfile.TemporaryDirectory() as tmp:
            patched = Path(tmp) / "patched.sav"
            patched.write_bytes(ascii_long_names(path.read_bytes()))
            df, meta = pyreadstat.read_sav(str(patched))
    return df, meta


def write_subset(df, meta, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    labels = meta.column_names_to_labels
    pyreadstat.write_sav(
        df,
        str(dst),
        column_labels=[labels.get(c) for c in df.columns],
        variable_value_labels={
            c: v
            for c, v in meta.variable_value_labels.items()
            if c in df.columns
        },
    )


def raw_column(column_mapping: dict, target: str) -> str:
    """The raw column renamed to `target`."""
    return next(raw for raw, new in column_mapping.items() if new == target)


def sample_arg(src_root: Path, dst_root: Path, n: int, seed: int) -> None:
    """Write `n` households per arg survey across its four tables."""
    from foundata.arg import SURVEYS
    from foundata.utils import get_config_path, load_yaml_config

    configs = get_config_path() / "arg"
    hh_cfg, person_cfg, trip_cfg = (
        load_yaml_config(configs / f"{name}_dictionary.yaml")["column_mappings"]
        for name in ("hh", "person", "trip")
    )
    for survey, paths in SURVEYS.items():
        hh_id = raw_column(hh_cfg[survey], "hid")
        person_hh = raw_column(person_cfg[survey], "hid")
        person_id = raw_column(person_cfg[survey], "pid")
        trip_person = raw_column(trip_cfg[survey]["trips"], "pid")
        trip_id = raw_column(trip_cfg[survey]["trips"], "tid")
        stage_trip = raw_column(trip_cfg[survey]["stages"], "tid")

        hhs, hh_meta = read_full(src_root / paths["hh"])
        hhs = hhs.sample(n=min(n, len(hhs)), random_state=seed)
        persons, p_meta = read_full(src_root / paths["persons"])
        persons = persons[persons[person_hh].isin(hhs[hh_id])]
        trips, t_meta = read_full(src_root / paths["trips"])
        trips = trips[trips[trip_person].isin(persons[person_id])]
        stages, s_meta = read_full(src_root / paths["stages"])
        stages = stages[stages[stage_trip].isin(trips[trip_id])]

        for table, df, meta in (
            ("hh", hhs, hh_meta),
            ("persons", persons, p_meta),
            ("trips", trips, t_meta),
            ("stages", stages, s_meta),
        ):
            write_subset(df, meta, dst_root / paths[table])
        print(
            f"  {survey}: households {len(hhs)}, persons {len(persons)}, "
            f"trips {len(trips)}, stages {len(stages)}"
        )
