"""Write samples of the KTDB 2016 and 2010 surveys' raw files.

Treated like the KTDB 2021 sample: households are sampled whole and
anonymised:

- households get fresh synthetic ids (2016: IDX and 행정구역번호; 2010: the
  4-digit serial of sheet_code, keeping its zone prefix, which the loader
  reads as the home 읍면동);
- household attributes are shuffled across the sampled households, and
  person attributes across the sampled persons. A weekend person's two
  diaries keep the same attributes, and the 2016 travel date stays with
  its diary (the date and weekday columns are shuffled as one block);
- trip rows are kept intact apart from the remapped ids, so trip
  sequencing, times and zones stay coherent.

Files are written in the raw formats and encodings, at the same relative
paths: 2016 as CP949 CSV; 2010 as tab-separated UTF-8 text, with the
weekend trips as xlsx whose numeric-looking codes are stored as numbers
(dropping leading zeros, as in the real file).
"""

from pathlib import Path

import openpyxl
import polars as pl

from foundata.utils import config_for_year, get_config_path, load_yaml_config

# 2016: columns describing the household or the person; the rest of each
# row describes the diary or the trip and stays put
HH_2016 = [
    "총가구원",
    "5세이상가구원",
    "주택종류",
    "월평균소득",
    "차량보유",
    *[f"{k}{i}" for i in range(1, 7) for k in ("종류", "연식")],
    "오토바이 50cc미만(대)",
    "오토바이 50cc이상 100cc미만(대)",
    "오토바이 100cc이상 260cc미만(대)",
    "오토바이 260cc이상(대)",
    "자전거(동력 전기)(대)",
    "자전거(비동력)(대)",
    "기 타(종류)",
    "기 타(대)",
]
PERSON_2016 = [
    "가구주관계",
    "출생연도",
    "성 별",
    "운전면허증",
    "정규교육기관",
    "학교_행정동코드",
    "직업",
    "직업기타",
    "재택근무(집)",
    "직장_행정동코드",
    "근무일수",
    "근무시간",
    "많은 통행을 하는 일",
]
DATE_2016 = ["통행일자(년)", "통행일자(월)", "통행일자(일)", "통행일자(요일)"]


def config(year: int) -> dict:
    return config_for_year(
        load_yaml_config(get_config_path("ktdb", "person_dictionary.yaml")),
        year,
    )


def shuffle_rows(
    df: pl.DataFrame, cols: list[str], seed: int, blocks=()
) -> pl.DataFrame:
    """Shuffle each column in `cols` independently across rows, and each
    block of columns together."""
    blocked = {c for block in blocks for c in block}
    groups = [list(b) for b in blocks] + [[c] for c in cols if c not in blocked]
    out = df
    for i, group in enumerate(groups):
        order = pl.Series(range(len(df))).shuffle(seed=seed + i).to_list()
        out = out.with_columns(df.select(group)[order].get_columns())
    return out


def sample_households(
    keys: pl.DataFrame, travelled: pl.Series, n: int, seed: int
) -> pl.DataFrame:
    """Sample n households, preferring ones with at least one trip."""
    with_trips = keys.filter(travelled).unique(maintain_order=True)
    return with_trips.sample(n=min(n, len(with_trips)), seed=seed)


# ---------------------------------------------------------------------------
# 2016
# ---------------------------------------------------------------------------


def sample_ktdb_2016(src: Path, dst: Path, n: int, seed: int):
    for survey, relpath in config(2016)["files"].items():
        raw = pl.read_csv(
            src / relpath, encoding="cp949", infer_schema_length=0
        )
        header = raw.columns
        # the weekend file spells "/" where the household file has " "
        raw = raw.rename({c: c.replace("/", " ") for c in raw.columns})
        hh_key = ["IDX", "행정구역번호"]

        sampled = sample_households(
            raw.select(hh_key), raw["통행유무"] == "1", n, seed
        ).with_columns(
            new_idx=pl.int_range(1, pl.len() + 1).cast(pl.String),
            new_district=pl.format(
                "9999{}-0000-01", pl.int_range(100, pl.len() + 100)
            ),
        )
        data = raw.join(sampled, on=hh_key, how="inner", maintain_order="left")

        hh_cols = [c for c in HH_2016 if c in data.columns]
        households = shuffle_rows(
            data.select(hh_key + hh_cols).unique(hh_key, maintain_order=True),
            hh_cols,
            seed,
        )
        person_key = hh_key + ["가구원구분"]
        persons = shuffle_rows(
            data.select(person_key + PERSON_2016).unique(
                person_key, maintain_order=True
            ),
            PERSON_2016,
            seed + 100,
        )
        # each diary keeps its own date; shuffle dates across diaries
        diary_key = person_key + (["토일구분"] if "토일구분" in data else [])
        dates = shuffle_rows(
            data.select(diary_key + DATE_2016).unique(
                diary_key, maintain_order=True
            ),
            DATE_2016,
            seed + 200,
            blocks=[DATE_2016],
        )

        data = (
            data.drop(hh_cols + PERSON_2016 + DATE_2016)
            .join(households, on=hh_key, maintain_order="left")
            .join(persons, on=person_key, maintain_order="left")
            .join(dates, on=diary_key, maintain_order="left")
            .with_columns(
                IDX=pl.col("new_idx"), 행정구역번호=pl.col("new_district")
            )
            .drop("new_idx", "new_district")
        )
        data = data.select(c.replace("/", " ") for c in header)
        data.columns = header

        out = dst / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data.write_csv().encode("cp949", errors="replace"))
        print(f"  2016 {survey}: households {len(sampled)}, rows {len(data)}")


# ---------------------------------------------------------------------------
# 2010
# ---------------------------------------------------------------------------


def read_2010(path: Path) -> pl.DataFrame:
    if path.suffix == ".xlsx":
        data = pl.read_excel(path, infer_schema_length=0)
    else:
        data = pl.read_csv(
            path, separator="\t", quote_char=None, infer_schema_length=0
        )
    data = data.rename({c: c.lstrip("﻿") for c in data.columns})
    return data.with_columns(sheet_code=pl.col("sheet_code").str.zfill(12))


def write_2010(data: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".xlsx":
        path.write_text(data.write_csv(separator="\t", quote_style="never"))
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(data.columns)
    for row in data.iter_rows():
        ws.append([int(v) if v and v.isdigit() else v for v in row])
    wb.save(path)


def sample_ktdb_2010(src: Path, dst: Path, n: int, seed: int):
    cfg = config(2010)
    offset = cfg["weekend_day2_offset"]
    for survey, files in cfg["files"].items():
        hhs = read_2010(src / files["households"])
        persons = read_2010(src / files["persons"])
        trips = read_2010(src / files["trips"])
        columns = {
            "households": hhs.columns,
            "persons": persons.columns,
            "trips": trips.columns,
        }

        has_trips = hhs["sheet_code"].is_in(trips["sheet_code"].implode())
        sampled = sample_households(
            hhs.select("sheet_code"), has_trips, n, seed
        ).with_columns(
            new_code=pl.col("sheet_code").str.slice(0, 8)
            + pl.int_range(9000, 9000 + pl.len()).cast(pl.String)
        )
        codes = dict(zip(sampled["sheet_code"], sampled["new_code"]))

        def remap(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(
                pl.col("sheet_code").is_in(sampled["sheet_code"].implode())
            ).with_columns(sheet_code=pl.col("sheet_code").replace(codes))

        hhs, persons, trips = remap(hhs), remap(persons), remap(trips)
        hhs = shuffle_rows(
            hhs,
            [c for c in hhs.columns if c not in ("sheet_code", "sheet_type")],
            seed,
        )

        # weekend members above `offset` are day-2 diaries of member - offset:
        # shuffle attributes per member so both diaries keep the same ones
        person_cols = ["relation", "birth_year", "sex", "drive_license"]
        person_cols += ["job_type", "employee_type", "ju5_il", "dongger"]
        member = pl.col("seq").cast(pl.Int32)
        if survey == "weekend":
            member = (
                pl.when(member > offset).then(member - offset).otherwise(member)
            )
        persons = persons.with_columns(_member=member)
        attrs = persons.select(["sheet_code", "_member"] + person_cols).unique(
            ["sheet_code", "_member"], maintain_order=True
        )
        attrs = shuffle_rows(attrs, person_cols, seed + 100)
        persons = (
            persons.drop(person_cols)
            .join(attrs, on=["sheet_code", "_member"], maintain_order="left")
            .drop("_member")
        )

        for name, df in [
            ("households", hhs),
            ("persons", persons),
            ("trips", trips),
        ]:
            write_2010(df.select(columns[name]), dst / files[name])
        print(
            f"  2010 {survey}: households {len(hhs)}, persons {len(persons)}, "
            f"legs {len(trips)}"
        )
