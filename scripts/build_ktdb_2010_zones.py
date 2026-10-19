"""Build the zone tables for the KTDB 2010 household and weekend surveys.

The 2010 surveys code home locations as "SS-GGG-DDD" (16 시도, 251 시군구
zones, last 3 digits of the 읍면동 code) and, in the national trip files, trip
ends as the 3-digit 251-zone number only. This script writes:

- configs/ktdb/zones_2010.csv: one row per 읍면동 (code = SSGGGDDD), with
  names, from the classification sheet shipped with the survey.
- configs/ktdb/zone_2010_sigungu.csv: each 251 zone's KTDB 시도 + 시군구
  code (SSGGG; one per zone), so trip ends resolve against the sigungu_2010
  location lookup (scripts/build_location_lookups.py), whose points also give
  2010 trip distances.

Usage:
    uv run --group scripts python scripts/build_ktdb_2010_zones.py \\
        --classification "/path/to/가구통행실태조사 분류 번호 및 행정코드.xls"
"""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

OUT_DIR = Path(__file__).parent.parent / "configs/ktdb"


def main(
    classification: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="가구통행실태조사 분류 번호 및 행정코드.xls from the 2010 download",
        ),
    ],
    out_dir: Path = OUT_DIR,
):
    """Write 2010 KTDB zones and their 시군구 codes to configs/ktdb."""
    out_dir.mkdir(parents=True, exist_ok=True)

    zones = (
        pl.read_excel(classification, infer_schema_length=0)
        .rename(
            {
                "시도": "sido",
                "시군구": "sigungu",
                "읍면동": "dong",
                "분류코드": "code",
                "행정구역 코드": "adm_cd",
            }
        )
        .select("code", "sido", "sigungu", "dong", "adm_cd")
        .with_columns(
            code=pl.col("code").str.replace_all("-", ""),
            zone=pl.col("code").str.slice(3, 3),
        )
        .sort("code")
    )
    zones.drop("adm_cd").write_csv(out_dir / "zones_2010.csv")
    print(f"Written {zones.height} 읍면동 to {out_dir / 'zones_2010.csv'}")

    sigungu = (
        zones.select("zone", sgg=pl.col("code").str.slice(0, 5))
        .unique()
        .sort("zone")
    )
    if sigungu["zone"].n_unique() != sigungu.height:
        raise ValueError("a 251 zone spans several 시군구 codes")
    sigungu = sigungu.select(**{"from": "zone", "to": "sgg"})
    sigungu.write_csv(out_dir / "zone_2010_sigungu.csv")
    print(
        f"Written {sigungu.height} zones to {out_dir / 'zone_2010_sigungu.csv'}"
    )


if __name__ == "__main__":
    typer.run(main)
