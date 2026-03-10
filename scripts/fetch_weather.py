#!/usr/bin/env python3
"""Fetch historical daily weather from Open-Meteo and write a CSV.

Usage:
    uv run python scripts/fetch_weather.py
    uv run python scripts/fetch_weather.py --start 2017-01-01 --end 2019-12-31
    uv run python scripts/fetch_weather.py --lat 37.5665 --lon 126.9780 \
        --start 2021-01-01 --end 2021-12-31 --out configs/ktdb/weather_seoul.csv
    uv run python scripts/fetch_weather.py --regions-csv configs/ktdb/regions.csv \
        --start 2021-01-01 --end 2021-12-31 --out configs/ktdb/weather_regions.csv
"""

import csv
import json
import subprocess
from pathlib import Path

import click

# Chicago city-centre coordinates (covers CMAP metro area)
CHICAGO_LAT = 41.85
CHICAGO_LON = -87.65

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "configs" / "cmap" / "weather_chicago.csv"
)


def fetch_daily(lat: float, lon: float, start: str, end: str) -> tuple[list, list, list]:
    """Fetch daily weather from Open-Meteo. Returns (dates, temps, precip)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=auto"
    )
    result = subprocess.check_output(["curl", "-s", "--max-time", "30", url])
    data = json.loads(result)
    return (
        data["daily"]["time"],
        data["daily"]["temperature_2m_max"],
        data["daily"]["precipitation_sum"],
    )


@click.command()
@click.option("--start", default="2017-01-01", show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default="2019-12-31", show_default=True, help="End date (YYYY-MM-DD)")
@click.option("--lat", default=CHICAGO_LAT, show_default=True, type=float, help="Latitude")
@click.option("--lon", default=CHICAGO_LON, show_default=True, type=float, help="Longitude")
@click.option(
    "--out",
    default=str(DEFAULT_OUT),
    show_default=True,
    type=click.Path(),
    help="Output CSV path",
)
@click.option(
    "--regions-csv",
    default=None,
    type=click.Path(exists=True),
    help="CSV with region_code,region_name,lat,lon columns; fetches all regions and writes combined output",
)
def main(start: str, end: str, lat: float, lon: float, out: str, regions_csv: str | None):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if regions_csv:
        # Multi-region mode: read regions CSV, fetch each, write combined output
        with open(regions_csv) as f:
            regions = list(csv.DictReader(f))

        print(f"Fetching weather for {len(regions)} regions ({start} → {end})...")
        with open(out_path, "w") as f:
            f.write("date,region_code,max_temp_c,precipitation_mm\n")
            total_rows = 0
            for region in regions:
                code = region["region_code"]
                name = region["region_name"]
                rlat = float(region["lat"])
                rlon = float(region["lon"])
                print(f"  {name} ({code})...")
                dates, temps, precip = fetch_daily(rlat, rlon, start, end)
                for date, temp, pr in zip(dates, temps, precip):
                    temp_str = "" if temp is None else str(temp)
                    pr_str = "" if pr is None else str(pr)
                    f.write(f"{date},{code},{temp_str},{pr_str}\n")
                total_rows += len(dates)
        print(f"Written {total_rows} rows to {out_path}")
    else:
        # Single-station mode (original behaviour)
        print(f"Fetching weather data from Open-Meteo ({start} → {end})...")
        dates, temps, precip = fetch_daily(lat, lon, start, end)

        with open(out_path, "w") as f:
            f.write("date,max_temp_c,precipitation_mm\n")
            for date, temp, pr in zip(dates, temps, precip):
                temp_str = "" if temp is None else str(temp)
                pr_str = "" if pr is None else str(pr)
                f.write(f"{date},{temp_str},{pr_str}\n")

        print(f"Written {len(dates)} rows to {out_path}")


if __name__ == "__main__":
    main()
