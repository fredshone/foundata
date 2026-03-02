#!/usr/bin/env python3
"""Fetch historical daily weather for Chicago from Open-Meteo and write a CSV.

Usage:
    uv run python scripts/fetch_weather.py
    uv run python scripts/fetch_weather.py --start 2017-01-01 --end 2019-12-31
"""

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


@click.command()
@click.option("--start", default="2017-01-01", show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default="2019-12-31", show_default=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--out",
    default=str(DEFAULT_OUT),
    show_default=True,
    type=click.Path(),
    help="Output CSV path",
)
def main(start: str, end: str, out: str):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={CHICAGO_LAT}&longitude={CHICAGO_LON}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=America%2FChicago"
    )

    print(f"Fetching weather data from Open-Meteo ({start} → {end})...")
    result = subprocess.check_output(
        ["curl", "-s", "--max-time", "30", url]
    )
    data = json.loads(result)

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_max"]
    precip = data["daily"]["precipitation_sum"]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("date,max_temp_c,precipitation_mm\n")
        for date, temp, pr in zip(dates, temps, precip):
            temp_str = "" if temp is None else str(temp)
            pr_str = "" if pr is None else str(pr)
            f.write(f"{date},{temp_str},{pr_str}\n")

    print(f"Written {len(dates)} rows to {out_path}")


if __name__ == "__main__":
    main()
