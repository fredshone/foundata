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
import time
from pathlib import Path
from typing import Optional

import typer

# Chicago city-centre coordinates (covers CMAP metro area)
CHICAGO_LAT = 41.85
CHICAGO_LON = -87.65

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "cmap"
    / "weather_chicago.csv"
)


def fetch_batch(
    coords: list[tuple[float, float]], start: str, end: str, retries: int = 4
) -> list[tuple[list, list, list]]:
    """Fetch daily weather for multiple locations in one Open-Meteo request.

    Open-Meteo accepts comma-separated latitude/longitude lists and returns
    one result object per location (in the same order). Batching keeps the
    total request count well under Open-Meteo's per-minute/per-hour quotas
    for large region lists.

    Returns a list of (dates, temps, precip), one per input coordinate.

    Retries on rate-limit/transient errors (Open-Meteo returns an error JSON
    body, not a non-zero curl exit, when throttled). Open-Meteo's quota
    resets on a rolling window, so back off a full 65s rather than a short
    exponential delay.
    """
    lats = ",".join(str(lat) for lat, _ in coords)
    lons = ",".join(str(lon) for _, lon in coords)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lats}&longitude={lons}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=auto"
    )
    for attempt in range(retries):
        result = subprocess.check_output(
            ["curl", "-s", "--max-time", "60", url]
        )
        data = json.loads(result)
        # A single location returns one object; multiple locations return a list.
        results = data if isinstance(data, list) else [data]
        if all("daily" in r for r in results):
            return [
                (
                    r["daily"]["time"],
                    r["daily"]["temperature_2m_max"],
                    r["daily"]["precipitation_sum"],
                )
                for r in results
            ]
        reason = next((r.get("reason") for r in results if "reason" in r), data)
        print(f"    retrying in 65s ({reason})...")
        time.sleep(65)
    raise RuntimeError(
        f"Open-Meteo request failed after {retries} attempts: {url}"
    )


def fetch_daily(
    lat: float, lon: float, start: str, end: str
) -> tuple[list, list, list]:
    """Fetch daily weather for a single location. Returns (dates, temps, precip)."""
    return fetch_batch([(lat, lon)], start, end)[0]


def main(
    start: str = typer.Option("2017-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2019-12-31", help="End date (YYYY-MM-DD)"),
    lat: float = typer.Option(CHICAGO_LAT, help="Latitude"),
    lon: float = typer.Option(CHICAGO_LON, help="Longitude"),
    out: Path = typer.Option(DEFAULT_OUT, help="Output CSV path"),
    regions_csv: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="CSV with region_code,region_name,lat,lon columns; fetches all regions and writes combined output",
    ),
    batch_size: int = typer.Option(
        10,
        help="Number of regions to fetch per Open-Meteo request (stays under rate limits)",
    ),
):
    """Fetch daily max temperature and precipitation from Open-Meteo."""
    out_path = out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if regions_csv:
        # Multi-region mode: read regions CSV, fetch in batches, write combined output
        with open(regions_csv) as f:
            regions = list(csv.DictReader(f))

        # Resume support: skip regions already written by a prior (rate-limited)
        # run rather than re-fetching and re-burning quota on them.
        done_codes: set[str] = set()
        if out_path.exists():
            with open(out_path) as f:
                done_codes = {row["region_code"] for row in csv.DictReader(f)}
            if done_codes:
                print(
                    f"Resuming: {len(done_codes)} regions already fetched, skipping"
                )
                regions = [
                    r for r in regions if r["region_code"] not in done_codes
                ]

        print(
            f"Fetching weather for {len(regions)} regions in batches of "
            f"{batch_size} ({start} → {end})..."
        )
        write_mode = "a" if done_codes else "w"
        with open(out_path, write_mode) as f:
            if not done_codes:
                f.write("date,region_code,max_temp_c,precipitation_mm\n")
            total_rows = 0
            for i in range(0, len(regions), batch_size):
                batch = regions[i : i + batch_size]
                names = ", ".join(
                    f"{r['region_name']} ({r['region_code']})" for r in batch
                )
                print(f"  {names}...")
                coords = [(float(r["lat"]), float(r["lon"])) for r in batch]
                results = fetch_batch(coords, start, end)
                for region, (dates, temps, precip) in zip(batch, results):
                    code = region["region_code"]
                    for date, temp, pr in zip(dates, temps, precip):
                        temp_str = "" if temp is None else str(temp)
                        pr_str = "" if pr is None else str(pr)
                        f.write(f"{date},{code},{temp_str},{pr_str}\n")
                    total_rows += len(dates)
                time.sleep(2.5)
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
    typer.run(main)
