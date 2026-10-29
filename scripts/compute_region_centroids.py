"""Compute ktdb region centroids from the 2021 dong location lookup.

Region code = first 2 characters of the 10-digit admin dong code.
Centroid = bounding-box centre of all dong points in the region.

Usage:
    uv run python scripts/compute_region_centroids.py
"""

import csv
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = ROOT / "ktdb"
ZONES = ROOT / "ktdb" / "geo" / "dong_2021.parquet"
OUT_CSV = CONFIGS / "region_centroids.csv"

# 2021 Korean si/do admin codes → English names
REGION_NAMES = {
    "11": "Seoul",
    "26": "Busan",
    "27": "Daegu",
    "28": "Incheon",
    "29": "Gwangju",
    "30": "Daejeon",
    "31": "Ulsan",
    "36": "Sejong",
    "41": "Gyeonggi",
    "42": "Gangwon",
    "43": "Chungbuk",
    "44": "Chungnam",
    "45": "Jeonbuk",
    "46": "Jeonnam",
    "47": "Gyeongbuk",
    "48": "Gyeongnam",
    "50": "Jeju",
}


def main():
    # lat/lon bounds of each region's dong points
    zones = pl.read_parquet(ZONES, columns=["code", "lat", "lon"])
    bounds = {
        row["region"]: row
        for row in zones.group_by(region=pl.col("code").str.slice(0, 2))
        .agg(
            min_lat=pl.col("lat").min(),
            max_lat=pl.col("lat").max(),
            min_lon=pl.col("lon").min(),
            max_lon=pl.col("lon").max(),
        )
        .iter_rows(named=True)
    }

    rows = []
    for code in sorted(bounds):
        b = bounds[code]
        lat = (b["min_lat"] + b["max_lat"]) / 2
        lon = (b["min_lon"] + b["max_lon"]) / 2
        name = REGION_NAMES.get(code, code)
        rows.append((code, name, round(lat, 6), round(lon, 6)))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_code", "region_name", "lat", "lon"])
        writer.writerows(rows)

    print(f"Written {len(rows)} region centroids to {OUT_CSV}")
    for code, name, lat, lon in rows:
        print(f"  {code}  {name:<12}  {lat:.4f}  {lon:.4f}")


if __name__ == "__main__":
    main()
