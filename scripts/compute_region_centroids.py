"""Compute ktdb region centroids from zone centroids.

Region code = first 2 characters of the 10-digit admin dong code.
Centroid = bounding-box centre of all dong centroids in the region.

Usage:
    uv run python scripts/compute_region_centroids.py
"""

import csv
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "ktdb"
ZONE_CSV = CONFIGS / "zone_centroids.csv"
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
    # Accumulate lat/lon bounds per region
    bounds: dict[str, dict] = {}
    with open(ZONE_CSV) as f:
        for row in csv.DictReader(f):
            code = str(row["zone"])[:2]
            lat = float(row["lat"])
            lon = float(row["lon"])
            if code not in bounds:
                bounds[code] = {"min_lat": lat, "max_lat": lat, "min_lon": lon, "max_lon": lon}
            else:
                b = bounds[code]
                b["min_lat"] = min(b["min_lat"], lat)
                b["max_lat"] = max(b["max_lat"], lat)
                b["min_lon"] = min(b["min_lon"], lon)
                b["max_lon"] = max(b["max_lon"], lon)

    rows = []
    for code in sorted(bounds):
        b = bounds[code]
        lat = (b["min_lat"] + b["max_lat"]) / 2
        lon = (b["min_lon"] + b["max_lon"]) / 2
        name = REGION_NAMES.get(code, code)
        rows.append((code, name, round(lat, 6), round(lon, 6)))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region_code", "region_name", "lat", "lon"])
        writer.writerows(rows)

    print(f"Written {len(rows)} region centroids to {OUT_CSV}")
    for code, name, lat, lon in rows:
        print(f"  {code}  {name:<12}  {lat:.4f}  {lon:.4f}")


if __name__ == "__main__":
    main()
