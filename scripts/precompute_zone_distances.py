"""Precompute pairwise haversine distances between all ktdb zone centroids.

Writes the upper triangle (including diagonal) as a long-format Parquet file
(zstd compressed):
    ozone, dzone, distance_km

where ozone <= dzone (string comparison).  dist(a,b) == dist(b,a), so callers
normalise the zone pair before lookup.

Usage:
    uv run python scripts/precompute_zone_distances.py
    uv run python scripts/precompute_zone_distances.py --centroids path/to/centroids.csv --out path/to/out.parquet
"""

from pathlib import Path

import numpy as np
import polars as pl
import typer

DEFAULT_CENTROIDS = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "ktdb"
    / "zone_centroids.csv"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "ktdb"
    / "zone_distances.parquet"
)


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return N×N distance matrix (km, float32) for the given lat/lon arrays."""
    R = 6371.0
    lat_r = np.radians(lat[:, None])  # (N, 1)
    lon_r = np.radians(lon[:, None])
    lat_r2 = np.radians(lat[None, :])  # (1, N)
    lon_r2 = np.radians(lon[None, :])

    dlat = lat_r2 - lat_r
    dlon = lon_r2 - lon_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r) * np.cos(lat_r2) * np.sin(dlon / 2) ** 2
    )
    return (2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)


def main(
    centroids_path: Path = typer.Option(DEFAULT_CENTROIDS, "--centroids"),
    out_path: Path = typer.Option(DEFAULT_OUT, "--out"),
) -> None:
    """Write pairwise distances between KTDB zone centroids."""
    # sort so the upper triangle really has ozone <= dzone
    df = pl.read_csv(centroids_path, schema_overrides={"zone": pl.String}).sort(
        "zone"
    )
    zones = df["zone"].to_list()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    n = len(zones)
    print(f"Loaded {n} zone centroids from {centroids_path}")

    print("Computing distance matrix...")
    dist = haversine_matrix(lat, lon)  # (N, N) float32

    # Upper triangle indices (including diagonal)
    i_idx, j_idx = np.triu_indices(n)

    oz = [zones[i] for i in i_idx]
    dz = [zones[j] for j in j_idx]
    d = dist[i_idx, j_idx]

    out_df = pl.DataFrame(
        {"ozone": oz, "dzone": dz, "distance_km": d},
        schema={
            "ozone": pl.String,
            "dzone": pl.String,
            "distance_km": pl.Float32,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path, compression="zstd")
    print(f"Written {len(out_df):,} rows to {out_path}")


if __name__ == "__main__":
    typer.run(main)
