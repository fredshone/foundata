#!/usr/bin/env python3
"""Run the full foundata pipeline and save outputs."""

from pathlib import Path

import click
import polars as pl

from foundata import (
    cmap,
    filter,
    fix,
    ltds,
    nhts,
    nts,
    post_process,
    qhts,
    verify,
    vista,
    viz,
)
from foundata.utils import check_overlap, load_yaml_config

CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"


@click.command()
@click.option(
    "--data-root",
    required=True,
    type=click.Path(),
    help="Base data directory, e.g. ~/Data/foundata",
)
@click.option(
    "--output",
    default="~/Data",
    show_default=True,
    type=click.Path(),
    help="Directory where CSVs and PNGs are written",
)
def main(data_root: str, output: str):
    data_root = Path(data_root).expanduser()
    output = Path(output).expanduser()
    output.mkdir(exist_ok=True, parents=True)

    all_attributes = []
    all_trips = []

    # ------------------------------------------------------------------
    # LTDS
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "ltds" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "ltds" / "person_dictionary.yaml"
    )
    person_data_config = load_yaml_config(
        CONFIGS_ROOT / "ltds" / "person_data_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "ltds" / "trip_dictionary.yaml"
    )
    stages_config = load_yaml_config(
        CONFIGS_ROOT / "ltds" / "stage_dictionary.yaml"
    )

    attributes, trips = ltds.load_years(
        years=["LTDS2425", "LTDS2324", "LTDS2223", "LTDS1920"],
        data_root=data_root / "LTDS",
        hh_config=hh_config,
        person_config=person_config,
        person_data_config=person_data_config,
        trips_config=trips_config,
        stages_config=stages_config,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(
        attributes, trips, on="pid", lhs_name="attributes", rhs_name="trips"
    )
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # VISTA
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "vista" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "vista" / "person_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "vista" / "trip_dictionary.yaml"
    )

    attributes, trips = vista.load_years(
        years=["2012-2020", "2022-2023", "2023-2024"],
        data_root=data_root / "VISTA",
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(attributes, trips, on="pid")
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # QHTS
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "qhts" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "qhts" / "person_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "qhts" / "trip_dictionary.yaml"
    )
    zone_mapping = qhts.load_zone_mapping(
        CONFIGS_ROOT / "qhts" / "sa1-correspondence-file.csv"
    )

    attributes, trips = qhts.load_years(
        data_root=data_root / "QHTS",
        years=["2019-22", "2022-25"],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
        zones_mapping=zone_mapping,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(attributes, trips, on="pid")
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # CMAP
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "cmap" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "cmap" / "person_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "cmap" / "trip_dictionary.yaml"
    )

    attributes, trips = cmap.load(
        data_root=data_root / "CMAP",
        configs_root=CONFIGS_ROOT,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(attributes, trips, on="pid")
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # NHTS
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "nhts" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "nhts" / "person_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "nhts" / "trip_dictionary.yaml"
    )

    attributes, trips = nhts.load(
        data_root=data_root / "NHTS",
        years=[2022, 2017, 2009, 2001],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(attributes, trips, on="pid")
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # NTS
    # ------------------------------------------------------------------
    hh_config = load_yaml_config(CONFIGS_ROOT / "nts" / "hh_dictionary.yaml")
    person_config = load_yaml_config(
        CONFIGS_ROOT / "nts" / "person_dictionary.yaml"
    )
    trips_config = load_yaml_config(
        CONFIGS_ROOT / "nts" / "trip_dictionary.yaml"
    )
    days_config = load_yaml_config(CONFIGS_ROOT / "nts" / "day_dictionary.yaml")

    attributes, trips = nts.load(
        data_root=data_root / "NTS",
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
        days_config=days_config,
    )
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    check_overlap(attributes, trips, on="pid")
    verify.columns(attributes, trips)
    verify.null_pids(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips"
    )
    all_attributes.append(attributes)
    all_trips.append(trips)

    # ------------------------------------------------------------------
    # Concat and write
    # ------------------------------------------------------------------
    all_attributes = pl.concat(all_attributes, how="vertical")
    all_trips = fix.trip_dtypes(all_trips)
    all_trips = pl.concat(all_trips, how="vertical")

    all_attributes.write_csv(output / "all_attributes.csv")
    all_trips.write_csv(output / "all_trips.csv")

    activities = post_process.trips_to_activities(all_attributes, all_trips)
    activities.write_csv(output / "activities.csv")

    trips_with_acts = post_process.trips_with_following_activity(all_attributes, all_trips)
    trips_with_acts.write_csv(output / "trips_with_activities.csv")

    print(f"Written to {output}")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    print(viz.summary_table(all_attributes, all_trips))

    viz.plot_numeric_hist_grid(
        all_attributes,
        on="source",
        cmap_name="Dark2",
        n_cols=3,
        bins=11,
        linewidth=2,
        density=True,
        ignore_cols={"weight"},
        tail_handling="clip",
        tail_ratio_threshold=4,
        outlier_share_max=0.2,
        clip_percentiles=(1.0, 99.0),
        min_unique=5,
        min_group_rows=10,
        verbose=True,
        save_path=output / "attributes_numeric.png",
    )

    viz.plot_categorical_bar_grid(
        all_attributes,
        on="source",
        save_path=output / "attributes_categorical.png",
    )

    print(f"Figures saved to {output}")


if __name__ == "__main__":
    main()
