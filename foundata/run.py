#!/usr/bin/env python3
"""Run the full foundata pipeline and save outputs."""

from pathlib import Path

import polars as pl

from foundata import (
    cmap,
    filter,
    fix,
    ktdb,
    ltds,
    nhts,
    nts,
    post_process,
    qhts,
    utils,
    verify,
    vista,
    viz,
)

CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"


def process_source(attributes, trips, source_name):
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    utils.check_overlap(
        attributes, trips, on="pid", lhs_name="attributes", rhs_name="trips"
    )
    attributes, trips = filter.missing_acts_or_modes(attributes, trips)
    attributes, trips = fix.missing_columns(attributes, trips)
    verify.columns(attributes, trips)
    attributes, trips = filter.columns(attributes, trips)
    attributes, trips = fix.fix_types(attributes, trips)
    attributes = fix.unknown_to_null(attributes)
    attributes = utils.norm_weights(attributes)
    print(
        f"Loaded {len(attributes)} persons, "
        f"{len(trips.select(pl.col('pid').unique()))} plans, "
        f"{len(trips)} trips from {source_name}"
    )
    return attributes, trips


def runner(
    data_root: str,
    output: str,
    select: list[str],
    omit: list[str],
    home_based: bool = False,
):
    data_root = Path(data_root).expanduser()
    output = Path(output).expanduser()
    output.mkdir(exist_ok=True, parents=True)

    sources = {"ltds", "vista", "qhts", "cmap", "nhts", "nts", "ktdb"}
    if select:
        sources = set(select)
    if omit:
        sources -= set(omit)
    if not sources:
        print("No sources selected. Exiting.")
        return

    print(f"Selected sources: {', '.join(sources)}")

    all_attributes = []
    all_trips = []

    # ------------------------------------------------------------------
    # KTDB
    # ------------------------------------------------------------------
    if "ktdb" in sources:
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ktdb" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ktdb" / "trip_dictionary.yaml"
        )

        attributes, trips = ktdb.load(
            data_root=data_root / "KTDB",
            person_config=person_config,
            trips_config=trips_config,
        )
        attributes, trips = process_source(attributes, trips, "KTDB")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # LTDS
    # ------------------------------------------------------------------
    if "ltds" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ltds" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ltds" / "person_dictionary.yaml"
        )
        person_data_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ltds" / "person_data_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "ltds" / "trip_dictionary.yaml"
        )
        stages_config = utils.load_yaml_config(
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
        attributes, trips = process_source(attributes, trips, "LTDS")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # VISTA
    # ------------------------------------------------------------------
    if "vista" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "vista" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "vista" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "vista" / "trip_dictionary.yaml"
        )

        attributes, trips = vista.load_years(
            years=["2012-2020", "2022-2023", "2023-2024"],
            data_root=data_root / "VISTA",
            hh_config=hh_config,
            person_config=person_config,
            trips_config=trips_config,
        )
        attributes, trips = process_source(attributes, trips, "VISTA")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # QHTS
    # ------------------------------------------------------------------
    if "qhts" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "qhts" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "qhts" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
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
        attributes, trips = process_source(attributes, trips, "QHTS")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # CMAP
    # ------------------------------------------------------------------
    if "cmap" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "cmap" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "cmap" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "cmap" / "trip_dictionary.yaml"
        )

        attributes, trips = cmap.load(
            data_root=data_root / "CMAP",
            configs_root=CONFIGS_ROOT,
            hh_config=hh_config,
            person_config=person_config,
            trips_config=trips_config,
        )
        attributes, trips = process_source(attributes, trips, "CMAP")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # NHTS
    # ------------------------------------------------------------------
    if "nhts" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nhts" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nhts" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nhts" / "trip_dictionary.yaml"
        )

        attributes, trips = nhts.load(
            data_root=data_root / "NHTS",
            years=[2022, 2017, 2009, 2001],
            hh_config=hh_config,
            person_config=person_config,
            trips_config=trips_config,
        )
        attributes, trips = process_source(attributes, trips, "NHTS")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # NTS
    # ------------------------------------------------------------------
    if "nts" in sources:
        hh_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nts" / "hh_dictionary.yaml"
        )
        person_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nts" / "person_dictionary.yaml"
        )
        trips_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nts" / "trip_dictionary.yaml"
        )
        stages_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nts" / "stage_dictionary.yaml"
        )
        days_config = utils.load_yaml_config(
            CONFIGS_ROOT / "nts" / "day_dictionary.yaml"
        )

        attributes, trips = nts.load(
            data_root=data_root / "NTS",
            hh_config=hh_config,
            person_config=person_config,
            trips_config=trips_config,
            stages_config=stages_config,
            days_config=days_config,
        )
        attributes, trips = process_source(attributes, trips, "NTS")
        all_attributes.append(attributes)
        all_trips.append(trips)

    # ------------------------------------------------------------------
    # Concat and write
    # ------------------------------------------------------------------

    all_attributes = pl.concat(all_attributes, how="vertical")
    all_trips = pl.concat(all_trips, how="vertical")

    all_attributes.write_csv(output / "attributes.csv")
    binned_attributes = post_process.discretise_numeric(
        all_attributes,
        n_bins=5,
        method="quantile",
        exclude_cols=["year", "month", "weight", "vehicles", "hh_size"],
    )
    binned_attributes = post_process.fill_nulls(binned_attributes)
    binned_attributes.write_csv(output / "binned_attributes.csv")
    all_trips.write_csv(output / "trips.csv")

    activities = post_process.trips_to_activities(all_attributes, all_trips)
    activities.write_csv(output / "activities.csv")

    # home based variations
    if home_based:
        hb_output = output / "home_based"
        hb_output.mkdir(exist_ok=True, parents=True)

        home_based_attributes, home_based_trips = filter.home_based(
            all_attributes, all_trips
        )
        home_based_attributes.write_csv(hb_output / "attributes.csv")
        home_based_trips.write_csv(hb_output / "trips.csv")

        binned_home_based_attributes = post_process.discretise_numeric(
            home_based_attributes,
            n_bins=5,
            method="quantile",
            exclude_cols=["year", "month", "weight", "vehicles", "hh_size"],
        )
        binned_home_based_attributes = post_process.fill_nulls(
            binned_home_based_attributes
        )
        binned_home_based_attributes.write_csv(
            hb_output / "binned_attributes.csv"
        )

        home_based_activities = post_process.trips_to_activities(
            home_based_attributes, home_based_trips
        )
        home_based_activities.write_csv(hb_output / "activities.csv")

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

    viz.plot_summary_trends(
        all_attributes,
        on="source",
        cmap_name="Dark2",
        save_path=output / "attributes_trends.png",
    )

    print(f"Figures saved to {output}")
