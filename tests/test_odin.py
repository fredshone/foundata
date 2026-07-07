import polars as pl

from foundata import odin
from foundata.utils import get_config_path

CONFIGS_ROOT = get_config_path()

STED_MAPPING = {
    "1": "urban",
    "2": "urban",
    "3": "suburban",
    "4": "rural",
    "5": "rural",
}


def test_load_gemeente_zone():
    gemeente_zone = odin.load_gemeente_zone(CONFIGS_ROOT)

    assert set(gemeente_zone.columns) == {
        "year",
        "gemeentecode",
        "stedelijkheidsklasse",
    }
    assert len(gemeente_zone) > 0
    assert gemeente_zone["gemeentecode"].str.len_chars().eq(4).all()
    assert set(gemeente_zone["stedelijkheidsklasse"].unique()) <= {
        "1",
        "2",
        "3",
        "4",
        "5",
    }


def test_gemeente_zone_join_maps_known_and_unknown_codes():
    gemeente_zone = pl.DataFrame(
        {
            "year": [2023, 2023],
            "gemeentecode": ["0363", "1680"],
            "stedelijkheidsklasse": ["1", "5"],
        }
    )
    trips = pl.DataFrame(
        {
            "origin_gemeente": ["0363", "9999"],
            "destination_gemeente": ["1680", "0363"],
        }
    )

    zone_lookup = gemeente_zone.filter(pl.col("year") == 2023).select(
        "gemeentecode", "stedelijkheidsklasse"
    )
    result = (
        trips.join(
            zone_lookup.rename(
                {
                    "gemeentecode": "origin_gemeente",
                    "stedelijkheidsklasse": "ozone",
                }
            ),
            on="origin_gemeente",
            how="left",
            maintain_order="left_right",
        )
        .join(
            zone_lookup.rename(
                {
                    "gemeentecode": "destination_gemeente",
                    "stedelijkheidsklasse": "dzone",
                }
            ),
            on="destination_gemeente",
            how="left",
            maintain_order="left_right",
        )
        .with_columns(
            ozone=pl.col("ozone").replace_strict(
                STED_MAPPING, default="unknown"
            ),
            dzone=pl.col("dzone").replace_strict(
                STED_MAPPING, default="unknown"
            ),
        )
    )

    assert result["ozone"].to_list() == ["urban", "unknown"]
    assert result["dzone"].to_list() == ["rural", "urban"]


def test_resolve_activity_chain_middle_round_trip():
    # seq 2 is a round trip (dact null, e.g. Doel code 10) between two real
    # activities — it should resolve to "work" (the last known activity),
    # and the following trip's oact should pick that up.
    data = pl.DataFrame(
        {
            "pid": ["1", "1", "1"],
            "seq": [1, 2, 3],
            "oact": ["home", "unset", "unset"],
            "dact": ["work", None, "shop"],
        }
    )

    result = odin.resolve_activity_chain(data).sort("seq")

    assert result["dact"].to_list() == ["work", "work", "shop"]
    assert result["oact"].to_list() == ["home", "work", "work"]


def test_resolve_activity_chain_leading_round_trip():
    # The person's very first trip is itself a round trip — there is no
    # previous dact to inherit, so it should fall back to its own
    # VertLoc-derived oact.
    data = pl.DataFrame(
        {
            "pid": ["1", "1"],
            "seq": [1, 2],
            "oact": ["home", "unset"],
            "dact": [None, "shop"],
        }
    )

    result = odin.resolve_activity_chain(data).sort("seq")

    assert result["dact"].to_list() == ["home", "shop"]
    assert result["oact"].to_list() == ["home", "home"]


def test_resolve_activity_chain_consecutive_round_trips():
    # Two round trips in a row should both cascade to the last real
    # activity, without needing multiple passes.
    data = pl.DataFrame(
        {
            "pid": ["1", "1", "1", "1"],
            "seq": [1, 2, 3, 4],
            "oact": ["home", "unset", "unset", "unset"],
            "dact": ["work", None, None, "shop"],
        }
    )

    result = odin.resolve_activity_chain(data).sort("seq")

    assert result["dact"].to_list() == ["work", "work", "work", "shop"]
    assert result["oact"].to_list() == ["home", "work", "work", "work"]


def test_resolve_activity_chain_no_round_trips_unaffected():
    data = pl.DataFrame(
        {
            "pid": ["1", "1", "2"],
            "seq": [1, 2, 1],
            "oact": ["home", "unset", "home"],
            "dact": ["work", "home", "shop"],
        }
    )

    result = odin.resolve_activity_chain(data).sort(["pid", "seq"])

    assert result["dact"].to_list() == ["work", "home", "shop"]
    assert result["oact"].to_list() == ["home", "work", "home"]


def test_load_prevents_cross_year_pid_collisions(monkeypatch):
    # Same raw OPID ("1") appears in two different years' files — OPID is
    # only unique within a single year's release, so odin.load() must embed
    # the year into pid/hid before concatenating, or the two distinct people
    # collide into one fabricated pid.
    def fake_load_households(root, config, year):
        return pl.DataFrame({"hid": ["1"]})

    def fake_load_persons(root, config, year):
        return pl.DataFrame(
            {"pid": ["1"], "hid": ["1"], "age": [30 if year == 2018 else 50]}
        )

    def fake_load_trips(root, config, year, gemeente_zone=None):
        return pl.DataFrame(
            {
                "pid": ["1"],
                "hid": ["1"],
                "tst": [480],
                "tet": [500],
                "distance": [5.0],
            }
        )

    monkeypatch.setattr(
        odin, "load_gemeente_zone", lambda configs_root: pl.DataFrame()
    )
    monkeypatch.setattr(odin, "load_households", fake_load_households)
    monkeypatch.setattr(odin, "load_persons", fake_load_persons)
    monkeypatch.setattr(odin, "load_trips", fake_load_trips)

    attributes, trips = odin.load(
        data_root="unused",
        configs_root="unused",
        years=[2018, 2019],
        hh_config={},
        person_config={},
        trips_config={},
    )

    assert attributes["pid"].n_unique() == 2
    assert set(attributes["pid"]) == {"odin20181", "odin20191"}
    assert set(attributes["hid"]) == {"odin20181", "odin20191"}
    assert set(trips["pid"]) == {"odin20181", "odin20191"}

    ages = dict(zip(attributes["pid"], attributes["age"]))
    assert ages["odin20181"] == 30
    assert ages["odin20191"] == 50
