# Output format

`foundata batch` (and `foundata run`) write the following files to the output
directory:

| File | One row per | Description |
|---|---|---|
| `attributes.csv` | person-day | Person, household and diary-day attributes |
| `trips.csv` | trip | Trip times, distance, mode, origin and destination activity and zone |
| `activity_schedules.csv` | activity | Plans as a sequence of activities with preceeding trip duration included with activity |
| `binned_attributes.csv` | person-day | `attributes.csv` with numeric columns binned into string labels and nulls filled |
| `locations.parquet` | location | GeoParquet of the home and trip-end zones (or points) referenced by the other tables |
| `original_ids.csv` | person-day | Only with `--keep-ids`: `source`, `pid`, `hid` and the survey's own `original_pid` and `original_hid`. Keep it private |
| `*.png` | — | Summary plots (add `--diagnostics` for anomaly reports and extra heatmaps) |

The canonical schema is `configs/core/template.yaml`.

## Conventions

- **Plans.** A plan is one person's diary day, from midnight to midnight.
  Some surveys record several diary days per person (NTS, the KTDB and
  Mexico weekend diaries, Mongolia). Each of those days becomes a separate
  `pid`, so a `pid` is really a person-day.
- **IDs.** `pid` and `hid` are integers, unique across all sources. They
  are numbered per source and do not carry the surveys' own ids, so records
  can't be joined back to the raw data by id (unless you run with
  `--keep-ids`). Each source's ids follow the
  previous source's, with gaps where plans were filtered out. Where a survey
  does not record households (KTDB 2021, ODIN), each person is their own
  household.
- **Times.** `tst`/`tet` (trip start and end) and `start`/`end` (activities)
  are integer minutes since midnight. All plans end by 1440 (midnight).
- **Distance** is in km. Depending on the source it is reported, routed,
  straight-line or estimated from duration × mode speed (see
  [Sources](sources.md)).
- **Missing values.** In `attributes.csv`, missing or unknown values are
  null (an empty cell). `void` means the field does not apply, for example
  `employment` for young children. In `trips.csv`, unknown categories are
  the string `unknown`.
- **Weights** are normalised to a mean of 1 within each source. They are not
  comparable across sources. KTDB has no weights.
- **Default filters.** By default `batch` keeps only plans that start and
  end at home (`--any-based` turns this off). It also merges consecutive
  home, work and education activities (`--any-consecutive` turns this off).
  Every source also has time-inconsistent, activity-inconsistent and
  infeasible plans removed (trips over 1000 km, over 12 h, or too fast).
- **Non-travellers.** Persons who made no trips stay in `attributes.csv` and
  get a single all-day `home` activity. They usually make up 15–30% of a
  source.

## attributes

| Column | Type | Values / range | Description |
|---|---|---|---|
| `pid` | int | | Person (person-day) ID |
| `hid` | int | | Household ID |
| `source` | str | e.g. `nhts`, `odin` | Survey source |
| `country` | str | e.g. `usa`, `nl` | Country code |
| `year` | int | | Survey year |
| `month` | int | 1–12 | Month of the diary day |
| `day` | str | `monday` … `sunday` | Day of week of the diary day |
| `weight` | float | ≥ 0 | Expansion weight, mean 1 per source |
| `age` | int | 0–120 | Age in years. Sampled within the band where the source gives bands |
| `sex` | str | male, female, other | |
| `relationship` | str | self, partner, child, parent, sibling, relative, unrelated, other | Relationship to the household reference person |
| `employment` | str | employed, unemployed, retired, student, void | |
| `employed_type` | str | ft, pt, void | Full time or part time, where known |
| `occupation` | str | clerical, manual, managerial, operator, professional, service, technical, other, void | |
| `education` | str | none, some, school, high-school, degree, void | Highest level attained |
| `can_wfh` | str | yes, no, void | Can work from home |
| `has_licence` | str | yes, no, void | Holds a driving licence |
| `disability` | str | yes, no | Has a disability that affects travel |
| `race` | str | white, black, asian, non-white, multiracial, other | Self-reported |
| `hh_size` | int | 1–100 | Persons in the household |
| `hh_income` | int | ≥ 0 | Annual gross household income in euros, usually sampled from a coded band |
| `vehicles` | int | 0–100 | Motor vehicles available to the household |
| `dwelling` | str | house, flat, other | |
| `ownership` | str | owned, rented, other | Housing tenure |
| `hh_zone` | str | urban, suburban, rural | Urban/rural class of the home |
| `hh_location` | int | | `location_id` of the home in `locations.parquet` |
| `access_egress_distance` | float | ≥ 0 | Average km of access and egress on transit trips |
| `avg_speed` | float | ≥ 0 | Total distance / total trip duration (km/h) |
| `max_temp_c` | float | | Daily maximum temperature on the diary day (CMAP, ODIN, some KTDB) |
| `rain` | bool | | Precipitation > 0 mm on the diary day (CMAP, ODIN, some KTDB) |

## trips

| Column | Type | Values / range | Description |
|---|---|---|---|
| `pid` | int | | Joins to `attributes.pid` |
| `seq` | int | ≥ 0 | Trip order within the plan |
| `oact` | str | home, work, education, shop, leisure, visit, escort, other | Activity at the origin |
| `dact` | str | same as `oact` | Activity at the destination (the trip purpose) |
| `mode` | str | walk, bike, car, bus, rail, other, unknown | Main mode |
| `tst` | int | ≥ 0 | Start time (minutes since midnight) |
| `tet` | int | ≤ 1440 | End time (minutes since midnight) |
| `distance` | float | ≥ 0 | km |
| `ozone` | str | urban, suburban, rural, unknown | Urban/rural class of the origin |
| `dzone` | str | same as `ozone` | Urban/rural class of the destination |
| `ocode` | int | | `location_id` of the origin, null if the source gives none |
| `dcode` | int | | `location_id` of the destination, null if the source gives none |

## activities

Built from the trips. Each activity includes the trip that leads to it, so
an activity's `start` is the start time of the trip that precedes it. The
first activity starts at 0 and the last ends at 1440.

| Column | Type | Description |
|---|---|---|
| `pid` | int | Joins to `attributes.pid` |
| `seq` | int | Activity order within the plan |
| `act` | str | Activity type, as in `trips.oact` |
| `start` | int | Start time (minutes since midnight) |
| `end` | int | End time (minutes since midnight) |
| `zone` | str | Urban/rural class of the activity's location |
| `location` | int | `location_id` of the activity. For home activities with no trip location, and for non-travellers, this is `hh_location` |

## binned_attributes

This table has the same columns as `attributes.csv`. Numeric columns
(except `year`, `month`, `weight`, `vehicles`, `hh_size` and `hh_location`)
are replaced by 5 quantile-bin labels such as `54-66`. Null strings become
`unknown` and null numerics become `-1`. You can re-bin any attributes file
with `foundata bin`.

## locations.parquet

This file has one row per distinct home or trip-end location. The geometry
is WKB in WGS84, so you can read it with `geopandas.read_parquet`.

| Column | Description |
|---|---|
| `location_id` | ID referenced by `hh_location`, `ocode`, `dcode` and `location`. It is only valid within this run |
| `location_key` | ID that is stable across runs: `<source>/<geo>/<code>` or `<source>/point/<lon>,<lat>` |
| `source`, `country` | Source and country |
| `level`, `code` | Unit type (for example `sa1`, `gemeente`, `point`) and its code |
| `geo` | Lookup table and vintage the code comes from (for example `gemeente_2018`) |
| `precision` | coordinate, small area, large area or region |
| `lon`, `lat` | A point inside the unit |
| `scale_m` | Radius of a circle with the unit's area, or the assumed uncertainty of a point |
| `geom_source` | `point`, `boundary`, `parent` (a containing unit), `estimated` or `none` |
| `geom_dataset`, `geom_year` | Boundary dataset and its vintage |
| `geometry` | Point, or the simplified polygon |

Location precision varies widely between sources, from about 50 m points
(Brazil, Chile) to whole regions (NTS, NHTS). The README's household
location summary has the figures for each source.
