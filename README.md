[![Weekly](https://github.com/fredshone/foundata/actions/workflows/weekly.yml/badge.svg)](https://github.com/fredshone/foundata/actions/workflows/weekly.yml)

Foundata is a pipeline for creating reconciled household travel surveys, aimed at enabling *foundational* or *worldwide* models of human behaviour — but also a useful source of code for those wishing to work with the available datasets. you can read more [here](https://fredshone.github.io/foundata/).

The project is intended to be uses as a discoverable cli via `uv run foundata --help`.

Run a single source with `foundata run <SOURCE> <SOURCE_DIR>`, pointing at that source's own data folder, e.g.:

```bash
uv run foundata run ltds ~/Data/foundata/LTDS         # writes to output/ltds
```

Or process and combine many sources with `foundata batch <DATA_ROOT>`, e.g. `uv run foundata batch --help`:
```
Usage: foundata batch [OPTIONS] DATA_ROOT

  Run the pipeline end-to-end over many sources and combine them.

Arguments:
  DATA_ROOT  Base data directory containing one folder per source (LTDS/, NTS/, ...), e.g. ~/Data/foundata  [required]

Options:
  -o, --output PATH                                   Directory where CSVs and PNGs are written.  [default: output]
  -s, --select SOURCE...                              Sources to process (e.g. -s nhts ktdb, or -s nhts,ktdb). Defaults to all.
  -x, --omit SOURCE...                                Sources to omit (e.g. -x nhts ktdb, or -x nhts,ktdb).
  --open                                              Run only the open-data sources (nhts, cmap, vista, qhts, ktdb, mexico, argentina, uruguay, chile). Overrides --select and --omit.
  -hb, --home-based / -ab, --any-based                Whether to only include home-based trips (i.e. those with 'home' as the origin or destination activity).  [default: home-based]
  -fc, --filter-consecutive / -ac, --any-consecutive  Whether to filter out consecutive home, work and education activities.  [default: filter-consecutive]
  -d, --diagnostics                                   Also produce the anomaly-detection reports and supplementary activity heatmaps.
  --keep-ids                                          Also write original_ids.csv, mapping each pid and hid back to the survey's own ids. It re-identifies records, so keep it private.
  -l, --log-level [debug|info|warning|error]          Minimum level of log messages to show.  [default: info]
  --help                                              Show this message and exit.
```

By default a run writes the outputs, the summary tables and the plots shown in this README. Add `--diagnostics` for the anomaly-detection tables (and their CSVs) and the per-attribute activity heatmaps. Add `--keep-ids` to also write `original_ids.csv`, which maps each `pid` and `hid` back to the survey's own ids for debugging; it re-identifies records, so don't share it. Every command takes `-l/--log-level` (default `info`); use `-l debug` for finer-grained detail. The per-source processing notes are logged after the processing log table, each at its own level (e.g. `warning`), so `-l error` shows only the most serious.

The base data directory needs to hold raw data downloaded from the various sources by the user. The template directory `data_dir_template`, shows the expected folder and files structure.

### Data summary

The latest output using `uv run foundata batch` is as follows:

| Source    | Plans         | Missing attributes | Trips          | Trip kms (millions) |
|-----------|---------------|--------------------|----------------|---------------------|
| nts       | 2,496,735     | 19%                | 4,438,809      | 47.7                |
| ktdb      | 2,225,798     | 29%                | 4,350,059      | 37.4                |
| india     | 875,802       | 48%                | 1,903,711      | 0.0                 |
| nhts      | 634,627       | 26%                | 2,216,636      | 26.4                |
| france    | 581,344       | 27%                | 2,046,694      | 15.3                |
| odin      | 384,215       | 25%                | 887,774        | 9.7                 |
| mexico    | 348,072       | 47%                | 494,953        | 5.3                 |
| ovin      | 291,125       | 31%                | 724,075        | 7.7                 |
| mon       | 158,815       | 39%                | 421,908        | 4.2                 |
| vista     | 89,507        | 28%                | 235,983        | 2.0                 |
| argentina | 85,950        | 32%                | 133,105        | 0.9                 |
| brazil    | 83,895        | 32%                | 150,846        | 0.9                 |
| ltds      | 60,621        | 35%                | 108,570        | 0.9                 |
| chile     | 56,580        | 24%                | 104,761        | 0.8                 |
| qhts      | 48,739        | 32%                | 117,936        | 1.2                 |
| mongolia  | 46,100        | 33%                | 99,267         | 0.0                 |
| cmap      | 25,733        | 7%                 | 76,739         | 0.5                 |
| uruguay   | 5,173         | 37%                | 10,753         | 0.1                 |
| **total** | **8,498,831** | **29%**            | **18,522,579** | **161.1**           |

\* Plans: a sequence of activities and associated trips within a 24hr period starting at midnight.

\*\* missing attributes: attribute data with `unknown`/`null` values.

### Data sources

See the [source docs](https://fredshone.github.io/foundata/sources/) for the full list of sources, how to access them and per-source notes.

### In progress or todo

- Bogotá EM 2023 / 2019
- MTUS harmonised episodes (Korea) https://timeuse.org/
- Tokyo PT 2008 / 2018
- JICA PT surveys via PFLOW
- Hong Kong TCS 2011 / 2022


### Formats

Foundata processes input surveys into *harmonised* "plans" and "attributes". These come in various formats as follows. Refer to the [output format docs](https://fredshone.github.io/foundata/format/) for more detail.

#### Plans

We encode human activity plans as sequences of activities and associated trips. Foundata `batch` or `run` will output both a trips table and an activity_schedules table. Temporal and spatial consistency is enforced, so that sequences should be physically plausible.

#### Trips

The trips output provides both trip information and origin/destination location and activity information:

|   pid | seq | ozone | dzone | oact | dact | mode | tst | tet | distance |
|------:|:----|------:|------:|------|------|------|-----|-----|----------|
|     0 | 0   | rural | urban | home | shop | car  | 600 | 620 | 12       |
|     0 | 1   | urban | rural | shop | home | car  | 660 | 680 | 12       |

Trip start time (tst) and trip end time (tet) are in minutes since midnight. Distance is in km.

We currently map all transport modes to the following types: {car, walk, bike, bus, rail, other}.

#### Activity schedules

The activities output combines activities with their **preceding** trip to create activity-only sequences:

|   pid | seq | act     |   start |   end |
|------:|-----|:--------|--------:|------:|
|     0 | 0   | home    |       0 |   600 |
|     0 | 1   |  shop   |     600 |  660  |
|     0 | 2   |  home   |     660 |  1440 |

Activities also carry `zone` (the urban/rural class of the trip end) and `location`, a `location_id` of `locations.parquet` (see [Household location summary](#household-location-summary)): the location of the trip ending there (the first activity takes the first trip's origin), or, for a home activity the trip gives none for, and for persons with no trips, the household's `hh_location`.

We currently map all activities to the following types: {home, work, education, visit, leisure, shop, escort, other}.


#### Person attributes

Plans are joinable by a unique person id (pid) to attributes. Attributes include household, individual, day and plan information.

|   pid | source   |   year | age   | employment   | hh_income   | hh_zone   | ... |
|------:|:---------|-------:|:------|:-------------|:------------|:----------|-----|
|     0 | xxxx     |   2021 | 51-65 | employed     | ≤21280      | urban     | ... |

Categorical person attributes, **Blank** signifies missing or "unknown" data:

![Categorical person attributes](assets/attributes_categorical.png)

There has been a lot of effort to consolidate categories across the different data sources. You can see the mappings used in `/configs`. Note that (i) we allow unknown categories (null or unknown), and (ii) in some cases we allow "overlapping" categories (for the `race` attribute in particular).

Numeric person attributes:

![Numeric person attributes](assets/attributes_numeric.png)

A sample of some trends:

![Attribute trends](assets/attributes_trends.png)

Data availability (non null or unknown) by source and attribute:

| Attribute              | argentina | brazil | chile | cmap | france | india | ktdb | ltds | mexico | mongolia | nhts | nts  | odin | mon  | ovin | qhts | uruguay | vista | all  |
|------------------------|-----------|--------|-------|------|--------|-------|------|------|--------|----------|------|------|------|------|------|------|---------|-------|------|
| country                | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| year                   | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| weight                 | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| sex                    | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| hh_zone                | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| age                    | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| hh_location            | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| hh_size                | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 99%  | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 100% |
| employment             | 100%      | 100%   | 99%   | 100% | 98%    | 99%   | 100% | 100% | 100%   | 96%      | 96%  | 100% | 95%  | 98%  | 98%  | 96%  | 98%     | 90%   | 99%  |
| vehicles               | 100%      | 98%    | 100%  | 100% | 100%   | 0%    | 100% | 100% | 100%   | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 100%    | 100%  | 90%  |
| hh_income              | 99%       | 100%   | 100%  | 99%  | 1%     | 100%  | 98%  | 45%  | 0%     | 0%       | 95%  | 92%  | 98%  | 75%  | 99%  | 39%  | 0%      | 96%   | 84%  |
| relationship           | 100%      | 100%   | 100%  | 100% | 100%   | 100%  | 95%  | 100% | 100%   | 100%     | 100% | 56%  | 84%  | 0%   | 100% | 100% | 99%     | 100%  | 83%  |
| employed_type          | 59%       | 53%    | 99%   | 45%  | 98%    | 57%   | 82%  | 53%  | 49%    | 40%      | 50%  | 100% | 95%  | 98%  | 98%  | 96%  | 54%     | 90%   | 83%  |
| day                    | 26%       | 100%   | 100%  | 100% | 98%    | 100%  | 39%  | 100% | 50%    | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 0%      | 100%  | 81%  |
| has_licence            | 95%       | 0%     | 100%  | 100% | 100%   | 0%    | 99%  | 100% | 0%     | 0%       | 88%  | 80%  | 100% | 100% | 100% | 100% | 93%     | 82%   | 77%  |
| ownership              | 0%        | 100%   | 100%  | 100% | 73%    | 50%   | 69%  | 0%   | 0%     | 100%     | 100% | 100% | 0%   | 0%   | 0%   | 0%   | 100%    | 100%  | 69%  |
| avg_speed              | 65%       | 70%    | 76%   | 78%  | 87%    | 0%    | 82%  | 63%  | 64%    | 0%       | 81%  | 65%  | 78%  | 82%  | 79%  | 78%  | 73%     | 74%   | 67%  |
| month                  | 26%       | 100%   | 100%  | 100% | 96%    | 0%    | 31%  | 0%   | 0%     | 100%     | 100% | 100% | 100% | 100% | 100% | 100% | 0%      | 100%  | 66%  |
| dwelling               | 100%      | 0%     | 0%    | 100% | 73%    | 0%    | 99%  | 0%   | 0%     | 100%     | 0%   | 90%  | 0%   | 0%   | 0%   | 84%  | 100%    | 100%  | 61%  |
| education              | 99%       | 100%   | 100%  | 100% | 80%    | 100%  | 4%   | 0%   | 100%   | 100%     | 100% | 67%  | 91%  | 83%  | 80%  | 0%   | 100%    | 0%    | 60%  |
| occupation             | 59%       | 100%   | 43%   | 100% | 97%    | 0%    | 100% | 100% | 48%    | 99%      | 0%   | 65%  | 0%   | 0%   | 0%   | 48%  | 56%     | 100%  | 59%  |
| race                   | 0%        | 0%     | 0%    | 99%  | 0%     | 0%    | 0%   | 83%  | 0%     | 0%       | 99%  | 100% | 84%  | 0%   | 100% | 0%   | 0%      | 0%    | 45%  |
| access_egress_distance | 0%        | 29%    | 33%   | 0%   | 11%    | 0%    | 6%   | 24%  | 29%    | 0%       | 0%   | 89%  | 0%   | 0%   | 0%   | 0%   | 27%     | 0%    | 30%  |
| disability             | 98%       | 0%     | 100%  | 100% | 1%     | 0%    | 0%   | 0%   | 0%     | 69%      | 91%  | 58%  | 0%   | 0%   | 0%   | 100% | 0%      | 0%    | 27%  |
| can_wfh                | 0%        | 0%     | 0%    | 98%  | 63%    | 0%    | 51%  | 100% | 0%     | 0%       | 0%   | 13%  | 0%   | 0%   | 0%   | 0%   | 0%      | 0%    | 23%  |
| max_temp_c             | 0%        | 0%     | 0%    | 100% | 0%     | 0%    | 30%  | 0%   | 0%     | 0%       | 0%   | 0%   | 100% | 0%   | 0%   | 0%   | 0%      | 0%    | 13%  |
| rain                   | 0%        | 0%     | 0%    | 100% | 0%     | 0%    | 30%  | 0%   | 0%     | 0%       | 0%   | 0%   | 100% | 0%   | 0%   | 0%   | 0%      | 0%    | 13%  |

#### Location summary

Attribute data includes home location, and most trips carry their origin and destination:

- `hh_location` in attributes
- `ocode`/`dcode` in trips
- `location` in activity_schedules

To minimise size, all locations are indexes, that refer to geoemtries defined by the output `locations.parquet` (see below), which gives the zone (or point), its precision, its polygon, centroid, its size and where the geometry came from. 

| Column | |
|---|---|
| `location_id` | id in this output: each source's locations are numbered in `location_key` order, after the previous source's |
| `location_key` | stable across runs: `<source>/<geo>/<code>`, or `<source>/point/<lon>,<lat>`; loaders write these keys, and the runner swaps them for ids per source |
| `source`, `country` | the survey source and its country |
| `level`, `code` | the kind of unit (`gemeente`, `sa1`, `point`) and the unit's code |
| `geo` | the lookup the code is from (`gemeente_2018`), which fixes the code list's vintage |
| `precision` | coordinate, small area, large area or region (from the level) |
| `lon`, `lat` | a point inside the polygon, or the point |
| `scale_m` | radius of a circle with the polygon's area (before simplification), or the assumed uncertainty of a point; use it to pick a raster resolution or a sampling radius |
| `geom_source` | `point` (the survey's coordinate), `boundary` (the unit's own polygon), `parent` (the unit has no published polygon, so a unit known to contain it), `estimated` (**work in progress**: a provisional area containing the unit) or `none` (no geometry: the code is not in the boundaries, e.g. trips abroad) |
| `geom_dataset`, `geom_year` | the boundary dataset and its vintage (of the survey, for point data) |
| `geometry` | the point, or the unit's polygon simplified to 5% of its equivalent radius (5 m to 1 km); polygons are simplified one at a time, so neighbours can gap or overlap by up to that much |

Conventions:

- Codes are standard codes where the survey allows (ONS, FIPS, ABS, CBS, INDEC, KOSTAT/MOIS, NSO), mapped from survey numbering in the source's loader. Survey-specific zone systems keep the survey's codes, prefixed where they would clash (French detailed zones are `<survey>/<zone>`).
- Where codes are renumbered or redrawn between rounds, each round reads the lookup of its year (`gemeente_2012`, `cbsa_2013`, `dong_2016`, `radio_2010`), so the same code in two vintages is two locations.
- Each source declares its locations in `home_location` (`hh_dictionary.yaml`) and `trip_location` (`trip_dictionary.yaml`, for both trip ends): per survey year, the levels finest first, each with its lookup file, e.g. ODIN 2018: `pc4` (`pc4.parquet`), then `gemeente` (`gemeente_2018.parquet`). The source's loader builds the codes (padding, mapping, validity), and a household or trip end takes the first level whose code is in its lookup, so a code the boundaries don't have falls back to the next level (NHTS: CBSA or MSA, else state; France: detailed zone, else point, else commune; Argentine radio to fracción, Korean 행정동 to 시군구, Mongolian bag to soum). See `foundata/location.py`.

Coverage is the share of households in the full batch with a geometry (any `geom_source` other than `none`); exact is the share whose unit has its own boundary, or is a point. Median `scale_m` is over households. Trip ends is the share of trip origins and destinations with a geometry. `foundata location-summary <batch output>` prints these numbers:

| Source | Home levels | Coverage | Exact | Median `scale_m` | Trip ends with geometry |
|---|---|---|---|---|---|
| argentina | fraccion, radio | 100% | 100% | 213 m | — |
| brazil | point | 100% | 100% | 50 m | 100% |
| chile | point | 100% | 100% | 50 m | 98.2% |
| cmap | tract | 100% | 100% | 875 m | 99.8% |
| france | commune, detailed_zone, point | 100% | 97.3% | 715 m | 99.5% |
| india | district | 100% | 100% | 35.0 km | — |
| ktdb | dong, sigungu | 100% | 100% | 969 m | 100% |
| ltds | lad | 100% | 100% | 3.7 km | 96.9% |
| mexico | eod_district | 100% | 0.5% | 6.0 km | 96% |
| mongolia | bagh, soum | 99.7% | 27.4% | 19.8 km | — |
| nhts | cbsa, census_division, msa, state | 99.6% | 99.6% | 167.8 km | — |
| nts | region | 100% | 100% | 70.6 km | — |
| odin | gemeente, pc4 | 99.7% | 99.7% | 1.3 km | 98.1% |
| mon | gemeente | 100% | 100% | 5.1 km | 99.8% |
| ovin | gemeente | 100% | 100% | 5.3 km | 97.4% |
| qhts | sa1 | 100% | 100% | 256 m | 99.8% |
| uruguay | segment | 100% | 100% | 320 m | 99.6% |
| vista | lga | 100% | 100% | 6.0 km | 99.8% |


Notes:

- Brazil and Chile give dwelling coordinates without a stated precision, so 50 m is an assumption. 99.9% and 99.8% of them fall in their own survey zone.
- Mexico and Mongolia are **work in progress**. Mexico's EOD 2017 districts were never published as polygons, and they cut across municipalities, so each district takes the union of the municipalities it overlaps; only Tizayuca (a whole municipality) is exact. Mongolian bag boundaries are not public outside Ulaanbaatar, so a rural bagh takes its soum, and Ulaanbaatar khoroos have been split since 2019, so 2019 khoroos take today's khoroo of that number. These `estimated` geometries contain the unit, so `scale_m` is an honest radius of uncertainty, but they are much larger than the unit itself. Possible sources: the district map in UNAM's EOD 2017 report (Mexico), OpenStreetMap or the NSO (Mongolia).
- French zones without a polygon (all of Calvados 2011, and a few zones in seven other surveys) take their draw zone, else their commune (`parent`). French trip ends without a geometry are mostly abroad (INSEE codes 99xxx).
- Lookups cover the survey area, sometimes a bit more (CMAP: three states; VISTA: Victoria; LTDS: the UK). Trip ends outside them have `geom_source` none.

Lookups are GeoParquet files in `configs/<source>/geo/`, named by the `geo` entries of the declarations (`code, lon, lat, area_km2, geom_source, geom_dataset, geom_year, geometry`; about 38 MB in all). Each file's parquet metadata (key `foundata`) records the datasets (URL, sha256), the method, alias files, and codes matched fuzzily or not at all. `scripts/build_location_lookups.py` builds them all, for the files each source declares, downloading the public boundary files it needs to `<data root>/boundaries/` (its docstring lists them) and recording each in `configs/location_datasets.lock.csv`. `scripts/build_location_lookup.py` builds one from a local boundary file. Trip distances use the same lookups (KTDB, Chile and Uruguay zone points).

## Usage

### Setup

```bash
uv sync          # install dependencies and register the CLI entry point
foundata --help
```

### Trying it out with toy data

`data_dir_template/` is a small, runnable stand-in for a real `~/Data/foundata/`
tree — same folder/file layout, just a handful of rows per source — so you can
try the pipeline without access to any of the real data:

```bash
foundata batch data_dir_template --output /tmp/out
```

### Adding a new source

1. **Scaffold boilerplate** — generates empty YAML configs and a stub loader:
   ```bash
   python scripts/scaffold_source.py <source>
   ```

2. **Populate YAML configs** in `configs/<source>/`:
   - `hh_dictionary.yaml` — household column mappings and value remappings
   - `person_dictionary.yaml` — person column mappings and value remappings
   - `trip_dictionary.yaml` — trip column mappings and value remappings

3. **Validate YAML configs** against the template schema:
   ```bash
   foundata validate-config <source>
   ```
   Fix any reported ERRORs (value labels not in the template set). WARNs for
   intermediate fields are expected and can be ignored.

4. **Implement `load()`** in `foundata/<source>.py` following the pattern of
   existing loaders (e.g. `nhts.py`). The function should return
   `(attributes_df, trips_df)` normalised to the template schema.

5. **Run the loader, write CSVs, then validate output**:
   ```bash
   foundata validate-table attributes.csv trips.csv
   ```

6. **Add the source to `foundata/run.py`**: write a `_load_<source>(source_dir)` function and register it in `SOURCES`. This makes it available to both `foundata run <source>` and `foundata batch`.

7. **Declare locations and build their lookups**: list the levels and their lookup files in `home_location` in `hh_dictionary.yaml` (and `trip_location` in `trip_dictionary.yaml`); in the loader, write a clean `home_<level>` code column per level and call `location.add` / `location.add_trip_ends`; then add the boundary datasets and a builder for the source to `scripts/build_location_lookups.py` and run it. See the LOCATIONS section of `llm_auto_source.txt`; `tests/test_location.py` and `tests/test_locations_batch.py` check the declarations, lookups and coverage.

### Running specific sources

Use `foundata run` for a single source, or `batch` with `--select` / `-s` to run a subset of sources, or `--omit` / `-x` to exclude sources:

```bash
# Run KTDB only
foundata run ktdb ~/Data/foundata/KTDB --output /tmp/out

# Run KTDB and NTS combined
foundata batch ~/Data/foundata -s ktdb -s nts --output /tmp/out

# Run everything except NTS
foundata batch ~/Data/foundata --omit nts --output /tmp/out
```

Available sources: `ltds`, `vista`, `qhts`, `cmap`, `nhts`, `nts`, `ktdb`, `odin`, `ovin`, `mon`, `india`, `brazil`, `mexico`, `argentina`, `uruguay`, `mongolia`, `chile`, `france`.

### Binning numeric attributes

The `bin` command discretises numeric columns in an attributes CSV into labelled string bins, using the same quantile/uniform logic as the pipeline's `binned_attributes.csv` output — but runnable on any attributes file with full control over bin counts.

```bash
# All numeric columns binned into 5 quantile bins (default)
foundata bin attributes.csv

# Override the default bin count
foundata bin attributes.csv --default 8

# Per-column overrides: --COLUMN N takes precedence over --default
foundata bin attributes.csv --default 5 --age 10 --hh_income 3

# Uniform (equal-width) bins, explicit output path
foundata bin attributes.csv --default 5 --method uniform --output binned.csv
```

Options:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--default N` | `-n` | `5` | Default number of bins for all numeric columns. |
| `--method` | `-m` | `quantile` | `quantile` (equal-frequency) or `uniform` (equal-width). |
| `--output PATH` | `-o` | `<input>_binned.csv` | Output CSV path. |
| `--select TEXT` | `-s` | | Columns to bin (repeatable, e.g. `--select age --select hh_income`). |
| `--omit TEXT` | `-x` | | Columns to exclude from binning (repeatable). |
| `--COLUMN N` | | | Per-column bin count override (e.g. `--age 10`). |

### Filling missing values

The `fill-unknown` command fills null/missing values in an attributes CSV with the string `"unknown"`, and reports the percentage of each column filled. It warns if a column is entirely unknown or looks numeric (filling a numeric column with a string is usually a mistake).

```bash
foundata fill-unknown attributes.csv --output filled.csv
```

Options:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output PATH` | `-o` | `<input>_filled.csv` | Output CSV path. |

### Filtering output CSVs

The `filter` command group applies post-processing filters to `attributes.csv` / `trips.csv` outputs.
All filter commands accept `-a`/`--attributes` (optional), `-t`/`--trips` (required), and output options `-o` (directory), `-oa` (explicit attributes path), `-ot` (explicit trips path).

```bash
foundata filter --help
```

| Command | Description |
|---------|-------------|
| `homebased` | Keep only plans whose first and last activity is home. |
| `missing-acts-or-modes` | Remove plans with null or `unknown` activities or modes. |
| `consecutive-activities` | Remove plans with consecutive same-type activities (e.g. work→work). Configurable via `-n`/`--non-consecutive-types` (default: `home`, `work`, `education`). |
| `negative-trips` | Remove plans containing trips where `tst > tet`. |
| `negative-activities` | Remove plans with overlapping trip times (negative activity durations). |
| `null-times` | Remove plans with null trip start or end times. |
| `time-consistent` | Apply all time-consistency filters in one step. |
| `attributes` | Filter persons on a column value and restrict trips to survivors. |

Example:

```bash
foundata filter consecutive-activities -t trips.csv -a attributes.csv -n work -n education -o output/
```

### Splitting into train/test sets

The `split` command creates train/test splits of one or more CSVs, keeping all records for each person entirely in one set (never split across both). Pass any number of CSV files — they must all share the same set of group IDs.

```bash
foundata split attributes.csv trips.csv activities.csv --split 20 --output /tmp/split/
```

Output:

```
Split on 'pid': 800 train / 200 test (20%)
  attributes.csv                 →     800 train /     200 test rows
  trips.csv                      →    6431 train /    1612 test rows
  activities.csv                 →    9204 train /    2301 test rows
Wrote outputs to /tmp/split/
```

Each input file produces `<stem>_train.csv` and `<stem>_test.csv` in the output directory.

Options:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--group COL` | `-g` | `pid` | Column to group by. |
| `--split PCT` | `-s` | `20` | Test set size as a percentage. |
| `--output DIR` | `-o` | parent of first input | Output directory. |
| `--seed N` | | `42` | Random seed for reproducibility. |


### Diagnostics

Departure/arrival time-of-day density, wrapped onto a 0-24h axis. The legend shows each source's share of trips with a raw start/end time greater than 1440 minutes (an uncorrected day-wrap is a common symptom of a source-specific time bug):

![Trip time of day](assets/trip_time_of_day.png)

Minute-within-hour "heaping" per source — self-reported times tend to round to :00/:15/:30/:45, and a source heaping much more than the others usually means less precise raw timestamps:

![Trip time heaping](assets/trip_time_heaping.png)

Trip duration, implied speed (`distance / duration`), and the share of trips with non-positive duration, per source:

![Trip time diagnostics](assets/trip_time_diagnostics.png)

Activity duration distributions, faceted by activity type:

![Activity duration by type](assets/activity_duration_by_type.png)

Activity participations by employment, income and age:

![Activity heatmap by age](assets/activity_counts_grid.png)
