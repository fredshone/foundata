# Gridded data

`foundata run` and `foundata batch` attach WorldTensor's 0.25° global
layers to each household. The pipeline first joins the layers to the
combined locations table, adding the `wt_` columns to
`locations.parquet`. It then copies the value columns to every person
row of `attributes.csv` by `hh_location`, the `location_id` of the home.
A household without a home location gets nulls.

```bash
uv run foundata batch ~/Data/foundata                 # WorldTensor from ~/Data/foundata/WORLD
uv run foundata batch ~/Data/foundata --world DIR     # another WorldTensor root
uv run foundata batch ~/Data/foundata --no-grid       # skip it: wt_ columns are null
```

A run checks that every layer exists before loading any survey. The
first run preprocesses each layer into a `.npy` under
`~/Data/foundata/cache/gridjoin/worldtensor` (about 4 MB each, under a
second in total). A layer is redone only when its source file changes,
and the WORLD folder is never written to. You can also do this step
ahead, with `uv run python scripts/preprocess_grids.py` or `foundata
gridjoin preprocess`. `foundata gridjoin join LOCATIONS` adds the
columns to an existing `locations.parquet`.

The join takes about 0.6 s for 183k locations.

## Variables

| column | WorldTensor layer | year | units |
|---|---|---|---|
| `wt_gdp_per_capita` | human_systems | 2022 | USD per capita (PPP, constant 2021) |
| `wt_ghsl_built_volume` | human_systems | 2020 | m³ |
| `wt_gini` | human_systems | 2023 | 1 |
| `wt_hdi` | human_systems | 2015 | 1 |
| `wt_ntl_harmonized` | human_systems | 2018 | 1 |
| `wt_population_density` | human_systems | 2020 | persons/km² |
| `wt_tp_sum` | climate (ERA5) | 2025 | m (see below) |
| `wt_t2m_mean` | climate (ERA5) | 2025 | K |
| `wt_skt_max` | climate (ERA5) | 2025 | K (skin temperature) |
| `wt_elevation_mean` | static topography | static | m |
| `wt_travel_time_to_cities` | static (Weiss et al. 2018) | 2015 | minutes |

!!! warning "Years"
    Every variable comes from its **most recent year on disk**, not the
    survey year, and those years differ by variable. The years used are
    recorded in `locations.parquet`'s metadata (`gridjoin` key) and in
    the cache's `manifest.json`. The table above shows the years as
    currently on disk.

`wt_tp_sum` averages 0.029 m a year worldwide. It appears to be the sum
of 12 monthly means of daily totals, not an annual total (multiply by
about 30.4 to get one). The values are kept as the files give them.

To change the variables, edit `VARIABLES` in
`foundata/gridjoin/worldtensor.py` and the matching `wt_` entries in
`configs/core/template.yaml`; a test checks that the two agree.

## How a location gets a cell

Each location has a point (`lon`, `lat`) and a radius (`scale_m`). The
point is the polygon's point on surface, so it is always inside the
polygon. The location takes the cell its point is in, found by
arithmetic with no geometry library, and each distinct cell is looked
up once.

The grid is point-registered: there are 721 × 1440 cell centres at
latitudes -90…90 and longitudes 0…359.75, so cell edges sit halfway
between centres. The cell is:

- `row = floor((lat + 90.125) / 0.25)`, clipped to 0…720;
- `col = floor((lon + 0.125) / 0.25) mod 1440`.

A point exactly on an edge goes to the cell north or east of it.
Latitudes ±90 stay in the polar rows, and longitudes 180 and -180 are
the same column.

These columns stay in `locations.parquet` only:

- `wt_cell` is `row * 1440 + col`.
- `wt_spans_cells` is true when the location's bounding box crosses a
  cell edge, so the single cell is an approximation. That is 33% of
  polygons and under 1% of points. A point's box is the point ± its
  radius. A polygon split at the antimeridian, like Alaska, gets the
  narrow box across ±180°, not one spanning the globe.
- `wt_<var>_fallback_km` exists for each layer that has gaps (NaN over
  the sea, or masked out). A location whose cell is NaN takes the
  nearest valid cell whose centre is within 50 km of its point; this
  column gives that distance, 0 for the location's own cell, or null if
  no valid cell is in range.

The fallback radius is 50 km, about two cells, because some masks are
eroded at coasts. For `gdp_per_capita`, the nearest valid cell is 35 km
from Nice and 43 km from Montevideo. Across the 183k locations from all
sources, the fallback fills 7,763 locations for GDP and 7,285 for
travel time. At most 25 stay null for any one variable.
