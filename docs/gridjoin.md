# Gridded data

`foundata run` and `foundata batch` attach WorldTensor's 0.25° global
layers to each household, from each layer's year nearest the survey
year. The pipeline first gives each location in the combined locations
table its cell, adding `wt_cell` and `wt_spans_cells` to
`locations.parquet`; these do not depend on the year. It then reads the
values for each distinct home location and survey year, and adds them to
every person row of `attributes.csv`: `wt_<var>`, with `wt_<var>_year`,
the year it is from. A household without a home location gets nulls.

```bash
uv run foundata batch ~/Data/foundata                 # WorldTensor from ~/Data/foundata/WORLD
uv run foundata batch ~/Data/foundata --world DIR     # another WorldTensor root
uv run foundata batch ~/Data/foundata --no-grid       # skip it: wt_ columns are null
```

A run checks that every layer exists before loading any survey. It
preprocesses each (variable, year) layer it needs into a `.npy` under
`~/Data/foundata/cache/gridjoin/worldtensor` (about 4 MB each, so under
1 GB for every survey year from 2008 to 2025). Layers stay cached across
runs, a layer is redone only when its source file changes, and the WORLD
folder is never written to.

## Variables

| column | WorldTensor layer | years on disk | units |
|---|---|---|---|
| `wt_gdp_per_capita` | human_systems | 1990–2022 | USD per capita (PPP, constant 2021) |
| `wt_ghsl_built_volume` | human_systems | 1975–2020 | m³ |
| `wt_gini` | human_systems | 1990–2023 | 1 |
| `wt_hdi` | human_systems | 1990–2015 | 1 |
| `wt_ntl_harmonized` | human_systems | 1992–2018 | 1 |
| `wt_population_density` | human_systems | 2000–2020 | persons/km² |
| `wt_tp_sum` | climate (ERA5) | 1940–2025 | m (see below) |
| `wt_t2m_mean` | climate (ERA5) | 1940–2025 | K |
| `wt_skt_max` | climate (ERA5) | 1940–2025 | K (skin temperature) |
| `wt_elevation_mean` | static topography | static | m |
| `wt_travel_time_to_cities` | static (Weiss et al. 2018) | 2015 | minutes |

## Years

Each variable is taken from its year on disk nearest the survey year,
the earlier of two equally near. Within a range with no gaps that is the
survey year itself; past the end of a range it is the last year, so a
2023 survey gets `hdi` from 2015 and `ntl_harmonized` from 2018. A
static layer has one year, its snapshot's where the file gives one
(`travel_time_to_cities` 2015), else null.

`wt_<var>_year` records the year used on every row, so you can see how
far each value is from its survey year (`year - wt_<var>_year`). The
`gridjoin` key of `locations.parquet`'s metadata has the same mapping,
per variable from each survey year to the year used, with its units and
source.

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

These columns are in `locations.parquet`:

- `wt_cell` is `row * 1440 + col`.
- `wt_spans_cells` is true when the location's bounding box crosses a
  cell edge, so the single cell is an approximation. That is 33% of
  polygons and under 1% of points. A point's box is the point ± its
  radius. A polygon split at the antimeridian, like Alaska, gets the
  narrow box across ±180°, not one spanning the globe.

A location whose cell is NaN in a layer (over the sea, or masked out,
which can differ by year) takes the nearest valid cell of that layer
whose centre is within 50 km of its point, else null. The `gridjoin`
metadata gives, per variable, how many home locations and years were on
a NaN cell (`on_nan_cells`) and how many the fallback filled (`filled`).

The fallback radius is 50 km, about two cells, because some masks are
eroded at coasts. For `gdp_per_capita`, the nearest valid cell is 35 km
from Nice and 43 km from Montevideo. With each layer's most recent year,
across the 183k locations from all sources, the fallback filled 7,763
locations for GDP and 7,285 for travel time, and at most 25 stayed null
for any one variable. The counts for a run are in its metadata.
