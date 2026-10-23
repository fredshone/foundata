# Gridded data

To allow for a worldwide model we provide household location labels that are:

- globally available
- that are invariant from location to location
- likely to be correlated to human behaviors

We use [WorldTensor](https://zenodo.org/records/19047618) datasets. WorldTensor provides various variables on a 0.25° global grid. Both `run` and `batch` join these variables to the attributes table using hh_location.

```bash
uv run foundata batch ~/Data/foundata                 # WorldTensor from ~/Data/foundata/WORLD
uv run foundata batch ~/Data/foundata --world DIR     # another WorldTensor root
uv run foundata batch ~/Data/foundata --grid-config F # other variables or weights
uv run foundata batch ~/Data/foundata --no-grid       # skip it: wt_ columns are null
```

## Precision

The 0.25° global grid call vary in size from zero a the poles to aroun 25km x 25km at the equator.

<figure markdown="span">
![WorldTensor UK population density (logged)](assets/wt_log_pop_density_uk.png){ style="display: block; margin: 0 auto; max-width: 100%;" }
<figcaption style="text-align: center">WorldTensor UK (log) population density </figcaption> 
</figure>


## Variables

| column | WorldTensor layer | available years | weight | units |
|---|---|---|---|---|
| `wt_gdp_per_capita` | human_systems | 1990–2022 | population | GDP as USD per capita (PPP, constant 2021) |
| `wt_ghsl_built_volume` | human_systems | 1975–2020 | population | Building volume as m³ |
| `wt_gini` | human_systems | 1990–2023 | population | Gini index |
| `wt_hdi` | human_systems | 1990–2015 | population | Human development index |
| `wt_ntl_harmonized` | human_systems | 1992–2018 | population | Night-time lighting |
| `wt_population_density` | human_systems | 2000–2020 | population | Persons/km² |
| `wt_tp_sum` | climate (ERA5) | 1940–2025 | area | Total precipitation as m (see below) |
| `wt_t2m_mean` | climate (ERA5) | 1940–2025 | area | Mean temperature in K |
| `wt_skt_max` | climate (ERA5) | 1940–2025 | area | Max temperature K |
| `wt_elevation_mean` | static topography | static | area | Elevation above sea level in m |
| `wt_travel_time_to_cities` | static (Weiss et al. 2018) | 2015 | population | Travel time to nearest city (minutes) |

<figure markdown="span">
![WorldTensor population density (logged)](assets/wt_log_pop_density.png){ style="display: block; margin: 0 auto; max-width: 100%;" }
<figcaption style="text-align: center"> WorldTensor (log) population density </figcaption> 
</figure>

## Years

Each variable is taken from its year on disk nearest the survey year,
the earlier of two equally near. Within a range with no gaps that is the
survey year itself; past the end of a range it is the last year, so a
2023 survey gets `hdi` from 2015 and `ntl_harmonized` from 2018. A
static layer has one year, its snapshot's where the file gives one
(`travel_time_to_cities` 2015), else null.

The `gridjoin` key of `locations.parquet`'s metadata records, per
variable, the year used for each survey year, with its units and
source.

`wt_tp_sum` averages 0.029 m a year worldwide. It appears to be the sum
of 12 monthly means of daily totals, not an annual total (multiply by
about 30.4 to get one). The values are kept as the files give them.

## Config

`configs/core/gridjoin.yaml` lists the variables (another file:
`--grid-config`). Each one has its
source under the WORLD root and its weight:

```yaml
weights:
  population_count: worldtensor_human_systems/human_systems/population_count
variables:
  gdp_per_capita:
    source: worldtensor_human_systems/human_systems/gdp_per_capita
    weight: population_count  # or area
```

Use `area` for land use, climate and static layers, and
`population_count` for human systems and travel time to cities. A variable added to the config
also needs its `wt_` entry in `configs/core/template.yaml`. A test
checks that the two agree.

## How a location gets its values

- **Points** take the value of the cell they are in. `scale_m` is ignored.
- **Polygons** take the mean over every cell they overlap, weighted by
  the variable's weight:
    - `area`: the share of each cell the polygon covers.
    - `population_count`: the people in each overlap, which is
      `population_count` (year nearest the survey) times that share.
      Where the overlaps have no people, the mean is weighted by area.
- **Missing cells** (sea, or masked) are left out of the mean. If none
  are left, the location takes the nearest valid cell within 50 km of
  its point, else null.

A polygon in a single cell takes that cell's value, the same as a point.

### Details

The grid is point-registered: 721 × 1440 cell centres at latitudes
-90…90 and longitudes 0…359.75, with edges halfway between them. A point
is in cell `row * 1440 + col`, with

- `row = floor((lat + 90.125) / 0.25)`, clipped to 0…720;
- `col = floor((lon + 0.125) / 0.25) mod 1440`.

A point on an edge goes to the cell north or east of it. Longitudes 180
and -180 are the same column. A polygon's point (`lon`, `lat`) is its
point on surface, so always inside it. Polygons split at the
antimeridian, like Alaska, are handled.

Overlaps are measured in degrees, which is equal-area within a cell.
Only polygons whose bounding box crosses a cell edge are intersected
with the grid. Across all sources that takes a few seconds.

`locations.parquet` has two of these columns:

- `wt_cell`: the cell of the location's point;
- `wt_spans_cells`: whether its bounding box crosses a cell edge, so
  that its values average several cells. This is true for about a third
  of polygons, and never for a point.

The `gridjoin` metadata gives each variable's weight and the
`population_count` year used for each survey year. Per variable it gives how many home locations and years
had no valid cell (`on_nan_cells`) and how many the fallback filled
(`filled`).

The fallback radius is 50 km, about two cells, because some masks are
eroded at coasts. For `gdp_per_capita`, the nearest valid cell is 35 km
from Nice and 43 km from Montevideo. For the 143k home locations and
years of all sources, the fallback filled about 8,100 for GDP and 6,800
for travel time, and at most 23 stayed null for any one variable.
