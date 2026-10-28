# Changelog

Notable changes to foundata, especially fixes and changes that affect the
outputs. Newest first. Add an entry under **Unreleased** with each change,
and give the section a date (or version) when a batch output is published.


## 2026-10-28 

### Changed

- **NTS diary sample and weight.** NTS now keeps only fully productive
  households (`W1 = 1`). Partially productive households include members who
  returned no diary, whose 7 blank days were loaded as stay-at-home plans
  (18,542 persons). The NTS `weight` is now the diary-sample weight `W2`
  instead of the interview weight `W3`. Persons with no trips all week are
  assumed to be at home every day; some may have been away (see
  `docs/sources.md`).

### Fixed

- **Trip and activity `seq` numbering.** cmap, odin, qhts, vista and some
  ovin plans numbered trips from 1, and filters that drop single trips
  (e.g. combining consecutive activities) left gaps, so 2.5M plans did not
  start at `seq` 0 and 313k rows skipped a number. Anything that read `seq`
  as a position could show a missing or non-home first activity. Trips are
  now ordered by start time and numbered 0, 1, 2, ... per plan, for each
  source (`fix.enumerate_trips`, in `run.process_source`) and again after
  sources are combined and filtered (in `run.runner`). Plans whose source
  order disagrees with their timings now fail the activity-consistency
  check.
- **Zero-length home activities.** Plans whose first trip leaves at minute 0
  had a 0-minute first home activity, so they looked like they started away
  from home (421 plans, e.g. NHTS and NTS midnight trips). The home-based
  filter (`filter.home_based`) now also removes plans whose first trip
  starts at 0 or whose last trip starts at 1440.
- **`--home-based` help text** described a trip-level filter; it now
  describes the plan-level one actually applied.

## 2026-10-27

### Added

- `hh_adults` and `hh_children` household counts for every source, from the
  survey's own counts where given, else counted from the household roster
  (null when the roster is incomplete or an age is unknown).
- `veh_per_adult` category (`0`, `0-1`, `1+`, `unknown`) from `vehicles` and
  `hh_adults`.
- Metadata output (`metadata.json`).

### Fixed

- Mexico: `EDAD` code 99 ("no sabe") is now an unknown age, not age 99.

## 2026-10-26

### Added

- Stratified sampler.

## 2026-10-23

### Changed

- Parquet is the default output format (`--csv` for CSV).

### Added

- Weighted grid join.

## 2026-10-22

### Added

- Survey year in the grid join.
- Location documentation.

### Changed

- Removed the `_year` columns from the locations table.

## 2026-10-21

### Added

- World tensor join.
