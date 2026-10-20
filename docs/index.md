# Foundata

Foundata turns household travel and time-use surveys from around the world
into one **harmonised** format. It is built to support *foundational* (or *worldwide* if you prefer) models of
human activity and travel. It is also a set of working loaders for anyone
who uses these openly available datasets.

Each survey uses its own file layout, coding and conventions. Foundata reads
the raw files and maps them all to one schema:

- **plans**: each person's 24-hour sequence of activities and the trips
  between them, from midnight to midnight;
- **attributes**: person, household and diary-day information (age,
  employment, income, vehicles, day of week, ...);
- **locations**: home and trip-end zones (or points), with their geometry.

The current batch covers 18 sources in 15 countries, about 8.5 million plans
and 18.5 million trips. See [Sources](sources.md).

## What harmonisation means here

- **Shared categories.** Activities, modes, employment, education and the
  other categorical fields are mapped to the fixed sets listed in
  [Output format](format.md). The mappings live in
  `configs/<source>/*_dictionary.yaml`.
- **Shared units.** Times are minutes since midnight, distances are in km,
  income is annual gross household income in euros, and weights are
  normalised to a mean of 1 within each source.
- **Validity.** Trips must be in time order and must not overlap,
  and each trip's origin activity must match the previous trip's
  destination activity. Implausible plans are removed. Attribute values are controlled using `configs/core/template.yaml`.
- **Missing data is allowed.** Not every survey records every attribute, so
  missing values are kept as nulls (for categorical variables converted to "unknwon"). The README has a per-source
  availability table.

## Limitations

The surveys are not directly comparable, even after harmonisation. They
differ in sampling, in which diary days they cover (many are weekday only),
in the age groups surveyed, in how precisely times and places are recorded,
and in whether values such as distance were measured or estimated. We use a broad variety of sampling, mapping and inferring to overcome these limitations as best as possible. Read
[Sources](sources.md) before you pool or compare them.

## Getting started

```bash
uv sync
foundata --help

# try the pipeline on the bundled toy data
foundata batch data_dir_template --output /tmp/out

# one source, or many combined
foundata run nhts ~/Data/foundata/NHTS
foundata batch ~/Data/foundata --open      # open-data sources only
```

You need to download the raw data yourself. `data_dir_template/` shows the
folder and file layout that is expected, and [Sources](sources.md) says where
to get each survey.

## Contents

- [Output format](format.md): output files, columns, types and value sets.
- [Sources](sources.md): each survey's coverage, how to access it, the files
  it needs, and its known limitations.
