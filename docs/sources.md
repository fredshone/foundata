# Sources

Each source reads its raw files from its own folder under the data root, for
example `~/Data/foundata/NHTS/`. `data_dir_template/` mirrors the expected
layout with toy files. Access is either **open** (data sharing is safe and is permited) or
**request** (registration or an application is needed). `foundata batch
--open` runs only the open sources.

| Source | Country | Years | Coverage | Access | Plans |
|---|---|---|---|---|---|
| [NTS](#nts) | UK | 2002–24 | National (GB; England from 2013) | Request | 2.5M |
| [KTDB](#ktdb) | KR | 2010, 16, 21 | National | Open | 2.2M |
| [INDIA](#india) | IN | 2019, 24 | National (time use) | Open | 0.9M |
| [NHTS](#nhts) | US | 2001, 09, 17, 22 | National | Open | 0.6M |
| [FRANCE](#france) | FR | 2008–23 | 85 city/regional surveys | Open (via MobiSurvStd) | 0.6M |
| [ODIN](#odin) | NL | 2018–24 | National | Request | 0.4M |
| [MEXICO](#mexico) | MX | 2017 | Mexico City metro | Open | 0.3M |
| [OVIN](#ovin) | NL | 2010–17 | National | Request | 0.3M |
| [MON](#mon) | NL | 2005–07 | National | Request | 0.2M |
| [VISTA](#vista) | AU | 2012–24 | Victoria | Open | 90k |
| [ARGENTINA](#argentina) | AR | 2009–13 | 5 metro areas | Open | 86k |
| [BRAZIL](#brazil) | BR | 2017 | São Paulo metro | Open | 84k |
| [LTDS](#ltds) | UK | 2019–25 | London | Request | 61k |
| [CHILE](#chile) | CL | 2012–13 | Santiago metro | Open | 57k |
| [QHTS](#qhts) | AU | 2019–25 | Queensland | Open | 49k |
| [MONGOLIA](#mongolia) | MN | 2011, 19, 23 | National (time use) | Request | 46k |
| [CMAP](#cmap) | US | 2017–19 | Chicago metro | Open | 26k |
| [URUGUAY](#uruguay) | UY | 2016 | Montevideo metro | Open | 5k |

Plan counts are from the latest batch run and are after filtering.

Some limitations apply to many sources:

- **Weekday only.** Many city surveys cover only weekdays, and
  `day = unknown` is common.
- **Estimated distances.** Where a survey records no distance, it is
  estimated from duration × mode speed or from zone centroids. These
  distances are not comparable with reported or routed ones.
- **Sampled values.** Income bands, age bands and, in LTDS, trip times are
  sampled within their recorded range, so values below the band's
  resolution are noise.
- **Time-use surveys** (India, Mongolia) infer trips from changes of
  activity or place. They have coarse times and few or no distances or
  modes.

---

## Europe

### NTS
UK National Travel Survey, DfT (Great Britain to 2012, England from 2013).

- **Access:** request, End User Licence from the [UK Data Service](https://ukdataservice.ac.uk/).
- **Files:** `NTS/tab/{household,individual,day,trip,stage}_eul_2002-2024.tab`
  (tab-separated, `NA` for missing).
- **Notes:** this is a 7-day diary, and each diary day becomes its own plan,
  which makes NTS the largest source. Home location is given only by region
  (about 70 km scale) and trips have no locations. Access and egress
  distance is derived from stages. "Day trip / just walk" purposes are
  resolved to the previous activity. `relationship` is only about 56%
  complete.

### LTDS
London Travel Demand Survey, TfL.

- **Access:** request from the TfL LTDS team.
- **Files:** `LTDS/LTDS{1920,2223,2324,2425}/` containing `Household.csv`,
  `person.csv`, `person data.csv`, `Trip.csv`, `Stage.csv` and
  `HABORO_T.csv`.
- **Notes:** trip times are released to the hour, with durations. Minute
  times are **sampled** to fit the durations and the trip order, so
  individual times are synthetic. Locations are at borough (LAD) level.
  Income is only about 45% complete.

### ODIN
Onderweg in Nederland, CBS/RWS.

- **Access:** request via [DANS](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/SS/TR1TUW).
- **Files:** `ODIN/<year>/ODiN<year>_Databestand*.tab` (the 2023 and 2024
  `.csv` files are also tab-separated).
- **Notes:** one respondent per household, so `hid = pid`. The data has one
  row per ride, and these are linked into trips. Age is sampled from age
  classes in some years. Income deciles are converted to gross euros. Home
  location is the 4-digit postcode (2018–23), otherwise the municipality.
  The 2024 release groups small municipalities into codes 9000+ for
  privacy, and these are treated as unknown. Weather is added from
  regional data.

### OVIN
Onderzoek Verplaatsingen in Nederland, the predecessor of ODIN.

- **Access:** request via [DANS](https://ssh.datastations.nl/).
- **Files:** `OVIN/<year>/OViN<year>_Databestand*.sav` (SPSS).
- **Notes:** respondents with foreign, truck or holiday trips are dropped.
  Professional travel has no trip detail, so it is kept only as time at
  work. Disposable-income bands are converted to gross. Trip urbanity comes
  from the respondent's home municipality.

### MON
Mobiliteitsonderzoek Nederland (MON), the predecessor of OVIN.

- **Access:** request via [DANS](https://ssh.datastations.nl/).
- **Files:** `MON/<year>/.../Databestand MON <year>.sav` (SPSS).
- **Notes:** whole households are sampled. Household income is summed from
  members' net income bands and converted to gross, and is null for about
  24% of persons. There is no migration background or relationship field.

### FRANCE
CEREMA city surveys (EMD, EDGT, EDVM, EMC²) and the Île-de-France EGT 2020,
via [Progedo](https://data.progedo.fr/) and [MobiSurvStd](https://mobisurvstd.github.io/MobiSurvStd/).

- **Access:** request via [Progedo](https://data.progedo.fr/). Survey deails are documented and converted to a standard format by [MobiSurvStd](https://mobisurvstd.github.io/MobiSurvStd/). Each survey has
  its own licence.
- **Files:** `FRANCE/<survey>/` containing `households`, `persons`, `trips`
  and `legs` `.parquet` files and `metadata.json`. Every folder that has a
  `metadata.json` is loaded.
- **Notes:** this source combines 85 surveys. The national EMP 2019 is
  excluded because its diary day is always a travel day. Coverage is mostly
  weekdays. Phone surveys cover only 1–2 members per household. Income is
  available for EGT 2020 only (about 1% overall). Weights are normalised
  within each survey. Distances are the survey's own, or straight line
  × 1.3.

## Americas

### NHTS
US National Household Travel Survey.

- **Access:** open, with [data](https://nhts.ornl.gov/downloads) and
  [docs](https://nhts.ornl.gov/documentation).
- **Files:** `NHTS/<year>/`. Use `HHPUB`, `PERPUB` and `DAYPUB` for 2001,
  the `*V2PUB` files for 2009 and 2022, and `hhpub`, `perpub` and
  `trippub` for 2017 (all `.csv`).
- **Notes:** this is a one-day diary. Home location is only CBSA, MSA or
  state (median scale about 170 km) and trips have no locations. Income is
  sampled from bands. Distances are reported in miles and converted to km.

### CMAP
Chicago Metropolitan Agency for Planning, My Daily Travel.

- **Access:** open ([GitHub](https://github.com/CMAP-REPOS/mydailytravel)).
- **Files:** `CMAP/{household,person,place,location}.csv`.
- **Notes:** small, but with the best attribute coverage. It is the only US
  source with exact travel dates, so it has weather. Locations are census
  tracts, and urbanity comes from a tract rural/suburban/urban
  classification.

### MEXICO
Encuesta Origen Destino en Hogares 2017, INEGI.

- **Access:** open ([INEGI](https://www.inegi.org.mx/programas/eod/2017/#microdatos)).
- **Files:** `MEXICO/{THOGAR,TSDEM,TVIAJE,TTRANSPORTE}.DBF`.
- **Notes:** each person has a weekday diary (Tuesday–Thursday, but the day
  is not recorded) and a Saturday diary. There is no income. Distances are
  estimated from stage durations. EOD districts were never published as
  polygons, so locations are `estimated` (the union of the overlapping
  municipalities, about 6 km scale).

### BRAZIL
Pesquisa Origem Destino 2017, Metrô São Paulo.

- **Access:** open ([Metrô SP](https://transparencia.metrosp.com.br/dataset/pesquisa-origem-e-destino/resource/4362eaa3-c0aa-410a-a32b-37355c091075)).
- **Files:** `BRAZIL/OD-2017/Banco de Dados-OD2017/OD_2017_v1.dbf` (one flat
  table) and `Mapas-OD2017/Shape-OD2017/Zonas_2017_region.dbf`.
- **Notes:** weekdays only. About 56% of incomes are imputed by the survey.
  Homes and trip ends are coordinates (the 50 m precision is assumed). There
  is no driving licence field.

### CHILE
Encuesta Origen Destino Santiago 2012, SECTRA.

- **Access:** open, as parquet from [palomatrix-dtpm](https://github.com/PLUMAS-research/palomatrix-dtpm/tree/main/src/palomatrix/eod/datos).
- **Files:** `CHILE/{hogares,personas,viajes,etapas,vehiculos,codigos}.parquet`.
- **Notes:** covers all days of the week, including the summer season.
  About 68% of households have some member income imputed by the survey.
  Distances are straight lines between block or zone centroids. Locations
  are coordinates.

### ARGENTINA
Household mobility surveys, Ministerio de Transporte (DGPPSE): Buenos Aires
(ENMODO), Salta, Santa Fe–Paraná, Neuquén–Cipolletti and
Resistencia–Corrientes.

- **Access:** open ([argentina.gob.ar](https://www.argentina.gob.ar/transporte/dgppse/publicaciones/encuestas)).
- **Files:** `ARGENTINA/<survey>/...` containing SPSS `.sav` household,
  person, trip and stage (modes) tables for each survey (see
  `data_dir_template`).
- **Notes:** weekdays only, with `day` only about 26% known. Weights are
  normalised within each survey. Distances are estimated from stage
  durations. Short personal trips have no times. There are no trip
  locations.

### URUGUAY
Encuesta Origen Destino Montevideo 2016.

- **Access:** open ([Montevideo](https://montevideo.gub.uy/tipo/area-tematica/movilidad/observatorio-de-movilidad)).
- **Files:** `URUGUAY/EOD2016/Base {Hogar Habitos,Personas,Viajes,Etapas}.csv`
  (`;`-separated, decimal comma) and the `Dicc *.txt` codebooks.
- **Notes:** one weekday diary (Monday–Friday, but the day is not recorded).
  There is no income. Distances come from blocks walked or census segment
  centroids. This is the smallest source.

## Asia–Pacific

### KTDB
Korea Transport Database national travel surveys.

- **Access:** open ([KTDB](https://www.ktdb.go.kr/www/index.do)). Stay on
  the Korean-language site.
- **Files:** `KTDB/2021/{persons,trips}.csv`. For 2016, the household and
  weekend survey files (CP949 encoding, one row per trip). For 2010,
  `household/` and `weekend/` tables (see `data_dir_template`).
- **Notes:** there are **no weights**. The 2021 personal survey has no
  households (`hid = pid`), was mostly collected on Thursdays, and has no
  income for young people. Weekend surveys give a Saturday and a Sunday
  plan for each person. Children under 5 (2016) or under school age (2010)
  are not surveyed. 2010 trip ends are only at 시군구 level, so distances
  come from centroids or duration × speed, and the 2010 weekday date is
  unknown.

### INDIA
Time Use Survey, NSO/MoSPI.

- **Access:** open ([microdata.gov.in](https://microdata.gov.in/nada/index.php/catalog/236/related-materials)).
- **Files:** `INDIA/TUS{2019,2024}/` containing the `.Nesstar` file and its
  DDI `.xml`, plus `INDIA/icatus_2016_codes.csv`.
- **Notes:** this is a **time-use** survey. Diaries run 04:00–04:00 in
  30-minute slots, and trips are the ICATUS "travel" episodes, so times are
  coarse. There are no distances and no vehicles. Home location is the
  district (about 35 km scale) and trips have no locations. Nearly half of
  the attributes are missing.

### MONGOLIA
Time Use Survey, National Statistics Office of Mongolia.

- **Access:** open ([NSO NADA](https://nada.nso.mn/en/catalog/53/get-microdata), free registration).
- **Files:** `MONGOLIA/2023/*.dta` (Stata), and `MONGOLIA/{2019,2011}/*.sav`
  (SPSS).
- **Notes:** this is a **time-use** survey. Many trips are implied by a
  change of place. People kept 2–3 diary days in 2011 and 2019, and each
  day is a plan. Mode is recorded only for travel in 2023 and 2011, and
  never in 2019. 2011 has no clock times. There is no income or distance.
  Rural bagh locations are estimated from their soum. The 2015 round is
  not loaded.

### VISTA
Victorian Integrated Survey of Travel and Activity, Department of
Transport and Planning.

- **Access:** open ([data.vic](https://opendata.transport.vic.gov.au/dataset/victorian-integrated-survey-of-travel-and-activity-vista)).
- **Files:** `VISTA/2012-2020/*_vista_2012_2020_lga_v1.csv`,
  `VISTA/2022-2023/*_vista_2022_2023.csv` and
  `VISTA/2023-2024/*_vista_2023_2024.csv` (household, person and trip
  files for each).
- **Notes:** locations are at LGA level (about 6 km scale). There is no
  education field.

### QHTS
Queensland Household Travel Survey, TMR.

- **Access:** open ([data.qld](https://www.data.qld.gov.au/dataset/queensland-household-travel-survey-series)).
- **Files:** `QHTS/{2019-22,2022-25}/` containing `1_QTS_HOUSEHOLDS.csv`,
  `2_QTS_PERSONS.csv` and `5_QTS_TRIPS.csv`.
- **Notes:** raw times count from 04:00 and are shifted to midnight. Homes
  and trip ends are SA1 zones (about 250 m scale). Income is only about 39%
  complete and there is no education field.
