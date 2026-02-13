### Data Status

|  source           |     | persons  | years     | label quality | data availability  |
| ----------------- |---- | -------- |-----------|---------------|--------------------|
| NTS               | UK  | 400k     | 02-23     | A             | [request](https://ukdataservice.ac.uk/)             |
| CMAP              | US  | 30k      | 17-19     | A-            | [data](https://github.com/CMAP-REPOS/mydailytravel) |
| NHTS              | US  | 1m       | 01,09,17,22 | A           | [data](https://nhts.ornl.gov/downloads) & [docs](https://nhts.ornl.gov/documentation) |
| Queensland        | AUS | 100k     | 12-24     | A-            | [data](https://www.data.qld.gov.au/dataset/queensland-household-travel-survey-series) |
| Melbourne         | AUS | 100k     | 12 -> 25  | B+            | [here](https://opendata.transport.vic.gov.au/dataset/victorian-integrated-survey-of-travel-and-activity-vista) |
| LTDS              | UK  | 100k     | 19 -> 24  | B+            | request from TfL |
| **Metropolitan (US datasets)** :   ||||                        | [data](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-metropolitan-travel-survey-archive) |
| California        | US  | 40k      | 01        | OK?           |
| LA                | US  | ?        | 01        | BAD?          |
| Seattle           | US  | 37k      | 00/02     | OK?           |
| SanFran           | US  | 35k      | 00        | OK?           |
| NY                | US  | 27k      | 98        | OK?           |
| Philly            | US  | 10k      | 00        | OK?           |
| Pheonix           | US  | 10k      | 02        | OK?           |
| Baltimore         | US  | 8k       | 01        | OK?           |
| Indiana           | US  | 8k       | 07/08     | OK?           |
| Spokane           | US  | 7k       | 05        | BAD?          |
| Idaho             | US  | 6k       | 02        | OK?           |
| Columbia          | US  | ~3k      | 07        | OK?           |
| Anchorage         | US  | 3k       | 01        | OK?           |

_________________________

### Labels

Age:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | 0-4, 5-12, 13-15, 16-17, 18-44, 45-64, 65+|
| Metropolitan      |           |
| NHTS              | numeric   |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | numeric & interpolated |

Sex:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | m,f,o     |
| Metropolitan      |           |
| NHTS              | m,f       |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | m,f       |

Education:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | detailed |
| Metropolitan      |           |
| NHTS              | detailed  |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING   |

Disability:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | visual, hearing, mobility, chair, mental |
| Metropolitan      |           |
| NHTS              | yes, no difficulty traveling or individual columns for each disability         |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING         |

Can Wfh:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | yes, no |
| Metropolitan      |           |
| NHTS              | Days per week worked from home |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING   |

Employmnet status:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | retired, volunteer, home, unemployed, student, etc |
| Metropolitan      |           |
| NHTS              | retired, volunteer, home, unemployed, student, etc |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | employed, unemployed, student  |

Industry:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | agriculture, utilities, construction, etc |
| Metropolitan      |           |
| NHTS              | ---       |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | detailed  |

Race:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | w, aab, asian, etc |
| Metropolitan      |           |
| NHTS              | w, b, asian, other, from respondent | 
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING   |

Has license:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | y, n      |
| Metropolitan      |           |
| NHTS              | derived drives as y, n |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | y,n       |

Relationship to respondent:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | spouse, father, child, self, etc|
| Metropolitan      |           |
| NHTS              | spouse, father, child, self, etc|
| Queensland (AUS)  |           |
| Melbourne (AUS)   | spouse, father, child, self, etc|


HH num people:
|                   | raw       |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | numeric   |
| Metropolitan      |           |
| NHTS              | numeric   |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | numeric   |

HH income:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | $0-15k, 25, 30, 35, 50, 60, 75, 100, 150, +|
| Metropolitan      |          |
| NHTS              | $0-10k, 15, 25, 35, 50, 75, 100, 125, 150, 200, + |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | detailed  |

HH type:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | detached, apartment, etc|
| Metropolitan      |           |
| NHTS              | None          |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING   |

HH ownership:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | own, rent, etc|
| Metropolitan      |           |
| NHTS              | own, rent, other |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | own, rent, other |

Num vehicles:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | numeric   |
| Metropolitan      |           |
| NHTS              | numeric   |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | numeric   |

Urban rural:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              |           |
| Metropolitan      |           |
| NHTS              | urban, rural |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | urban, rural |

HH composition:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              |           |
| Metropolitan      |           |
| NHTS              | 1a, 2+a, 1a1c, etc, also by age of child |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | MISSING   |

___________________

### Vehicles

Type/body:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | car, van, suv, truck, etc |
| Metropolitan      |           |
| NHTS              |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | detailed for hh |

___________________

### Locations

Type/body:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP              | tract, county, state FIPS codes and tract lat/lon |
| Metropolitan      |           |
| NHTS              |   ???     |
| Queensland (AUS)  |           |
| Melbourne (AUS)   | lgas      |

