### Data Status

|                   | persons   | years     | label availability   |
| ----------------- | --------- |-----------|----------------------|
| NTS (UK)          |           |           |                   |
| CMAP (US)         | 30k       | 17,18,19  | A                 |
| Metropolitan (US) |           |           |                   |
| NHTS (US)         |           | 01,09,17,22 |                 |
| Queensland (AUS)  |           |           |                   |
| Melbourne (AUS)   |           |           |                   |

_________________________

### Persons

Age:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | 0-4, 5-12, 13-15, 16-17, 18-44, 45-64, 65+|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Sex:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | m,f,o|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Education:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | visual, hearing, mobility, chair, mental |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Disability:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | visual, hearing, mobility, chair, mental |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Can Wfh:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | yes, no |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Employmnet status:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | retired, volunteer, home, unemployed, student, etc |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Industry:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | agriculture, utilities, construction, etc |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Race:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | w, aab, asian, etc |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Has license:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | y, n |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Relationship to respondent:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | spouse, father, child, self, etc|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

___________________

### HHs

Num people:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | numeric|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Income:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | $0-15k, 25, 30, 35, 50, 60, 75, 100, 150, +|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Type:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | detached, apartment, etc|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Ownership:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | own, rent, etc|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

Num vehicles:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | numeric|
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

___________________

### Vehicles

Type/body:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | car, van, suv, truck, etc |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

___________________

### Locations

Type/body:
|                   | raw    |
| ----------------- | --------- |
| NTS (UK)          |           |
| CMAP (US)         | tract, county, state FIPS codes and tract lat/lon |
| Metropolitan (US) |           |
| NHTS (US)         |           |
| Queensland (AUS)  |           |
| Melbourne (AUS)   |           |

_________________
Resources:

CMAP (Chicago, 2017-2019) data and some code sitting in a repo [here](https://github.com/CMAP-REPOS/mydailytravel).

Metropolitan (US) surveys - loads sitting [here](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-metropolitan-travel-survey-archive), mostly old and variable documentation, but can filter for recent stuff.

NHTS (US) 2022, 2017, 2009... all [here](https://nhts.ornl.gov/downloads) including [docs](https://nhts.ornl.gov/documentation).

Queensland, [SQL](https://www.data.qld.gov.au/dataset/queensland-household-travel-survey-series)? and not sure about docs.

Melbourne, some data found via [here](https://opendata.transport.vic.gov.au/dataset/victorian-integrated-survey-of-travel-and-activity-vista).