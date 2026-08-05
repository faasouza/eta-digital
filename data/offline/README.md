# ETA-DIGITAL offline data

`raw/` preserves source files without modification. `interim/` is for parsed and standardized records. `processed/` is for validated, time-aligned training datasets. `schemas/` defines expected columns.

The two source `.xls` examples supplied for ETA Aquiraz contain quality-analysis and chemical-product structures. Before model training, the numerical historical export must include timestamps, raw-water measurements, applied PAC/POL dosages, and filtered-water outcomes aligned by treatment delay.

Do not commit credentials, personal data, or unrestricted operational exports to a public repository. Replace these examples with approved or anonymized data before production use.
