# External data and provenance

The author-designated source entry points are:

- [China National Environmental Monitoring Centre](https://www.cnemc.cn/)
- [Global River Water Quality Archive, Zenodo record 15335450](https://doi.org/10.5281/zenodo.15335450)

## Station files

Required fields and units:

| Field | Meaning | Unit required before loading |
|---|---|---|
| `time` or `Monitoring Time` | Sampling timestamp in a consistent, unambiguous timezone | ISO date and time |
| `water_temperature` | Water temperature | °C |
| `ph` | pH | 1 |
| `conductivity` | Electrical conductivity | µS/cm |
| `turbidity` | Turbidity | NTU |
| `nh4_n` | Ammonia nitrogen expressed as nitrogen | mg N/L |



## Monthly tables

Cross-basin functions require `basin,param,station,month,value,unit`. Use `YYYY-MM` for `month` and one row per station–indicator–month. `NH4N` uses a log1p transform; confirm that the parameter is ammonia nitrogen expressed as nitrogen and that its units are consistent. `pH` and other indicators remain on their original scales.
