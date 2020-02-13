*preds_incountry.csv*, *preds_ooc.csv*
- columns: `lat,lon,country,country_year,urban,wealthpooled,[different models]`
- each row is a DHS survey location with label and predictions from each model
- "ooc" = "out of country"

*r2_ooc.csv*, *r2_incountry.csv*
- columns: `country,subset,metric,[different models]`
- `country` contains the usual countries, as well as `overall`
- `subset` contains `['all', 'urban', 'rural']`
- `metric` contains `['r2', 'R2', 'mse', 'rank']`
  - 'r2': squared Pearson correlation coefficient
  - 'R2': coefficient of determination
  - 'mse': mean squared-error
  - 'rank': Spearman rank correlation coefficient
- we are breaking down how each model (trained on all data) performs on each (country, all/urban/rural) slice of data
- for overall model results, look at rows starting with ('overall', 'all', 'r2', ...)

*r2_wealth_ooc.csv*, *r2_wealth_incountry.csv*
- columns: `wealthpooled,[different models]`
- rows are sorted by increasing `wealthpooled` value
- test r^2 values are the cumulative squared correlation between predictions and labels
- 1st row is NaN (as expected), since the correlation of a single (prediction, label) pair is undefined

*preds_keep.csv*
- columns: `lat,lon,country,country_year,urban,wealthpooled,[different models]`
- each row is a DHS survey location with label and predictions from each model
- models are named `{model}, keep{k}, seed{s}`, e.g. `Resnet-18 MS+NL, keep0.25, seed456`
- `k` = fraction of training data used, `k` in `[0.05, 0.1, 0.25, 0.5, 1.0]`
- `s` = random seed used for selecting the fraction of training data, `s` in `[123, 456, 789]`
- NOTE: for the full-data models (`k=1.0`), there wasn't any data selection procedure involved, so I just arbitrarily gave it seed `s=123` in the CSV

*r2_keep.csv*
- columns: `model,keep_frac,seed,country,r2,R2,mse`
- each row is a different `model`/`keep_frac`/`seed`/`country` combination
- `keep_frac` is identical to `k` above
- `seed` is identical to `s` above
- `country` contains the usual countries, as well as `overall`
- for the overall model results, look at rows starting with `[model name], keep1.0, seed123, overall, ...`

The models should include the following (listed in no particular order):
```
Resnet-18 MS
Resnet-18 MS Transfer
Resnet-18 MS+NL
Resnet-18 MS+NL concat
Resnet-18 NL
Resnet-18 RGB
Resnet-18 RGB Transfer
Resnet-18 RGB+NL
Resnet-18 RGB+NL concat
Ridge MS hist
Ridge MS+NL hist
Ridge NL center scalar
Ridge NL hist
Ridge NL mean scalar
Ridge RGB hist
Ridge RGB+NL hist
```
