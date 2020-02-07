**dhs_wealth_index.csv**
- DHS survey data
- columns
  - "cluster": index of the village within a survey
  - "svyid": survey ID, includes country and year that the survey started
  - "wealthpooled": the "asset wealth index" (AWI) of each household, standardized across all surveys, then averaged to the cluster level
  - "wealthpooled5country": same as "wealthpooled" but normalized only in the 5 countries used in the original work by Michael and Neal (Uganda, Nigeria, Tanzania, Malawi, Rwanda)
  - "wealth": AWI standardized within each country at the household level, aggregated to the cluster level
  - "households": number of households surveyed in the cluster
  - "LATNUM": latitude
  - "LONGNUM": longitude
  - "URBAN_RURA": "U" for urban, "R" for rural
  - "year": year that the cluster was surveyed; the survey year corresponds to the start of the survey, but some clusters may be surveyed in later years

**image_hists_lsms.npz**

- created in `models/baselines_lsms.ipynb`
```
- 'image_hists': np.array, shape [N, C, nbins], type int64
- 'labels': np.array, shape [N], type float32, all labels
- 'locs': np.array, shape [N, 2], type float32, all locs
- 'years': np.array, shape [N], type int32, year for each image
- 'nls_center': np.array, shape [N], type float32, center nightlight value
- 'nls_mean': np.array, shape [N], type float32, mean nightlight value
```

**lsmsdelta_pairs.csv**

- 1539 rows (+1 header), 5 columns
- columns: `['lat', 'lon', 'year.x', 'country', 'year.y', 'index', 'index_diff', 'geolev1', 'geolev2', 'x', 'tfrecords_index.x', 'tfrecords_index.y']`
- each row refers to a pair of LSMS data points at the same location over 2 different years
- each "index" refers to the index in each np.array in `image_hists_lsms.npz`
- only includes the "forward" direction for each pair, i.e. year.x < year.y
- created in `data_analysis/lsms_merge_dfs.ipynb`

***lsms_incountry_folds.pkl***

- contains a dictionary of dictionaries: `incountry_folds[fold][split] = np.array`
- `fold`: one of `['A', 'B', 'C', 'D', 'E']`
- `split`: one of `['train', 'val', 'test']`
- the 'test' split of each fold is disjoint from the test splits of all other folds
- created in `data_analysis/lsms.ipynb`

**lsms_labels_agg.csv**

- 3020 rows (+1 header), 5 columns
- columns: `['lat', 'lon', 'year', 'country', 'index', 'ea_id']`
- each row corresponds to a village
- `index` column gives the asset wealth index averaged over wealth values at the village level
- created by Anne
