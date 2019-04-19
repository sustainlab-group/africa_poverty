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

**lsms_deltas_pairs.csv**

- 1728 rows (+1 header), 5 columns
- columns: `['country', 'year1', 'year2', 'index1', 'index2']`
- each row refers to a pair of LSMS data points at the same location over 2 different years
- each "index" refers to the index in each np.array in `image_hists_lsms.npz`
- only includes the "forward" direction for each pair, i.e. year1 < year2
- created in `data_analysis/lsms.ipynb`

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
