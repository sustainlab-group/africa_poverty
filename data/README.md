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

**folders**
- contains all data necessary to reproduce figures
- also 
  - `dhs_jitter_results_knn_jitter.csv`
  - `DHS_output_2019-12-18_aggregatedforidentification.RDS`
  - `DHS_output_2019-12-18_countryvariance.RDS`
  - these are processed DHS summary stats for plotting since we can't release the microdata
  - `wc2.0_bio_10m_01.tif`
  - `wdi.csv`

**crosswalks/crosswalk_countries.csv**
- Crosswalk between ISO2, ISO3, World Bank country names, simplified country names (removing apostrophes, commaes etc), and an additional column of the names in the prediciton data. 

**dhs/floor_recode**
- crosswalk between the specific codes in DHS data (11, 21, etc) to a ranking of 1-5 of the quality of floor. The conversion between test and their DHS code can be found in DHS documentation. 
- There are also recode files for toilet and water. 

**download_locations/dhs_clusters.csv**
- list of unique clusters, their locations, labels, and the number of clusters

**output/overpass.csv**
- file with an overpass rate for several satellites between 2000 and 2018 for 500 randomly selected sites in sub-saharan africa.

**output/.**
- lsms_labels_index.csv: the labels for the lsms locations.
- lsms_labels_index_agg.csv: the cluster level labels for lsms locations at a time point. 
- lsms_labels_index_of_diffs.csv: the labels for the difference in lsms locations over time. Differs from the lsms_labels_index.csv file because it only uses households that are consistent over time. 
- geolevel2_ipums_dhs_indices_ipums.csv: the labels from DHS and IPUMS data, aggregated to the second geolevel, as designated by the IPUMS geo-files. 
- geolevel2_dhs_indices_gadm2.csv: labels from DHS, aggregated to the second geolevel, as defined by GADM.
- cluster_pred_dhs_indices_gadm2.csv: same as above but at cluster level. 

**overpass/dhs_sample_GEE.csv**
- for 500 randomly selected DHS cluster locations, the number of times each satellite captures the location per year. 
- dhs_sample_Planet.csv: same structure, but for PlanetScope and RapidEye.

**shapefiles/.**
- shapefiles at geo2 for the countries in the study from GADM
- a combined shapefile of all those countries.

**surveys/.**
- dhs_time.csv: For each country-year, a number of sampled individuals in each year. Compiled from https://dhsprogram.com/data/available-datasets.cfm
- population_time.csv: Annual population estimates for each country between 1960 and 2017, inclusive. Downloaded from the World Bank, variable "Population, total (SP.POP.TOTL)", version "2018 Oct". See the World Bank World Development Indicators Database Archives: [https://datacatalog.worldbank.org/dataset/wdi-database-archives](https://datacatalog.worldbank.org/dataset/wdi-database-archives).
- povcal_time_pop.csv: For each country-year, a number of sampled individuals in that year. Compiled from http://iresearch.worldbank.org/PovcalNet/povOnDemand.aspx, by finding the number of observations in each "detailed output" for each survey. For type C surveys (where consumption is at a group level and individual sample sizes aren't reported), "X" is recorded to indicate there was a survey but no sample size is given, in the code X is substituted on a country level for the largest sample seen in that country as that is the most conservative assumption.
- us_surveys_time.csv: the number of ppl sampled in surveys as pulled from:
  - ACS: https://www.census.gov/acs/www/methodology/sample-size-and-data-quality/sample-size/index.php
      calculated by summing the Final Interviews and Final Actual Interviews columns
  - AHS until 2015: https://www.census.gov/content/dam/Census/programs-surveys/ahs/publications/AHS%20Sample%20Determination%20and%20Decisions.pdf
  - AHS in 2017: https://www.census.gov/programs-surveys/ahs/about/methodology.html
  - AHS in 2019: https://www.reginfo.gov/public/do/PRAViewDocument?ref_nbr=201810-2528-002
  - CPS: http://www.census.gov/prod/2006pubs/tp-66.pdf
  - NSCG: https://www.nsf.gov/statistics/srvygrads/overview.htm
  - PSID 1: https://psidonline.isr.umich.edu/publications/Papers/tsp/2000-04_Imm_Sample_Addition.pdf
  - PSID 2: https://nsf.gov/news/special_reports/survey/index.jsp?id=income (interpolated between)
  - SIPP 1993: https://www2.census.gov/prod2/sipp/wp/SIPP_WP_203.pdf
  - SIPP: http://www.nber.org/sipp/2008/ch2_nov20.pdf. For 2014 sample we assume same size as 2008.
  All sample numbers that are household counts are multiplied by the mean household size in that year (as found at https://www.census.gov/data/tables/time-series/demo/families/households.html)

