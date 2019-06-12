import os

import numpy as np
import pandas as pd


def main(survey_path: str, out_dir: str):
    '''
    Args
    - survey_path: str, path to complete survey CSV file
    - out_dir: str, path to output directory
    '''
    os.makedirs(out_dir, exist_ok=True)
    data = pd.read_csv(survey_path, float_precision='high')

    # convert URBAN_RURA column from string to int: 'U' (urban) => 1, 'R' (rural) => 0
    data.loc[:, 'URBAN_RURA'] = data['URBAN_RURA'].map({'U': 1, 'R': 0})

    use_years = set(range(2009, 2017+1))
    digits = [str(i) for i in range(10)]

    # survey IDs are [cname][year][optional 'a'], e.g. "BF2010" or "UG2011a"
    def svyid_to_year(x):
        return int(''.join([c for c in x if c in digits]))

    # locations are (lat, lon) tuples
    unused_survey_ids = []

    survey_ids = set(data['svyid'])
    num_surveys = len(survey_ids)

    for i, survey_id in enumerate(survey_ids):
        survey_data = data.loc[data['svyid'] == survey_id]
        survey_country = survey_data['country'].iloc[0]
        survey_year = svyid_to_year(survey_id)
        print(f'{survey_country} ({survey_id}): {i} / {num_surveys}')

        # only use entries whose latitude coordinates that are not NaN
        if survey_data['LATNUM'].isna().all():
            print(f'- {survey_id} contains no non-NaN-located clusters, skipping')
            unused_survey_ids.append(survey_id)
            continue
        elif survey_id[-1] == 'a':
            print(f'- {survey_id} is AIS, skipping')
            unused_survey_ids.append(survey_id)
            continue
        elif survey_year not in use_years:
            print(f'- {survey_id} out of desired year range, skipping')
            unused_survey_ids.append(survey_id)
            continue

        # drop any rows with NaN values
        nrows_with_nans = survey_data.isna().any(axis=1).sum()
        if nrows_with_nans > 0:
            print(f'- skipping {nrows_with_nans} rows with NaN')
            for column in survey_data.columns:
                num_nans_in_col = survey_data[column].isna().sum()
                if num_nans_in_col > 0:
                    print(f'    {column}: {num_nans_in_col} NaN values')
            survey_data = survey_data.dropna(axis=0)

        assert np.all(survey_data['country'] == survey_country)
        country_str = survey_country.lower().replace(' ', '_').replace("'", '_')
        survey_out_name = f'{country_str}_{survey_year}.csv'
        survey_out_path = os.path.join(out_dir, survey_out_name)

        # sort the data by increasing cluster number
        survey_data = survey_data.sort_values(by='cluster', ascending=True)

        # save CSV: pandas uses float64 which maintains enough precision for all of our numbers
        columns = ['cluster', 'LATNUM', 'LONGNUM', 'wealth', 'wealthpooled', 'wealthpooled5country', 'households', 'URBAN_RURA']
        header = ['cluster_index', 'lat', 'lon', 'wealth', 'wealthpooled', 'wealthpooled5country', 'households', 'urban_rural']
        survey_data.to_csv(survey_out_path, columns=columns, header=header, index=False)

    # save a list of unused survey IDs
    with open(os.path.join(out_dir, 'unused.txt'), 'w') as f:
        f.write('\n'.join(unused_survey_ids))


if __name__ == '__main__':
    survey_path = '../data/dhs_wealth_index.csv'
    out_dir = '../data/dhs_surveys/'
    main(survey_path=survey_path, out_dir=out_dir)
