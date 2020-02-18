import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import sklearn.cross_decomposition
import sklearn.decomposition
import sklearn.preprocessing

import bb_network_decomposition.data
import bb_network_decomposition.evaluation


def scale_projection_by_day(factor_df, column, inplace=False, how='percentiles'):
    if not inplace:
        factor_df = factor_df.copy()

    if not how:
        return factor_df

    factor_df[column] *= np.sign(scipy.stats.pearsonr(factor_df.age, factor_df[column])[0])

    for day in factor_df.day.unique():
        idxer = np.argwhere((factor_df.day == day).values)[:, 0], np.argwhere(factor_df.columns == column)[0, 0]

        if how == 'percentiles':
            factor_df.iloc[idxer] -= np.percentile(factor_df.iloc[idxer], 5)
            factor_df.iloc[idxer] /= np.percentile(factor_df.iloc[idxer], 95)
            factor_df.iloc[idxer] *= 40
        elif how == 'minmax':
            factor_df.iloc[idxer] = sklearn.preprocessing.MinMaxScaler().fit_transform(factor_df.iloc[idxer][:, None])[:, 0]
            factor_df.iloc[idxer] *= 40
        else:
            assert False

    return factor_df


def get_pca_projection(factor_df, num_components=2, inplace=False, scale_by_day='percentiles'):
    if not inplace:
        factor_df = factor_df.copy()

    factors = bb_network_decomposition.data.factors_from_dataframe(factor_df)
    projection = sklearn.decomposition.PCA(n_components=num_components).fit_transform(factors)

    factor_df['network_age'] = projection[:, 0]
    scale_projection_by_day(factor_df, 'network_age', inplace=True, how=scale_by_day)

    for f in range(num_components):
        column_name = 'network_age_{}'.format(f)
        factor_df[column_name] = projection[:, f]
        scale_projection_by_day(factor_df, column_name, inplace=True, how=scale_by_day)

    return factor_df


def get_cca_projection(factor_df, location_df, num_components=2, inplace=False,
                       scale_by_day='percentiles', cca=None, return_cca=False,
                       target_cols=bb_network_decomposition.evaluation.location_labels):
    if not inplace:
        factor_df = factor_df.copy()

    merged_df = bb_network_decomposition.data.merge_location_data(factor_df, location_df)

    factors = bb_network_decomposition.data.factors_from_dataframe(merged_df)
    targets = merged_df[target_cols].values

    if cca is None:
        cca = sklearn.cross_decomposition.CCA(n_components=num_components).fit(factors, targets)
    factor_projection, location_projection = cca.transform(factors, targets)

    merged_df['network_age'] = factor_projection[:, 0]
    scale_projection_by_day(merged_df, 'network_age', inplace=True, how=scale_by_day)

    merged_df['location_age'] = location_projection[:, 0]
    scale_projection_by_day(merged_df, 'location_age', inplace=True, how=scale_by_day)

    for f in range(num_components):
        column_name = 'network_age_{}'.format(f)
        merged_df[column_name] = factor_projection[:, f]
        scale_projection_by_day(merged_df, column_name, inplace=True, how=scale_by_day)

        column_name = 'location_age_{}'.format(f)
        merged_df[column_name] = location_projection[:, f]
        scale_projection_by_day(merged_df, column_name, inplace=True, how=scale_by_day)

    if return_cca:
        return merged_df, cca
    else:
        return merged_df
