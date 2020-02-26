import numpy as np
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
        factors = factor_df.iloc[idxer]

        if how == 'percentiles':
            factor_df.iloc[idxer] -= np.percentile(factors, 5)
            factor_df.iloc[idxer] /= np.percentile(factors, 95)
            factor_df.iloc[idxer] *= 40
        elif how == 'minmax':
            factor_df.iloc[idxer] = sklearn.preprocessing.MinMaxScaler().fit_transform(factors[:, None])[:, 0]
            factor_df.iloc[idxer] *= 40
        else:
            assert False

    return factor_df


def extract_and_scale_factors(factor_df, projection, how, factor_prefix='network', inplace=False):
    if not inplace:
        factor_df = factor_df.copy()

    num_components = projection.shape[-1]

    factor_df['{}_age'.format(factor_prefix)] = projection[:, 0]
    scale_projection_by_day(factor_df, '{}_age'.format(factor_prefix), inplace=True, how=how)

    for f in range(num_components):
        column_name = '{}_age_{}'.format(factor_prefix, f)
        factor_df[column_name] = projection[:, f]
        scale_projection_by_day(factor_df, column_name, inplace=True, how=how)

    return factor_df


def get_pca_projection(factor_df, num_components=2, inplace=False,
                       scale_by_day='percentiles', return_pca=False):
    if not inplace:
        factor_df = factor_df.copy()

    factors = bb_network_decomposition.data.factors_from_dataframe(factor_df)
    pca = sklearn.decomposition.PCA(n_components=num_components).fit(factors)
    projection = pca.transform(factors)

    factor_df = extract_and_scale_factors(factor_df, projection, how=scale_by_day, inplace=True)

    if return_pca:
        return factor_df, pca
    else:
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

    merged_df = extract_and_scale_factors(merged_df, factor_projection, factor_prefix='network',
                                          how=scale_by_day, inplace=True)
    merged_df = extract_and_scale_factors(merged_df, location_projection, factor_prefix='location',
                                          how=scale_by_day, inplace=True)

    if return_cca:
        return merged_df, cca
    else:
        return merged_df
