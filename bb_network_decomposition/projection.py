import numpy as np
import scipy
import scipy.stats
import sklearn
import sklearn.cross_decomposition
import sklearn.decomposition
import sklearn.preprocessing

import bb_network_decomposition.data
import bb_network_decomposition.evaluation


def scale_projection_by_day(factor_df, column, inplace=False, how="percentiles"):
    if not inplace:
        factor_df = factor_df.copy()

    if not how:
        return factor_df

    if 'age' in factor_df.columns:
        factor_df[column] *= np.sign(scipy.stats.pearsonr(factor_df.age, factor_df[column])[0])

    for day in factor_df.day.unique():
        idxer = (
            np.argwhere((factor_df.day == day).values)[:, 0],
            np.argwhere(factor_df.columns == column)[0, 0],
        )

        if how == "percentiles":
            factor_df.iloc[idxer] -= np.percentile(factor_df.iloc[idxer], 5)
            factor_df.iloc[idxer] /= np.percentile(factor_df.iloc[idxer], 95)
            factor_df.iloc[idxer] *= 40
        elif how == "minmax":
            factor_df.iloc[idxer] = sklearn.preprocessing.MinMaxScaler().fit_transform(
                factor_df.iloc[idxer][:, None]
            )[:, 0]
            factor_df.iloc[idxer] *= 40
        else:
            assert False

    return factor_df


def extract_and_scale_factors(
    factor_df, projection, how, factor_prefix="network", inplace=False
):
    if not inplace:
        factor_df = factor_df.copy()

    num_components = projection.shape[-1]

    factor_df[f"{factor_prefix}_age"] = projection[:, 0]
    scale_projection_by_day(factor_df, f"{factor_prefix}_age", inplace=True, how=how)

    for f in range(num_components):
        column_name = f"{factor_prefix}_age_{f}"
        factor_df[column_name] = projection[:, f]
        scale_projection_by_day(factor_df, column_name, inplace=True, how=how)

    return factor_df


def get_pca_projection(
    factor_df,
    num_components=2,
    inplace=False,
    scale_by_day="percentiles",
    return_pca=False,
):
    if not inplace:
        factor_df = factor_df.copy()

    factors = bb_network_decomposition.data.factors_from_dataframe(factor_df)
    pca = sklearn.decomposition.PCA(n_components=num_components).fit(factors)
    projection = pca.transform(factors)

    factor_df = extract_and_scale_factors(
        factor_df, projection, how=scale_by_day, inplace=True
    )

    factor_df.sort_values("date", inplace=True)

    if return_pca:
        return factor_df, pca
    else:
        return factor_df


def get_cca_projection(factor_df, location_df=None, num_components=2, inplace=False,
                       scale_by_day='percentiles', cca=None, return_cca=False,
                       target_cols=bb_network_decomposition.evaluation.location_labels):
    if not inplace:
        factor_df = factor_df.copy()

    if location_df is not None:
        merged_df = bb_network_decomposition.data.merge_location_data(factor_df, location_df)
    else:
        merged_df = factor_df

    factors = bb_network_decomposition.data.factors_from_dataframe(merged_df)
    targets = merged_df[target_cols].values

    if targets.ndim == 1:
        targets = targets[:, None]

    if cca is None:
        cca = sklearn.cross_decomposition.CCA(n_components=num_components).fit(
            factors, targets
        )
    factor_projection, location_projection = cca.transform(factors, targets)

    if location_projection.ndim == 1:
        location_projection = location_projection[:, None]

    merged_df = extract_and_scale_factors(
        merged_df,
        factor_projection,
        factor_prefix="network",
        how=scale_by_day,
        inplace=True,
    )
    merged_df = extract_and_scale_factors(
        merged_df,
        location_projection,
        factor_prefix="location",
        how=scale_by_day,
        inplace=True,
    )

    merged_df.sort_values("date", inplace=True)

    if return_cca:
        return merged_df, cca
    else:
        return merged_df
