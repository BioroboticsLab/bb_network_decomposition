import datetime

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.linear_model
import sklearn.model_selection

import bb_network_decomposition.data
import bb_network_decomposition.normalization
import bb_network_decomposition.projection
import bb_network_decomposition.spectral
from bb_network_decomposition.constants import (
    default_factors,
    location_labels,
    supplementary_labels,
)


def evaluate_network_factors(
    df,
    model=None,
    n_splits=25,
    groupby=None,
    labels=location_labels,
    factors=default_factors,
    scoring=None,
):
    def get_factor(factor):
        if "+" in factor:
            return np.stack([get_factor(f) for f in factor.split("+")], axis=1)
        is_log = "log" in factor
        if is_log:
            factor = factor[4:]
        X = df[factor].values
        if is_log:
            X = np.log1p(X)
        return X

    if model is None:
        model = sklearn.linear_model.LinearRegression()

    if scoring is None:
        scoring = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)

    if groupby is None:
        groups = None
        cv = sklearn.model_selection.ShuffleSplit(n_splits=n_splits)
    else:
        cv = sklearn.model_selection.GroupShuffleSplit(n_splits=n_splits)
        if groupby == "days":
            mapper = dict(zip(df.day.unique(), np.arange(len(df.day.unique()))))
            groups = df.day.apply(lambda day: mapper[day])
        elif groupby == "bees":
            mapper = dict(zip(df.bee_id.unique(), range(len(df.bee_id.unique()))))
            groups = df.bee_id.apply(lambda bee_id: mapper[bee_id])
        else:
            assert False

    regression_results_df = []
    for factor in factors:
        for label in labels:
            X = get_factor(factor)
            Y = df[label].values

            if X.ndim == 1:
                X = X[:, None]

            scores = sklearn.model_selection.cross_val_score(
                model, X, Y, cv=cv, groups=groups, scoring=scoring, n_jobs=-1
            )

            for score in scores:
                regression_results_df.append(
                    dict(variable=factor, target=label, r_squared=score)
                )

    regression_results_df = pd.DataFrame(regression_results_df)

    regression_pivot = regression_results_df.pivot_table(
        values="r_squared", index="variable", columns="target"
    )
    regression_pivot["mean"] = np.mean(regression_pivot.values, axis=1)
    regression_pivot = regression_pivot.sort_values("mean")

    return regression_results_df, regression_pivot


def get_timeshifted_df(
    df,
    factors=default_factors,
    labels=location_labels,
    days_into_future=0,
    min_age=0,
    max_age=100,
    return_unshifted_labels=False,
):
    df_shifted = df.copy()
    df_shifted.date -= datetime.timedelta(days=days_into_future)

    raw_factors = []
    for f in factors:
        if "+" in f:
            for f_part in f.split("+"):
                raw_factors.append(f_part)
        elif "log" in f:
            continue
        else:
            raw_factors.append(f)

    labels_unshifted = []
    for l in labels:
        label_name = l + "_unshifted"
        df[label_name] = df[l].copy()
        labels_unshifted.append(label_name)

    df_idxer = (df.age >= min_age) & (df.age < max_age)
    df_shifted = df_shifted[["date", "bee_id"] + labels].merge(
        df[df_idxer][["date", "bee_id", "day"] + raw_factors + labels_unshifted],
        on=("date", "bee_id"),
    )

    if return_unshifted_labels:
        return df_shifted, labels_unshifted
    else:
        return df_shifted


def evaluate_future_predictability(
    df,
    factors=default_factors,
    labels=location_labels,
    days_into_future=0,
    min_age=0,
    max_age=100,
    evaluation_kws={},
):
    df_shifted = get_timeshifted_df(
        df, factors, labels, days_into_future, min_age, max_age
    )

    return evaluate_network_factors(df_shifted, **evaluation_kws), df_shifted


def get_bee_subsample_network_age(
    keep_proportion,
    interactions,
    alive_matrices,
    bee_ids,
    bee_ages,
    alive_df,
    from_date,
    num_factors_per_mode,
    location_dataframe,
):
    num_days = interactions.shape[0]
    num_entities = int(interactions.shape[1] * keep_proportion)
    keep_entities = sorted(
        np.random.choice(range(interactions.shape[1]), size=num_entities, replace=False)
    )

    alive_matrices = alive_matrices[
        np.ix_(range(num_days), keep_entities, keep_entities)
    ]
    interaction_idxer = np.ix_(
        range(num_days), keep_entities, keep_entities, range(interactions.shape[-1])
    )
    interactions = interactions[interaction_idxer].copy()
    bee_ids = bee_ids[keep_entities].copy()
    bee_ages = bee_ages[:, keep_entities].copy()
    alive_df = alive_df[alive_df.bee_id.isin(bee_ids)].sort_values("bee_id")

    interactions = bb_network_decomposition.normalization.rank_transform(
        interactions, alive_matrices
    )
    daily_factors, _ = bb_network_decomposition.spectral.decomposition_by_day(
        interactions, alive_matrices, num_factors_per_mode, num_jobs=4
    )
    daily_factors_aligned = bb_network_decomposition.spectral.temporal_alignment(
        daily_factors, alive_matrices
    )
    factor_df = bb_network_decomposition.data.get_factor_dataframe(
        daily_factors_aligned, from_date, alive_df, bee_ids
    )

    cca_factor_df = bb_network_decomposition.projection.get_cca_projection(
        factor_df, location_dataframe
    )

    return cca_factor_df


def evaluate_bee_subsample(
    keep_proportion,
    interactions,
    alive_matrices,
    bee_ids,
    bee_ages,
    alive_df,
    from_date,
    num_factors_per_mode,
    location_dataframe,
    evaluation_kws={},
):
    cca_factor_df = get_bee_subsample_network_age(
        keep_proportion,
        interactions,
        alive_matrices,
        bee_ids,
        bee_ages,
        alive_df,
        from_date,
        num_factors_per_mode,
        location_dataframe,
    )

    return evaluate_network_factors(cca_factor_df, **evaluation_kws)


def likelihood_ratio_test(log_prob_null, log_prob_model, dof):
    G = -2 * (log_prob_null - log_prob_model)
    p_value = scipy.stats.chi2.sf(G, dof)

    return p_value


def rho_mcf(log_prob_model, log_prob_null):
    return 1 - log_prob_model / log_prob_null
