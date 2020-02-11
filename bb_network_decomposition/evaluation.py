import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection


location_labels = [
    'dance_floor',
    'honey_storage',
    'brood_area',
    'near_exit',
]

default_factors = [
    'age',
    'network_age',
    'network_age_0+network_age_1'
]


def evaluate_network_factors(df, model, n_splits=25, groupby=None, labels=location_labels, factors=default_factors):
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

    if groupby is None:
        groups = None
        cv = sklearn.model_selection.ShuffleSplit(n_splits=n_splits)
    else:
        cv = sklearn.model_selection.GroupShuffleSplit(n_splits=n_splits)
        if groupby == 'days':
            mapper = dict(zip(df.day.unique(), np.arange(len(df.day.unique()))))
            groups = df.day.apply(lambda day: mapper[day])
        elif groupby == 'bees':
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
                model, X, Y,
                cv=cv, groups=groups,
                scoring=sklearn.metrics.make_scorer(sklearn.metrics.r2_score),
                n_jobs=-1
            )

            for score in scores:
                regression_results_df.append(dict(
                    variable=factor,
                    target=label,
                    r_squared=score))

    regression_results_df = pd.DataFrame(regression_results_df)

    regression_pivot = regression_results_df.pivot_table(values="r_squared", index="variable", columns="target")
    regression_pivot["mean"] = np.mean(regression_pivot.values, axis=1)
    regression_pivot = regression_pivot.sort_values("mean")

    return regression_results_df, regression_pivot
