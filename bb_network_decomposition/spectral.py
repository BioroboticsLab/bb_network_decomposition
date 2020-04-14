import multiprocessing
import multiprocessing.pool
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sknetwork


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def decompose_day(interactions, alive_matrices, num_factors, spectral_kws):
    day_alive = alive_matrices.max(axis=-1)
    num_modes = interactions.shape[-1]
    num_entities = interactions.shape[1]

    embeddings = []
    num_factors_by_mode = []
    alive_graph = interactions[
        np.ix_(np.argwhere(day_alive)[:, 0], np.argwhere(day_alive)[:, 0])
    ]
    for mode in range(num_modes):
        alive_graph_mode = alive_graph[:, :, mode]
        if not check_symmetric(alive_graph_mode):
            bispectral = sknetwork.embedding.BiSpectral(
                embedding_dimension=num_factors, **spectral_kws
            )
            _ = bispectral.fit(alive_graph_mode)
            embeddings.append(bispectral.col_embedding_)
            embeddings.append(bispectral.row_embedding_)

            num_factors_by_mode.append(num_factors * 2)
        else:
            spectral = sknetwork.embedding.Spectral(
                embedding_dimension=num_factors, **spectral_kws
            )
            _ = spectral.fit(alive_graph_mode)
            embeddings.append(spectral.embedding_)

            num_factors_by_mode.append(num_factors)

    all_embeddings_day = np.zeros(
        (num_entities, sum(map(lambda e: e.shape[1], embeddings)))
    )
    all_embeddings_day[day_alive] = np.concatenate(embeddings, axis=-1)

    return all_embeddings_day, num_factors_by_mode


def decomposition_by_day(
    interactions,
    alive_matrices,
    num_factors,
    num_jobs=-1,
    spectral_kws={"regularization": 0.01, "scaling": "divide"},
):
    num_days = interactions.shape[0]
    assert interactions.shape[1] == interactions.shape[2]

    def decompose_wrapper(day):
        return decompose_day(
            interactions[day], alive_matrices[day], num_factors, spectral_kws
        )

    if num_jobs == -1:
        num_jobs = multiprocessing.cpu_count()

    with multiprocessing.pool.ThreadPool(num_jobs) as pool:
        results = pool.imap(decompose_wrapper, range(num_days))

    daily_factors = []
    for all_embeddings_day, num_factors_by_mode in results:
        daily_factors.append(all_embeddings_day)

    return daily_factors, num_factors_by_mode


def get_factor_labels(num_factors_by_mode, labels):
    label_names = []
    factor_idx = 0
    for num, label in zip(num_factors_by_mode, labels):
        for _ in range(num):
            label_names.append((factor_idx, label, f"{factor_idx:0>2d}_{label}"))
            factor_idx += 1

    return pd.DataFrame(
        label_names, columns=["factor_idx", "label", "sequential_label"]
    )


def temporal_alignment(daily_factors, alive_matrices, scaler=None):
    num_days = len(daily_factors)
    num_factors = daily_factors[0].shape[-1]

    unscaled_features = []
    for day in range(num_days):
        if day == 0:
            features_unscaled = daily_factors[day].copy()
            day_alive = alive_matrices[day].max(axis=-1)

            if scaler is not None:
                features_unscaled[day_alive] = scaler.fit_transform(
                    features_unscaled[day_alive]
                )

            unscaled_features.append(features_unscaled)
        else:
            day_alive = alive_matrices[day].max(axis=-1)
            both_days_alive = alive_matrices[day].max(axis=-1) & alive_matrices[
                day - 1
            ].max(axis=-1)
            features_unscaled_day = daily_factors[day].copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corrs = [
                    scipy.stats.spearmanr(
                        unscaled_features[-1][both_days_alive][:, mode],
                        daily_factors[day][both_days_alive][:, mode],
                    )
                    for mode in range(num_factors)
                ]
                corr_signs = np.array([np.sign(c.correlation) for c in corrs])[None, :]

            corr_signs[np.isnan(corr_signs)] = 1.0
            features_unscaled_day[day_alive] *= corr_signs

            if scaler is not None:
                features_unscaled_day[day_alive] = scaler.fit_transform(
                    features_unscaled_day[day_alive]
                )

            unscaled_features.append(features_unscaled_day[:, :])

    return unscaled_features
