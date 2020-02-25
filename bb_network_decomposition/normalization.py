import numpy as np
import scipy
import scipy.stats
import sklearn
import sklearn.neighbors


def get_alive_idxer(alive_matrices, day, mode):
    day_alive = alive_matrices[day].max(axis=-1)
    ix = np.ix_([day], np.argwhere(day_alive)[:, 0], np.argwhere(day_alive)[:, 0], [mode])

    return ix


def alive_graph_transform(interactions, alive_matrices, transform):
    for mode in range(interactions.shape[-1]):
        for day in range(interactions.shape[0]):
            ix = get_alive_idxer(alive_matrices, day, mode)
            alive_graph = interactions[ix]

            interactions[ix] = transform(alive_graph)

    return interactions


def rank_transform(interactions, alive_matrices, rank_method='dense'):
    def _transform(alive_graph):
        ranks = scipy.stats.rankdata(alive_graph.flatten(), method=rank_method)
        ranks = ranks.astype(np.float32) / ranks.max()

        return ranks.reshape(alive_graph.shape)

    return alive_graph_transform(interactions, alive_matrices, _transform)


def std_transform(interactions, alive_matrices):
    def _transform(alive_graph):
        flat_graph = alive_graph.flatten()
        flat_graph = flat_graph.astype(np.float32) / flat_graph.std()

        return flat_graph.reshape(alive_graph.shape)

    return alive_graph_transform(interactions, alive_matrices, _transform)


def knn_transform(interactions, alive_matrices, mode_subset=None):
    if mode_subset is None:
        mode_subset = range(interactions.shape[-1])

    for mode in mode_subset:
        for day in range(interactions.shape[0]):
            day_alive = alive_matrices[day].max(axis=-1)
            ix = np.ix_([day], np.argwhere(day_alive)[:, 0], np.argwhere(day_alive)[:, 0], [mode])

            is_greater_zero = (interactions[day, :, :, mode] > 0).astype(np.float)

            connectivity = sklearn.neighbors.KNeighborsTransformer(
                mode='connectivity',
                n_neighbors=int(np.ceil(np.log(interactions.shape[1]))),
                metric='precomputed'
            ).fit_transform(-1 * (interactions[day, :, :, mode]) + (interactions[day, :, :, mode].max()))
            affinity_matrix = (0.5 * (connectivity + connectivity.T)).toarray()
            affinity_matrix *= is_greater_zero

            alive_affinity_matrix = np.zeros_like(affinity_matrix)
            alive_affinity_matrix[ix[1:-1]] = affinity_matrix[ix[1:-1]]

            interactions[day, :, :, mode] = alive_affinity_matrix

    return interactions

