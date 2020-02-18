import numpy as np
import scipy
import scipy.stats


def alive_graph_transform(interactions, alive_matrices, transform):
    for mode in range(interactions.shape[-1]):
        for day in range(interactions.shape[0]):
            day_alive = alive_matrices[day].max(axis=-1)
            ix = np.ix_([day], np.argwhere(day_alive)[:, 0], np.argwhere(day_alive)[:, 0], [mode])
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
