import numpy as np
import scipy
import scipy.stats
import tqdm


def rank_transform(interactions, alive_matrices, rank_method='dense'):
    for mode in tqdm.trange(interactions.shape[-1]):
        for day in range(interactions.shape[0]):
            day_alive = alive_matrices[day].max(axis=-1)
            ix = np.ix_([day], np.argwhere(day_alive)[:, 0], np.argwhere(day_alive)[:, 0], [mode])
            alive_graph = interactions[ix]

            ranks = scipy.stats.rankdata(alive_graph.flatten(), method=rank_method)
            ranks = ranks.astype(np.float32) / ranks.max()

            interactions[ix] = ranks.reshape(alive_graph.shape)

    return interactions