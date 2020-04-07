import scipy.stats


def likelihood_ratio_test(log_prob_null, log_prob_model, dof):
    G = -2 * (log_prob_null - log_prob_model)
    p_value = scipy.stats.chi2.sf(G, dof)

    return p_value


def rho_mcf(log_prob_model, log_prob_null):
    return 1 - log_prob_model / log_prob_null
