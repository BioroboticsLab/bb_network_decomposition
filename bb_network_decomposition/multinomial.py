import functools

import numpy as np
import torch

import bb_network_decomposition.constants


def get_fitted_model(X, Y, null=False, num_steps=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    intercepts = torch.nn.Parameter(torch.zeros((Y.shape[-1],)).to(device))
    coeffs = torch.zeros((X.shape[-1], Y.shape[-1])).to(device)
    params = [intercepts]

    # for null model, simply don't update coeffs -> X will never affect
    # regression results
    if not null:
        coeffs = torch.nn.Parameter(coeffs)
        params.append(coeffs)

    optimizer = torch.optim.LBFGS(params, lr=0.1)

    def evaluate(X, Y, intercepts, coeffs):
        X_ = torch.from_numpy(X.astype(np.float32)).to(device)
        Y_ = torch.from_numpy(Y.astype(np.float32)).to(device)

        logits = torch.mm(X_, coeffs) + intercepts

        probs = torch.nn.functional.softmax(logits, dim=0)
        multinomial = torch.distributions.multinomial.Multinomial(logits=logits)
        log_probs = multinomial.log_prob(Y_)

        return log_probs.cpu(), probs.detach().cpu().numpy()

    def closure():
        optimizer.zero_grad()

        log_probs, _ = evaluate(X, Y, intercepts, coeffs)
        nll = -log_probs.sum()

        nll.backward()
        return nll

    for _ in range(num_steps):
        optimizer.step(closure)

    evaluate = functools.partial(evaluate, coeffs=coeffs, intercepts=intercepts)

    return evaluate


def get_location_multinomial_likelihoods(loc_df, predictors):
    labels = bb_network_decomposition.constants.location_labels

    X = loc_df[predictors].values
    probs = loc_df[labels].values

    total_counts_used = loc_df["total_count"] - loc_df["other"] - loc_df["not_comb"]
    counts = total_counts_used.values[:, None] * probs

    log_likelhood, _ = get_fitted_model(X, counts, null=False)(X, counts)
    log_likelhood_null, _ = get_fitted_model(X, counts, null=True)(X, counts)

    return dict(fitted=log_likelhood.sum().item(), null=log_likelhood_null.sum().item())
