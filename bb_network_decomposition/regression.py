import functools

import numpy as np
import torch

import bb_network_decomposition.constants


def get_logits(X, Y, intercepts, coeffs):
    for i in range(len(coeffs)):
        X = torch.mm(X, coeffs[i]) + intercepts[i]
        if i < len(coeffs) - 1:
            X = torch.tanh(X)

    # null model
    if len(coeffs) == 0:
        # workaround for RuntimeError: unsupported operation: more than one element of
        # the written-to tensor refers to a single memory location. Please clone() the
        # tensor before performing the operation.
        X = intercepts[-1] + Y * 0

    return X


def evaluate_binomial(X, Y, total_counts, intercepts, coeffs):
    logits = get_logits(X, Y, intercepts, coeffs)[:, 0]

    probs = torch.sigmoid(logits)
    binomial = torch.distributions.binomial.Binomial(
        logits=logits, total_count=total_counts
    )
    log_probs = binomial.log_prob(Y[:, 0])

    return log_probs.cpu(), probs.detach().cpu().numpy()


def evaluate_multinomial(X, Y, total_counts, intercepts, coeffs):
    logits = get_logits(X, Y, intercepts, coeffs)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    multinomial = torch.distributions.multinomial.Multinomial(logits=logits)
    log_probs = multinomial.log_prob(Y)

    return log_probs.cpu(), probs.detach().cpu().numpy()


def get_fitted_model(
    X,
    Y,
    total_counts,
    evaluation_fn,
    null=False,
    nonlinear=False,
    num_steps=10,
    hidden_size=8,
):
    def _get_weight(shape):
        return torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.randn(shape).to(device))
        )

    def _get_intercept(shape):
        return torch.nn.Parameter(torch.zeros(shape).to(device))

    device = X.device

    intercepts = []
    coeffs = []
    if not null:
        if nonlinear:
            intercepts += [_get_intercept((hidden_size,))]
            coeffs += [
                _get_weight((X.shape[-1], hidden_size)),
                _get_weight((hidden_size, Y.shape[-1])),
            ]
        else:
            coeffs += [_get_weight((X.shape[-1], Y.shape[-1]))]

    intercepts.append(_get_intercept((Y.shape[-1],)))

    params = coeffs + intercepts
    optimizer = torch.optim.LBFGS(params, lr=0.1)

    def closure():
        optimizer.zero_grad()

        log_probs, _ = evaluation_fn(X, Y, total_counts, intercepts, coeffs)
        nll = -log_probs.sum()

        nll.backward()
        return nll

    for _ in range(num_steps):
        optimizer.step(closure)

    evaluate = functools.partial(evaluation_fn, coeffs=coeffs, intercepts=intercepts)

    return evaluate


def get_location_likelihoods(
    loc_df,
    predictors,
    labels=bb_network_decomposition.constants.location_labels,
    evaluation_fn=evaluate_multinomial,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    X = loc_df[predictors].values.astype(np.float)
    X /= X.std(axis=0)[None, :]

    probs = loc_df[labels].values

    total_counts_used = loc_df["location_descriptor_count"].values
    counts = total_counts_used[:, None] * probs

    X = torch.from_numpy(X.astype(np.float32)).to(device)
    counts = torch.from_numpy(counts.astype(np.float32)).to(device)
    total_counts_used = torch.from_numpy(total_counts_used.astype(np.int)).to(device)

    results = dict()

    for nonlinear in (False, True):
        name = "nonlinear" if nonlinear else "linear"

        log_likelihood, _ = get_fitted_model(
            X, counts, total_counts_used, evaluation_fn, null=False, nonlinear=nonlinear
        )(X, counts, total_counts_used)

        results[f"fitted_{name}"] = log_likelihood.sum().item()
        results[f"fitted_{name}_mean"] = log_likelihood.mean().item()

    log_likelihood, _ = get_fitted_model(
        X, counts, total_counts_used, evaluation_fn, null=True, nonlinear=False
    )(X, counts, total_counts_used)

    results["null"] = log_likelihood.sum().item()
    results["null_mean"] = log_likelihood.mean().item()

    return results
