import functools

import numpy as np
import torch

import bb_network_decomposition.constants


def get_logits(X, Y, intercepts, coeffs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_ = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_ = torch.from_numpy(Y.astype(np.float32)).to(device)

    for i in range(len(coeffs)):
        X_ = torch.mm(X_, coeffs[i]) + intercepts[i]
        if i < len(coeffs) - 1:
            X_ = torch.tanh(X_)

    # null model
    if len(coeffs) == 0:
        # workaround for RuntimeError: unsupported operation: more than one element of
        # the written-to tensor refers to a single memory location. Please clone() the
        # tensor before performing the operation.
        X_ = intercepts[-1] + Y_ * 0

    return X_


def evaluate_binomial(X, Y, total_counts, intercepts, coeffs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    C_ = torch.from_numpy(total_counts.astype(np.int)).to(device)
    Y_ = torch.from_numpy(Y.astype(np.float32)[:, 0]).to(device)

    logits = get_logits(X, Y, intercepts, coeffs)[:, 0]

    probs = torch.sigmoid(logits)
    binomial = torch.distributions.binomial.Binomial(logits=logits, total_count=C_)
    log_probs = binomial.log_prob(Y_)

    return log_probs.cpu(), probs.detach().cpu().numpy()


def evaluate_multinomial(X, Y, total_counts, intercepts, coeffs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Y_ = torch.from_numpy(Y.astype(np.float32)).to(device)

    logits = get_logits(X, Y, intercepts, coeffs)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    multinomial = torch.distributions.multinomial.Multinomial(logits=logits)
    log_probs = multinomial.log_prob(Y_)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_weight(shape):
        return torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.randn(shape).to(device))
        )

    def _get_intercept(shape):
        return torch.nn.Parameter(torch.zeros(shape).to(device))

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
):
    X = loc_df[predictors].values.astype(np.float)
    X /= X.std(axis=0)[None, :]

    probs = loc_df[labels].values

    total_counts_used = loc_df["location_descriptor_count"].values
    counts = total_counts_used[:, None] * probs

    log_likelhood_linear, _ = get_fitted_model(
        X, counts, total_counts_used, evaluation_fn, null=False
    )(X, counts, total_counts_used)
    log_likelhood_nonlinear, _ = get_fitted_model(
        X, counts, total_counts_used, evaluation_fn, null=False, nonlinear=True,
    )(X, counts, total_counts_used)
    log_likelhood_null, _ = get_fitted_model(
        X, counts, total_counts_used, evaluation_fn, null=True
    )(X, counts, total_counts_used)

    return dict(
        fitted_linear=log_likelhood_linear.sum().item(),
        fitted_nonlinear=log_likelhood_nonlinear.sum().item(),
        null=log_likelhood_null.sum().item(),
        fitted_linear_mean=log_likelhood_linear.mean().item(),
        fitted_nonlinear_mean=log_likelhood_nonlinear.mean().item(),
        null_mean=log_likelhood_null.mean().item(),
    )
