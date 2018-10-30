import math

import torch

from pyGaussianProcess.predictor import CholeskyDecompositionPrediction
from pyGaussianProcess.util import diff


def loss_neg_loss_likelihood(y, K):
    return torch.mean(-0.5 * torch.log(var_hat) - (y - y_hat).pow(2) / 2 * var_hat - 0.5 * math.log(2 * math.pi))


def loss_fn(y, y_hat, var_hat):
    return torch.mean(-0.5 * torch.log(var_hat) - (y - y_hat).pow(2) / 2 * var_hat - 0.5 * math.log(2 * math.pi))


def loss_fn2(y, y_hat, var_hat):
    return ((y - y_hat).pow(2)).mean()



def neg_loss_likelihood(y, K_dd, K_xd, K_xx):
    predictor = CholeskyDecompositionPrediction().fit(y, K_dd)
    y L

    return torch.mean(-0.5 * torch.log(var_hat) - (y - y_hat).pow(2) / 2 * var_hat - 0.5 * math.log(2 * math.pi))


def crossvalidation(y, K_dd, K_xd, K_xx):

    _mean = torch.zeros(y.shape[0])
    _var = torch.zeros(y.shape[0])

    for i in range(y.shape[0]):

        I = torch.ones(y.shape[0]).byte()
        I[i] = 0

        # preprocess data to simulate cross validation
        _K_xd = K_xd[[i], :][:, I]
        _K_xx = K_xx[i, i]
        _K_dd = K_dd[I, :][:, I]
        _y = y[I]

        _mean[i], _var[i] = CholeskyDecompositionPrediction().fit(_y, _K_dd).predict(_K_xd, _K_xx)

    return _mean, _var


def optimize(X, y, kernel, n_max_epochs=1000, disp=True):

    theta = torch.nn.Parameter(torch.tensor(1 * torch.ones(X.shape[1]), requires_grad=True))
    optimizer = torch.optim.SGD([theta], lr=1e-1, momentum=0.9)

    loss = 1e20

    for t in range(n_max_epochs):

        K_xx = kernel.calc(diff(X, X), theta)
        #y_hat, var_hat = crossvalidation(y, K_xx, K_xx, K_xx)
        #_loss = loss_fn2(y, y_hat, var_hat)

        neg_loss_likelihood
        _loss = loss_neg_loss_likelihood(y,K_xx)

        optimizer.zero_grad()
        _loss.backward(retain_graph=True)
        optimizer.step()

        if t >= n_max_epochs or loss - _loss < 0.00000001:
            return theta

        loss = _loss

        if disp:
            print(t, loss.item(), theta.data.numpy())

    return theta
