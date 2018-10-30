import numpy as np
import torch

from pyGaussianProcess import optimizer
from pyGaussianProcess.kernel import GaussianKernel
from pyGaussianProcess.predictor import CholeskyDecompositionPrediction
from pyGaussianProcess.util import to_tensor, diff


def fit(X, y, kernel=GaussianKernel(), optimize=True, theta=None, disp=False):
    if type(X) is np.ndarray:
        X = to_tensor(X)

    if type(y) is np.ndarray:
        y = to_tensor(y)

    if len(y.shape) == 2:
        y = y[:, 0]

    # save the mean and standard deviation of the input
    mX, sX = X.mean(0), X.std(0)
    mY, sY = y.mean(0), y.std(0)

    # standardize the input
    nX = (X - mX) / sX
    nY = (y - mY) / sY

    predictor = CholeskyDecompositionPrediction()

    if optimize:
        theta = optimizer.optimize(X, y, kernel, disp=disp)
    elif theta is None:
        theta = torch.tensor(1 * torch.ones(X.shape[1]), requires_grad=True)

    K_dd = kernel.calc(diff(nX, nX), theta)
    predictor.fit(nY, K_dd)

    model = {'mX': mX, 'sX': sX, 'nX': nX, 'mY': mY, 'sY': sY, 'nY': nY, 'kernel': kernel,
             'predictor': predictor, 'theta': theta}

    return model


def predict(model, _X):
    if type(_X) is np.ndarray:
        _X = to_tensor(_X)

    _nX = (_X - model['mX']) / model['sX']

    kernel, theta, predictor = model['kernel'], model['theta'], model['predictor']

    K_xx = kernel.calc(diff(_nX, _nX), theta)
    K_xd = kernel.calc(diff(_nX, model['nX']), theta)

    nY, nVar = predictor.predict(K_xd, K_xx)

    y = (nY * model['sY']) + model['mY']
    var = nVar * model['sY']

    return y.data.numpy(), var.data.numpy()
