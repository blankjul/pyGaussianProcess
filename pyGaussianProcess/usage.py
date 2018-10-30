import random

import matplotlib.pyplot as plt
import numpy as np

from pyGaussianProcess.model import fit, predict

if __name__ == '__main__':

    # number of samples we will use for this example
    n_samples = 120

    # ---------------------------------------------------------
    # Example 1: One input variable and one target
    # ---------------------------------------------------------

    random.seed(1)
    np.random.seed(1)

    X = np.random.rand(n_samples, 1) * 4 * np.pi
    Y = np.cos(X)[:,0]

    model = fit(X, Y, optimize=True)

    _X = np.linspace(0, 4 * np.pi, 1000)[:,None]
    _Y,_ = predict(model, _X)

    plt.scatter(X, Y, label="Observations")
    plt.plot(_X, _Y, label="True")
    plt.show()
