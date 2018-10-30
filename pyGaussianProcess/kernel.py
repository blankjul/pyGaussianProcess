class Kernel:
    pass


class GaussianKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def calc(self, D, theta):
        return (-0.5 * (D * theta).pow(2).sum(2)).exp()
