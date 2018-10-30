import torch


def to_tensor(a):
    return torch.from_numpy(a).float()


def diff(A, B):
    return A.unsqueeze(1) - B.unsqueeze(0)
