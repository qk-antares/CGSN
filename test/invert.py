import torch
from torch import Tensor


def compute_a_normalize(adj):
    deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
    adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
    return adj


result = compute_a_normalize(Tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
                                     [0., 1., 0., 1., 0., 0., 1., 0., 0., 0.],
                                     [0., 0., 1., 0., 1., 0., 1., 1., 0., 0.],
                                     [0., 0., 1., 0., 0., 1., 0., 0., 1., 0.],
                                     [0., 0., 0., 1., 1., 0., 1., 0., 0., 0.],
                                     [0., 0., 0., 0., 1., 0., 0., 1., 1., 0.],
                                     [0., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
