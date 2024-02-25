import pygmtools as pygm
import torch.nn as nn
from torch import Tensor


class Sinkhorn(nn.Module):
    r"""
    Sinkhorn算法将输入的Affinity矩阵转化为bi-stochastic矩阵.
    1. 首先对矩阵中的每个元素 exp(s[i,j]/tau)
    2. 然后迭代地执行行列归一化

    :param max_iter: 最大迭代次数，（默认是10）
    :param tau: 超参数 （默认1）
    :param epsilon: a small number for numerical stability 默认1e-4
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)
    """

    def __init__(self, max_iter: int = 10, tau: float = 1., epsilon: float = 1e-4,
                 log_forward: bool = True, batched_operation: bool = False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        # batched operation may cause instability in backward computation, but will boost computation.
        self.batched_operation = batched_operation

    def forward(self, s: Tensor, nrows: Tensor = None, ncols: Tensor = None, dummy_row: bool = False) -> Tensor:
        r"""
        :param s: b×n1×n2 input 3d tensor. b是batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: b×n1×n2 the computed doubly-stochastic matrix

        . note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        . note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        . note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, nrows, ncols, dummy_row)
        else:
            # deprecated
            return self.forward_ori(s, nrows, ncols, dummy_row)

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False):
        """Compute sinkhorn with row/column normalization in the log space."""
        return pygm.sinkhorn(s, n1=nrows, n2=ncols, dummy_row=dummy_row, max_iter=self.max_iter, tau=self.tau,
                             batched_operation=self.batched_operation, backend='pytorch')

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        """
        Computing sinkhorn with row/column normalization.

        . warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        print("warning:This function is deprecated because `~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` "
              "is more numerically stable.")
        return pygm.sinkhorn(s, n1=nrows, n2=ncols, dummy_row=dummy_row, max_iter=self.max_iter, tau=self.tau,
                             batched_operation=self.batched_operation, backend='pytorch')