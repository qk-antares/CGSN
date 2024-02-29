import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GConv(nn.Module):
    """
    :param in_dimension: the dimension of input node features
    :param out_dimension: the dimension of output node features
    """
    def __init__(self, in_dimension: int, out_dimension: int):
        super(GConv, self).__init__()
        self.num_inputs = in_dimension
        self.num_outputs = out_dimension
        # 邻居更新
        self.a_fc = torch.nn.Linear(self.num_inputs, self.num_outputs)
        # 自更新
        self.u_fc = torch.nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, x: Tensor, adj: Tensor, norm: bool = True) -> Tensor:
        """
        Forward computation of graph convolution network.
        :param adj: b×n×n {0,1} 邻接矩阵. b是batch大小, n是节点数
        :param x: b×n×d 输入的节点嵌入. d是嵌入维度
        :param norm: 是否对邻接矩阵归一化
        :return: b×n×d 新的节点嵌入
        """
        if norm is True:
            adj = F.normalize(adj, p=1, dim=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(adj, F.relu(ax)) + F.relu(ux)
        return x
