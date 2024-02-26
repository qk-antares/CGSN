import torch


class DenseAttentionModule(torch.nn.Module):
    """
    得到图嵌入的DenseAttentionModule
    """
    def __init__(self, dim):
        super(DenseAttentionModule, self).__init__()
        self.dim = dim

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: GNN模块的输出
        :param mask: 表示每个图valid节点的Mask matrix
        :return representation: 图级别的表示矩阵
        """
        B, N, _ = x.size()

        if mask is not None:
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            mean = x.sum(dim=1) / num_nodes.to(x.dtype)
        else:
            mean = x.mean(dim=1)

        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        koefs = torch.sigmoid(torch.matmul(x, transformed_global.unsqueeze(-1)))
        weighted = koefs * x

        if mask is not None:
            weighted = weighted * mask.view(B, N, 1).to(x.dtype)

        return weighted.sum(dim=1)
