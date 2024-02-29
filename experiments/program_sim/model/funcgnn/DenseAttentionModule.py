
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

    def forward(self, embs, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embs: GNN模块的输出，是一个 batch_size × max_nodes × filter_dim的Tensor
        :param mask: 表示每个图valid节点的Mask matrix
        :return representation: 图级别的表示矩阵
        """
        B, N, _ = embs.size()

        if mask is not None:
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            global_context = torch.sum(torch.matmul(embs, self.weight_matrix.unsqueeze(0)), dim=1) / num_nodes
        else:
            global_context = embs.mean(dim=1)

        transformed_global = torch.tanh(global_context)

        nodes_weight = torch.sigmoid(torch.matmul(embs, transformed_global.unsqueeze(-1)))
        graph_emb = torch.matmul(torch.transpose(embs, 1, 2), nodes_weight)
        return graph_emb.squeeze(-1)
