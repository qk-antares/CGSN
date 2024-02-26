import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseTensorNetwork(nn.Module):
    def __init__(self, input_dim, tensor_neurons):
        """
        支持batch的张量网络模块
        :param input_dim:
        :param tensor_neurons:
        """
        super(DenseTensorNetwork, self).__init__()
        self.input_dim = input_dim
        self.tensor_neurons = tensor_neurons

        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim, self.input_dim, self.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2 * self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, graph_emb1, graph_emb2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param graph_emb1: 图级嵌入，batch_size × input_dim
        :param graph_emb2: 图级嵌入，batch_size × input_dim
        :return scores: A similarity score vector.
        """
        batch_size = len(graph_emb1)

        scoring = torch.matmul(graph_emb1, self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(batch_size, self.input_dim, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, graph_emb2.unsqueeze(-1)).view(batch_size, -1)
        combined_representation = torch.cat((graph_emb1, graph_emb2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        # todo: 移除了此处的激活函数，因为现在NTN模块输出的是mean和log_var，不应该对其正负和范围做限制
        scores = scoring + block_scoring + self.bias.view(-1)
        return scores
