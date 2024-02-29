import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv

from experiments.batch.model.common.DenseTensorNetwork import DenseTensorNetwork
from experiments.batch.model.simgnn.DenseAttentionModule import DenseAttentionModule


class funcGNN(nn.Module):
    def __init__(self, args, onehot_dim):
        """
        :param args: Arguments object.
        :param onehot_dim: 初始one-hot编码的维度
        """
        super(funcGNN, self).__init__()
        self.args = args
        self.onehot_dim = onehot_dim
        self.setup_layers()

    def setup_layers(self):
        self.calculate_bottleneck_features()

        # 节点嵌入模块GNN
        filters = self.args.gnn_filters.split('-')
        gnn_filters = [int(n_filter) for n_filter in filters]
        gnn_numbers = len(gnn_filters)
        self.gcn_numbers = gnn_numbers

        sage_parameters = [dict(
            in_channels=gnn_filters[i - 1], out_channels=gnn_filters[i])
            for i in range(1, gnn_numbers)]
        sage_parameters.insert(0, dict(
            in_channels=self.onehot_dim, out_channels=gnn_filters[0]))

        setattr(self, 'gnn{}'.format(1), DenseSAGEConv(**sage_parameters[0]))
        for i in range(1, gnn_numbers):
            setattr(self, 'gnn{}'.format(i + 1), DenseSAGEConv(**sage_parameters[i]))

        self.attention = DenseAttentionModule(gnn_filters[-1])

        self.tensor_network = DenseTensorNetwork(gnn_filters[-1], self.args.tensor_neurons)

        # 预测SimScore的MLP
        layers = []
        reg_neurons = self.args.reg_neurons
        layers.extend([
            nn.Linear(self.feature_count, reg_neurons),
            nn.ReLU(),
            nn.Linear(reg_neurons, 1),
            nn.Sigmoid()
        ])
        self.score_reg = nn.Sequential(*layers)

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def calculate_histogram(self, embs1, embs2):
        """
        Calculate histogram from similarity matrix.
        :param embs1: Feature matrix for graph 1.
        :param embs2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        batch_size = embs1.size()[0]
        scores = torch.matmul(embs1, embs2).detach()
        scores = scores.view(batch_size, -1)
        hist = torch.stack([torch.histc(t, bins=self.args.bins) for t in scores])
        hist = hist / torch.sum(hist, dim=1).unsqueeze(-1)
        return hist

    def forward_dense_sage_layers(self, feat, adj, mask):
        feat_in = feat
        for i in range(1, self.gcn_numbers):
            feat_out = getattr(self, 'gnn{}'.format(i))(x=feat_in, adj=adj, mask=mask)
            feat_out = F.relu(feat_out, inplace=True)
            feat_out = F.dropout(feat_out, p=self.args.dropout, training=self.training)
            feat_in = feat_out
        # 最后一层GraphSAGE出来不用再经过relu和dropout
        feat_out = getattr(self, 'gnn{}'.format(self.gcn_numbers))(x=feat_in, adj=adj, mask=None)
        return feat_out

    def forward(self, batch_feat_1, batch_feat_2, batch_adj_1, batch_adj_2,
                batch_mask_1, batch_mask_2, batch_avg_v, batch_hb):
        """
        Forward pass with graphs.
        :param batch_feat_1: b × max_nodes × onehot_dim
        :param batch_feat_2: b × max_nodes × onehot_dim
        :param batch_adj_1: b × max_nodes × max_nodes
        :param batch_adj_2: b × max_nodes × max_nodes
        :param batch_mask_1: b × max_nodes
        :param batch_mask_2: b × max_nodes
        :param batch_avg_v: b × 1
        :param batch_hb: b × 1
        :return: 相似度分数(b×1),GED(b×1)
        """
        embs1 = self.forward_dense_sage_layers(feat=batch_feat_1, adj=batch_adj_1, mask=batch_mask_1)
        embs2 = self.forward_dense_sage_layers(feat=batch_feat_2, adj=batch_adj_2, mask=batch_mask_2)

        graph_emb1 = self.attention(embs1, mask=batch_mask_1)
        graph_emb2 = self.attention(embs2, mask=batch_mask_2)

        scores = self.tensor_network(graph_emb1, graph_emb2)

        if self.args.histogram:
            hist = self.calculate_histogram(embs1, torch.transpose(embs2, 1, 2))
            scores = torch.cat((scores, hist), dim=1)

        score = self.score_reg(scores).view(-1)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * batch_avg_v
        elif self.args.target_mode == "linear":
            pre_ged = score * batch_hb
        else:
            assert False
        return score, pre_ged
