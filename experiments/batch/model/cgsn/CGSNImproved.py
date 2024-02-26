import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.batch.model.common.DenseTensorNetwork import DenseTensorNetwork
from experiments.batch.model.cgsn.Affinity import Affinity
from experiments.batch.model.cgsn.DenseAttentionModule import DenseAttentionModule
from experiments.batch.model.cgsn.GConv import GConv
from experiments.batch.model.cgsn.Sinkhorn import Sinkhorn


class CGSNImproved(nn.Module):
    def __init__(self, args, onehot_dim):
        super(CGSNImproved, self).__init__()
        self.args = args
        self.onehot_dim = onehot_dim
        self.setup_layers()

    def setup_layers(self):
        # 节点嵌入模块的GNN
        filters = self.args.gnn_filters.split('-')
        gnn_filters = [int(n_filter) for n_filter in filters]
        self.gnn_numbers = len(gnn_filters)

        gnn_settings = [dict(
            in_dimension=gnn_filters[i - 1], out_dimension=gnn_filters[i])
            for i in range(1, self.gnn_numbers)]
        gnn_settings.insert(0, dict(
            in_dimension=self.onehot_dim, out_dimension=gnn_filters[0]))

        setattr(self, 'gnn{}'.format(1), GConv(**gnn_settings[0]))
        for i in range(1, self.gnn_numbers):
            setattr(self, 'gnn{}'.format(i + 1), GConv(**gnn_settings[i]))

        # 与gnn对应的affinity模块
        affinity_settings = [dict(d=gnn_filters[i]) for i in range(0, self.gnn_numbers)]
        for i in range(0, self.gnn_numbers):
            setattr(self, 'affinity{}'.format(i + 1), Affinity(**affinity_settings[i]))

        # sinkhorn模块
        self.sinkhorn = Sinkhorn(max_iter=self.args.max_iter, tau=self.args.tau, epsilon=self.args.epsilon)

        # cross_graph模块
        for i in range(0, self.gnn_numbers):
            setattr(self, 'cross_graph{}'.format(i + 1), nn.Linear(gnn_filters[i] * 2, gnn_filters[i]))

        # 得到图嵌入的att模块
        self.attn_pool_1 = DenseAttentionModule(gnn_filters[-1])
        self.attn_pool_2 = DenseAttentionModule(gnn_filters[-1])

        # 图图交互的ntn模块
        self.tensor_network = DenseTensorNetwork(gnn_filters[-1], self.args.tensor_neurons)

        # 最后预测相似度的reg模块
        self.final_reg = nn.Sequential(
            nn.Linear(self.args.tensor_neurons, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward_att_feat_agg_layers(self, emb1, adj1, mask1, emb2, adj2, mask2):
        for i in range(1, self.gnn_numbers + 1):
            gnn_layer = getattr(self, 'gnn{}'.format(i))
            emb1, emb2 = gnn_layer(emb1, adj1), gnn_layer(emb2, adj2)

            affinity = getattr(self, 'affinity{}'.format(i))
            s = affinity(emb1, emb2)
            s = self.sinkhorn(s, torch.sum(mask1, dim=1), torch.sum(mask2, dim=1), dummy_row=True)

            cross_graph = getattr(self, 'cross_graph{}'.format(i))
            new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
            new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
            emb1 = new_emb1
            emb2 = new_emb2
        return emb1, emb2

    def forward(self, batch_feat_1, batch_feat_2, batch_adj_1, batch_adj_2,
                batch_mask_1, batch_mask_2, batch_avg_v, batch_hb):
        """
        Forward pass with graphs.
        :param batch_feat_1:
        :param batch_feat_2:
        :param batch_adj_1:
        :param batch_adj_2:
        :param batch_mask_1:
        :param batch_mask_2:
        :param batch_avg_v:
        :param batch_hb:
        :return:
        """
        emb1, emb2 = self.forward_att_feat_agg_layers(batch_feat_1, batch_adj_1, batch_mask_1,
                                                      batch_feat_2, batch_adj_2, batch_mask_2)

        graph_emb1 = self.attn_pool_1(emb1, batch_mask_1)
        graph_emb2 = self.attn_pool_2(emb2, batch_mask_2)

        scores = self.tensor_network(graph_emb1, graph_emb2)

        score = self.final_reg(scores).view(-1)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * batch_avg_v
        elif self.args.target_mode == "linear":
            pre_ged = score * batch_hb
        else:
            assert False
        return score, pre_ged
