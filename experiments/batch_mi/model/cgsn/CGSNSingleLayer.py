import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.batch_mi.model.common.DenseTensorNetwork import DenseTensorNetwork
from experiments.batch_mi.model.cgsn.Affinity import Affinity
from experiments.batch_mi.model.cgsn.DenseAttentionModule import DenseAttentionModule
from experiments.batch_mi.model.cgsn.GConv import GConv
from experiments.batch_mi.model.cgsn.Sinkhorn import Sinkhorn

class CGSNSingleLayer(nn.Module):
    def __init__(self, args, onehot_dim):
        super(CGSNSingleLayer, self).__init__()
        self.args = args
        self.onehot_dim = onehot_dim
        self.setup_layers()

    def setup_layers(self):
        # 节点嵌入模块的GNN
        filters = self.args.gnn_filters
        self.g_conv = GConv(in_dimension=self.onehot_dim, out_dimension=filters)
        # affinity模块
        self.affinity = Affinity(d=self.onehot_dim)
        # sinkhorn模块
        self.sinkhorn = Sinkhorn(max_iter=self.args.max_iter, tau=self.args.tau, epsilon=self.args.epsilon)
        # cross_graph模块
        self.cross_graph = nn.Linear(filters+self.onehot_dim, filters)

        # 得到图嵌入的att模块
        self.attn_pool_1 = DenseAttentionModule(filters)
        self.attn_pool_2 = DenseAttentionModule(filters)

        # 图图交互的ntn模块
        self.tensor_network = DenseTensorNetwork(filters, self.args.tensor_neurons * 2)

        # 最后预测相似度的reg模块
        self.final_reg = nn.Sequential(
            nn.Linear(self.args.tensor_neurons, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward_gnn(self, feat, adj, i):
        feat_out = getattr(self, 'gnn{}'.format(i))(x=feat, adj=adj)
        feat_out = F.relu(feat_out, inplace=True)
        feat_out = F.dropout(feat_out, p=self.args.dropout, training=self.training)
        return feat_out

    def forward_att_feat_agg_layers(self, emb1, adj1, mask1, emb2, adj2, mask2):
        # 图内更新
        inter_emb1, inter_emb2 = self.g_conv(emb1, adj1), self.g_conv(emb2, adj2)

        # 跨图信息（实际上是每个节点，求另一个图所有节点特征的加权和）
        s = self.affinity(emb1, emb2)
        s = self.sinkhorn(s, torch.sum(mask1, dim=1), torch.sum(mask2, dim=1), dummy_row=True)
        crs_emb1 = torch.bmm(s, emb2)
        crs_emb2 = torch.bmm(s.transpose(1, 2), emb1)

        emb1 = self.cross_graph(torch.cat((inter_emb1, crs_emb1), dim=-1))
        emb2 = self.cross_graph(torch.cat((inter_emb2, crs_emb2), dim=-1))
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
        graph_emb2 = self.attn_pool_2(emb1, batch_mask_2)

        scores = self.tensor_network(graph_emb1, graph_emb2)

        mean, log_var = scores.split(self.args.tensor_neurons, dim=1)
        # KL散度
        kl_loss = - 0.5 * torch.mean(1 + log_var - torch.square(mean) - torch.exp(log_var), dim=1)

        # 重参数化
        if self.training:
            u = torch.randn_like(log_var)
        else:
            u = torch.zeros_like(log_var)
        scores = mean + torch.exp(log_var / 2) * u

        score = self.final_reg(scores).view(-1)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * batch_avg_v
        elif self.args.target_mode == "linear":
            pre_ged = score * batch_hb
        else:
            assert False
        return score, pre_ged, kl_loss
