import json
import random
from glob import glob
from os.path import basename

import networkx as nx
import torch.cuda
from collections import Counter

import tqdm


class ProgramDataset:
    def __init__(self, data_location, dataset, onehot, target_mode, batch_size, device):
        self.data_location = data_location
        self.dataset = dataset
        self.onehot = onehot
        self.target_mode = target_mode
        self.device = device
        self.batch_size = batch_size

        random.seed(1)
        self.load_data()

    def load_data(self):
        base_path = f'{self.data_location}/{self.dataset}'

        self.training_pairs = glob(f'{base_path}/train/*.json')
        self.testing_pairs = glob(f'{base_path}/test/*.json')

        dataset_attr = json.load(open(f'{base_path}/properties.json', 'r'))
        self.max_nodes, self.labels = dataset_attr["maxNodes"], dataset_attr["labels"]
        self.onehot_dim = len(self.labels)

    def preprocess(self):
        """
        返回graphs数据集中所有图的邻接矩阵和掩码masks
        :return:
        adjs: graphs_num × max_nodes × max_nodes 不包含自环，后面的GNN模块会自动加上
        masks: graphs_num × max_nodes {0,1}代表该图的实际节点数
        """
        self.adjs = []
        self.masks = []
        for i in range(len(self.graphs)):
            g = self.graphs[i]
            adj_padded, mask = self.get_adj_padded_and_mask(g)
            self.adjs.append(adj_padded)
            self.masks.append(mask)

    def get_adj_padded_and_mask(self, n, edges):
        # 创建一个 n x n 的零矩阵
        adj = torch.zeros((n, n), device=self.device)

        # 根据连接关系填充邻接矩阵
        for edge in edges:
            node1, node2 = edge
            adj[node1][node2] = 1
            adj[node2][node1] = 1

        adj_padded = torch.zeros((self.max_nodes, self.max_nodes), device=self.device)
        adj_padded[:adj.shape[0], :adj.shape[1]] = adj

        mask = torch.zeros(self.max_nodes, dtype=torch.int32, device=self.device)
        mask[:n] = 1
        return adj_padded, mask

    def get_batch_data(self, batch):
        batch_feat_1 = []
        batch_adj_1 = []
        batch_feat_2 = []
        batch_adj_2 = []
        batch_masks_1 = []
        batch_masks_2 = []

        batch_avg_v = []
        batch_hb = []
        batch_target_similarity = []
        batch_target_ged = []

        for graph_pair in batch:
            data = json.load(open(graph_pair))
            ged = data['ged']
            batch_target_ged.append(ged)

            # 获取图对的节点数和边数
            n1, m1 = (len(data['labels_1']), len(data['graph_1']))
            n2, m2 = (len(data['labels_2']), len(data['graph_2']))

            # 获取图对的邻接矩阵 和 节点掩码
            adj_1, mask_1 = self.get_adj_padded_and_mask(n1, data['graph_1'])
            adj_2, mask_2 = self.get_adj_padded_and_mask(n2, data['graph_2'])
            
            feature_1, feature_2 = self.global_encode({'labels': data['labels_1']}, {'labels': data['labels_2']})

            if self.target_mode == "exp":
                avg_v = (n1 + n2) / 2.0
                batch_avg_v.append(avg_v)
                similarity = torch.exp(torch.tensor([-ged / avg_v]).float()).to(self.device)
            elif self.target_mode == "linear":
                higher_bound = max(n1, n2) + max(m1, m2)
                batch_hb.append(higher_bound)
                similarity = torch.tensor([ged / higher_bound]).float().to(self.device)
            else:
                assert False

            batch_target_similarity.append(similarity)

            batch_feat_1.append(feature_1)
            batch_adj_1.append(adj_1)
            batch_masks_1.append(mask_1)

            batch_feat_2.append(feature_2)
            batch_adj_2.append(adj_2)
            batch_masks_2.append(mask_2)

        return (torch.stack(batch_feat_1), torch.stack(batch_adj_1), torch.stack(batch_masks_1),
                torch.stack(batch_feat_2), torch.stack(batch_adj_2), torch.stack(batch_masks_2),
                torch.Tensor(batch_avg_v).to(self.device), torch.Tensor(batch_hb).to(self.device),
                torch.stack(batch_target_similarity).squeeze(), torch.Tensor(batch_target_ged).to(self.device))

    def single_graph_encode(self, graph, labels_map):
        features = torch.zeros(self.max_nodes, self.onehot_dim, device=self.device)
        for i, label in enumerate(graph["labels"]):
            index = labels_map[label]
            features[i][index] = 1
        return features

    def global_encode(self, graph1, graph2):
        return self.single_graph_encode(graph1, self.labels), self.single_graph_encode(graph2, self.labels)

    def pair_encode(self, graph1, graph2):
        """
        将两个图的labels属性转成one-hot编码（需要保证graph1和graph2的labels类型总和不超过模型的处理范围）
        :param graph1:
        :param graph2:
        :return:
        """
        # 统计合并两个graph的标签
        counter = Counter(graph1["labels"] + graph2["labels"])
        sorted_labels = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        labels_map = {label: index for index, (label, _) in enumerate(sorted_labels)}
        return self.single_graph_encode(graph1, labels_map), self.single_graph_encode(graph2, labels_map)

