import json
import random
from glob import glob
from os.path import basename

import networkx as nx
import torch.cuda
from collections import Counter


class Dataset:
    def __init__(self, data_location, dataset, onehot, num_testing_graphs, target_mode, batch_size, device):
        self.data_location = data_location
        self.dataset = dataset
        self.onehot = onehot
        self.num_testing_graphs = num_testing_graphs
        self.target_mode = target_mode
        self.device = device
        self.batch_size = batch_size

        random.seed(1)
        self.load_data()

        self.synthetic_graphs = [None] * len(self.graphs)
        if dataset == 'IMDB':
            self.gen_synthetic_graphs()

        self.init_graph_pairs()

    def gen_synthetic_graphs(self):
        # todo: 每个图生成100个合成图组成图对
        k = 100
        n = len(self.graphs)
        for i, g in enumerate(self.graphs):
            # 无需生成合成图
            if g['n'] <= 10:
                continue
            self.synthetic_graphs[i] = [self.synthesis_graph(g) for j in range(k)]

    @staticmethod
    def synthesis_graph(g):
        # 获取原始图的信息
        n = g['n']  # 节点数
        edge = g['graph'].copy()   # 边，这里要copy，确保不对原图产生影响
        edge_set = set()    # 边集
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        # 开始创建合成图
        new_data = dict()

        # 首先添加一些节点，添加的节点至少有1条边，确保图是连通的
        ni_num = random.randint(0, min(n//10, 5))
        for i in range(0, ni_num):
            n += 1
            x = n-1
            y = random.randint(0, x-1)
            while (x, y) in edge_set:
                y = random.randint(0, x - 1)
            edge_set.add((x, y))
            edge_set.add((y, x))
            edge.append([x, y])
        # 打乱节点的顺序
        permute = list(range(n))
        random.shuffle(permute)

        # 打乱边
        random.shuffle(edge)
        m = len(edge)
        # 接着执行边的删除和添加
        eid_num = random.randint(1, 10) if n <= 20 else random.randint(1, 20)
        ed_num = min(m, random.randint(0, eid_num))
        ei_num = eid_num - ed_num
        # 删除一些边
        edge = edge[:(m - ed_num)]
        # 确定插入边的数量
        if (ei_num + m) > n * (n - 1) // 2:
            ei_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < ei_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - ed_num + ei_num

        new_data["n"] = n
        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_data["m"] = len(new_edge)
        new_data["graph"] = new_edge
        ged = 2 * ni_num + eid_num
        new_data["ged"] = ged
        return new_data

    def load_data(self):
        dataset = self.dataset
        data_location = self.data_location

        base_path = f'{data_location}/{dataset}'

        self.graphs, self.graphs_num = load_all_graphs(f'{base_path}/json')
        # 打乱graphs使数据分布更均匀
        random.shuffle(self.graphs)
        print(f"Load {self.graphs_num} graphs. Generate {(self.graphs_num + 1) * self.graphs_num // 2} graph pairs.")

        # 获取所有图的gid，节点数和边数
        self.gid = [g['gid'] for g in self.graphs]
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]

        dataset_attr = json.load(open(f'{base_path}/properties.json', 'r'))
        self.max_nodes, self.labels = dataset_attr["maxNodes"], dataset_attr["labels"]
        # 如果是带标签的AIDS数据集，加载该数据集的属性信息
        if dataset == 'AIDS_700':
            if self.onehot == 'global':
                self.onehot_dim = len(self.labels)
            elif self.onehot == 'pair':
                self.onehot_dim = dataset_attr["dimension"]
        else:
            # 对于无标签图，初始的嵌入维度为1，都是1
            self.onehot_dim = 1

        # 数据预处理，加载adj和mask
        self.preprocess()

        self.ged_dict = load_ged(f'{data_location}/{dataset}/TaGED.json')
        print(f"Load ged dict. size={len(self.ged_dict)}")

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

    def get_adj_padded_and_mask(self, g):
        n = g['n']
        edges = g['graph']

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

    def init_graph_pairs(self):
        train_val_graphs = []
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []

        train_val_num = int(self.graphs_num * 0.8)
        test_num = int(self.graphs_num * 0.2)

        sg = self.synthetic_graphs
        for i in range(train_val_num):
            if self.gn[i] <= 10:
                for j in range(i, train_val_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        train_val_graphs.append(tmp)
            elif sg[i] is not None:
                k = len(sg[i])
                for j in range(k):
                    train_val_graphs.append((1, i, j))

        random.shuffle(train_val_graphs)
        self.training_graphs = train_val_graphs[0: int(len(train_val_graphs) * 0.9)]
        self.val_graphs = train_val_graphs[int(len(train_val_graphs) * 0.9):]

        # li只收集节点数不超过10的图
        li = []
        for i in range(train_val_num):
            if self.gn[i] <= 10:
                li.append(i)

        # test集合上的图，如果节点数不超过10，从train_val集中随机挑选出100个组成图对，否则使用合成图
        for i in range(train_val_num, train_val_num + test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing_graphs.append((0, i, li[:self.num_testing_graphs]))
            elif sg[i] is not None:
                k = len(sg[i])
                self.testing_graphs.append((1, i, list(range(k))))

        print(f"Generate {len(self.training_graphs)} training graph pairs.")
        print(f"Generate {len(self.val_graphs)} val graph pairs.")
        print(f"Generate {len(self.testing_graphs)} * {self.num_testing_graphs} testing graph pairs.")

    def check_pair(self, i, j):
        if i == j:
            return 0, i, j
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return 0, i, j
        elif (id2, id1) in self.ged_dict:
            return 0, j, i
        else:
            return None

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
            (pair_type, idx_1, idx_2) = graph_pair
            if pair_type == 0:
                if idx_1 == idx_2:
                    ged = 0
                else:
                    # 确保节点数较少的图位于前面
                    gid_pair = (self.gid[idx_1], self.gid[idx_2])
                    if gid_pair not in self.ged_dict:
                        idx_1, idx_2 = (idx_2, idx_1)
                        gid_pair = (gid_pair[1], gid_pair[0])
                    # 根据gid_pair后去图对的真实编辑距离
                    ged = self.ged_dict[gid_pair]

                batch_target_ged.append(ged)
                n1, m1 = (self.gn[idx_1], self.gm[idx_1])
                n2, m2 = (self.gn[idx_2], self.gm[idx_2])

                adj_1, adj_2 = self.adjs[idx_1], self.adjs[idx_2]
                mask_1, mask_2 = self.masks[idx_1], self.masks[idx_2]

                if self.dataset == 'AIDS_700':
                    if self.onehot == 'global':
                        feature_1, feature_2 = self.global_encode(self.graphs[idx_1], self.graphs[idx_2])
                    elif self.onehot == 'pair':
                        feature_1, feature_2 = self.pair_encode(self.graphs[idx_1], self.graphs[idx_2])
                    else:
                        assert False
                else:
                    feature_1, feature_2 = (self.no_label_encode(self.graphs[idx_1]),
                                            self.no_label_encode(self.graphs[idx_2]))
            elif pair_type == 1:
                sg: dict = self.synthetic_graphs[idx_1][idx_2]
                ged = sg['ged']
                batch_target_ged.append(ged)
                n1, m1 = (self.gn[idx_1], self.gm[idx_1])
                n2, m2 = (sg['n'], sg['m'])

                adj_1, mask_1 = self.adjs[idx_1], self.masks[idx_2]
                adj_2, mask_2 = self.get_adj_padded_and_mask(sg)

                feature_1, feature_2 = (self.no_label_encode(self.graphs[idx_1]),
                                        self.no_label_encode(sg))
            else:
                assert False

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

    def pack_graph_pair(self, graph_pair):
        """
        生成图对数据，注意参数的graph_pair是图对的索引，而不是id
        :param graph_pair: (idx_1, idx_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (idx_1, idx_2) = graph_pair

        # 确保节点数较少的图位于前面
        gid_pair = (self.gid[idx_1], self.gid[idx_2])
        if gid_pair not in self.ged_dict:
            idx_1, idx_2 = (idx_2, idx_1)
            gid_pair = (gid_pair[1], gid_pair[0])

        new_data["idx_1"], new_data["idx_2"] = idx_1, idx_2
        new_data["adj_1"], new_data["adj_2"] = self.adjs[idx_1], self.adjs[idx_2]
        new_data["mask_1"], new_data["mask_2"] = self.masks[idx_1], self.masks[idx_2]

        if self.onehot == 'global':
            new_data["features_1"], new_data["features_2"] = self.global_encode(self.graphs[idx_1], self.graphs[idx_2])
        elif self.onehot == 'pair':
            new_data["features_1"], new_data["features_2"] = self.pair_encode(self.graphs[idx_1], self.graphs[idx_2])
        else:
            assert False

        # 根据gid_pair后去图对的真实编辑距离
        real_ged = self.dataset.ged_dict[gid_pair]
        new_data["ged"] = real_ged

        n1, m1 = (self.dataset.gn[idx_1], self.dataset.gm[idx_1])
        n2, m2 = (self.dataset.gn[idx_2], self.dataset.gm[idx_2])
        new_data["n1"] = n1
        new_data["n2"] = n2
        if self.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            new_data["avg_v"] = avg_v
            new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
        elif self.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            new_data["hb"] = higher_bound
            new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)
        else:
            assert False

        return new_data

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

    def no_label_encode(self, graph):
        """
        生成无标签图各个节点的初始嵌入
        :param graph:
        :return:
        """
        features = torch.zeros(self.max_nodes, self.onehot_dim, device=self.device)
        features[:graph["n"], :] = 1
        return features


def sorted_nicely(file_names):
    """
    对文件名进行排序
    :param file_names: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def try_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    import re

    def alphanum_key(s):
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    return sorted(file_names, key=alphanum_key)


def get_file_paths(path, file_format='json'):
    """
    返回排序后的文件路径列表
    :param path: 存放图数据的文件夹路径.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    path = path.rstrip('/')
    paths = sorted_nicely(glob(path + '/*.' + file_format))
    return paths


def iterate_get_graphs(path, file_format):
    """
    读取某个文件夹下的所有图数据
    :param path: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(path, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot']:
            g = json.load(open(file, 'r'))
        else:
            raise RuntimeError('Not supported file format: {}'.format(file_format))
        graphs.append(g)
    return graphs


def load_onehot_dim(path):
    # 读取one_hot_dim
    try:
        with open(f'{path}/dimension.txt', 'r') as f:
            one_hot_dim = int(f.read())
    except (FileNotFoundError, ValueError):
        one_hot_dim = 0
    return one_hot_dim


def load_ged(filepath):
    """
    加载每个图对的GED
    :param filepath: 保存GED的json
    :return:
    """
    ged_dict = dict()
    TaGED = json.load(open(filepath, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ged_dict[(id_1, id_2)] = ged_value
    return ged_dict


def load_all_graph_train_test(path):
    graphs = iterate_get_graphs(f"{path}/train", "json")
    graphs += iterate_get_graphs(f"{path}/test", "json")
    return graphs, len(graphs)


def load_all_graphs(path):
    """
    加载某个目录下的所有图
    :param path: 目录
    :return:
    """
    graphs = iterate_get_graphs(path, "json")
    graphs_num = len(graphs)
    return graphs, graphs_num
