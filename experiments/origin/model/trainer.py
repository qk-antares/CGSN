import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch.cuda
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from texttable import Texttable
from tqdm import tqdm

from experiments.origin.model.gedgnn.GedGNN import GedGNN
from experiments.origin.model.gedgnn.GedMatrixModule import fixed_mapping_loss
from experiments.origin.model.simgnn.SimGNN import SimGNN
from utils.data_loader import load_all_graphs, load_ged


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.cur_epoch = args.epoch_start
        self.use_gpu = args.use_gpu
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.load_data()
        self.transfer_data_to_torch()
        self.init_graph_pairs()

        self.setup_model()

    def setup_model(self):
        if self.args.model_name == "SimGNN":
            self.model = SimGNN(self.args, self.onehot_dim).to(self.device)
        elif self.args.model_name == "GedGNN":
            if self.args.dataset in ["AIDS_700", "Linux"]:
                self.args.value_loss_weight = 10.0
            else:
                self.args.value_loss_weight = 1.0
            self.model = GedGNN(self.args, self.onehot_dim).to(self.device)
        else:
            assert False

    def load_data(self):
        dataset = self.args.dataset
        data_location = self.args.data_location

        base_path = f'{data_location}/{dataset}'

        random.seed(1)
        self.graphs, self.graphs_num = load_all_graphs(f'{base_path}/json')
        random.shuffle(self.graphs)
        self.dataset_num = (self.graphs_num + 1) * self.graphs_num // 2
        print(f"Load {self.graphs_num} graphs. Generate {self.dataset_num} graph pairs.")

        # 获取所有图的gid
        self.gid = [g['gid'] for g in self.graphs]
        # 获取所有图的节点数和边数
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]

        # 加载该数据集的属性信息
        dataset_attr = json.load(open(f'{base_path}/properties.json', 'r'))
        self.max_nodes, self.labels = dataset_attr["maxNodes"], dataset_attr["labels"]

        # 如果是带标签的AIDS数据集，加载该数据集的属性信息
        if dataset == 'AIDS_700':
            if self.args.onehot == 'global':
                self.onehot_dim = len(self.labels)
            elif self.args.onehot == 'pair':
                self.onehot_dim = dataset_attr["dimension"]
            else:
                assert False
        else:
            # 对于无标签图，初始的嵌入维度为1，都是1
            self.onehot_dim = 1

        print(f'one-hot encoding of {dataset}. dim = {self.onehot_dim}')

        self.ged_dict = load_ged(f'{data_location}/{dataset}/TaGED.json')
        print(f"Load ged dict. size={len(self.ged_dict)}")

    def transfer_data_to_torch(self):
        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            # 包含自环
            # edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long().to(self.device)
            self.edge_index.append(edge)

        # 建立节点映射
        n = self.graphs_num
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = self.gid
        mapping = [[None for i in range(n)] for j in range(n)]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float, device=self.device)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float().to(self.device)
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

    def init_graph_pairs(self):
        train_val_graphs = []
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []

        train_val_num = int(self.graphs_num * 0.8)
        test_num = int(self.graphs_num * 0.2)

        for i in range(train_val_num):
            for j in range(i, train_val_num):
                train_val_graphs.append((i, j))

        random.shuffle(train_val_graphs)
        self.training_graphs = train_val_graphs[0: int(len(train_val_graphs) * 0.9)]
        self.val_graphs = train_val_graphs[int(len(train_val_graphs) * 0.9):]

        li = []
        for i in range(train_val_num):
            li.append(i)

        # test集合上的图，都从train_val集中随机挑选出100个组成图对
        for i in range(train_val_num, train_val_num + test_num):
            random.shuffle(li)
            self.testing_graphs.append((i, li[:self.args.num_testing_graphs]))

        print(f"Generate {len(self.training_graphs)} training graph pairs.")
        print(f"Generate {len(self.val_graphs)} val graph pairs.")
        print(f"Generate {len(self.testing_graphs)} * {self.args.num_testing_graphs} testing graph pairs.")

    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()

        with tqdm(total=len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",
                  file=sys.stdout) as pbar:
            batches = self.create_batches()
            loss_sum = 0
            main_index = 0
            for index, batch in enumerate(batches):
                # batch的总loss
                batch_total_loss = self.process_batch(batch)
                # epoch的sum
                loss_sum += batch_total_loss
                # 当前epoch处理的数据条数
                main_index += len(batch)
                # 当前epoch的平均loss
                loss = loss_sum / main_index
                pbar.update(len(batch))
                pbar.set_description("Epoch_{}: loss={} - Batch_{}: loss={}".format(
                    self.cur_epoch + 1, round(1000 * loss, 3),
                    index, round(1000 * batch_total_loss / len(batch), 3)))

            tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
            training_loss = round(1000 * loss, 3)
        # 本epoch训练完成
        t2 = time.time()
        training_time = t2 - t1

        # 记录模型表现
        table = Texttable()
        table.add_row(['model_name', 'dataset', 'graph_set', "current_epoch",
                       "training_time(s/epoch)", "training_loss(1000x)"])
        table.add_row([self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1,
                       training_time, training_loss])
        table.set_max_width(1000)
        print(table.draw())
        self.append_result_to_file("Training", table)

        self.cur_epoch += 1

    def append_result_to_file(self, status, table):
        with open(f'{self.args.model_path}/{self.args.model_name}/{self.args.dataset}/results.txt', 'a') as f:
            print(f"## {status}", file=f)
            print(table.draw(), file=f)

    def create_batches(self):
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph: graph + self.args.batch_size])
        return batches

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        if self.args.model_name == "SimGNN":
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target = data["target"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(target, prediction)
        elif self.args.model_name == "GraphSim":
            # todo: 待实现
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target = data["target"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(target, prediction)
        elif self.args.model_name == "GedGNN":
            weight = self.args.value_loss_weight
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target, gt_mapping = data["target"], data["mapping"]
                prediction, _, mapping = self.model(data)
                losses = losses + fixed_mapping_loss(mapping, gt_mapping) + weight * F.mse_loss(target, prediction)
        elif self.args.model_name == "TaGSim":
            # todo: 待实现
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target = data["target"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(target, prediction)
        else:
            assert False

        losses.backward()
        self.optimizer.step()
        return losses.item()

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

        new_data["idx_1"] = idx_1
        new_data["idx_2"] = idx_2

        new_data["edge_index_1"] = self.edge_index[idx_1]
        new_data["edge_index_2"] = self.edge_index[idx_2]

        new_data["mapping"] = self.mapping[idx_1][idx_2]

        if self.args.dataset == 'AIDS_700':
            if self.args.onehot == 'global':
                new_data["features_1"], new_data["features_2"] = self.global_encode(self.graphs[idx_1],
                                                                                    self.graphs[idx_2])
            elif self.args.onehot == 'pair':
                new_data["features_1"], new_data["features_2"] = self.pair_encode(self.graphs[idx_1],
                                                                                  self.graphs[idx_2])
            else:
                assert False
        else:
            new_data["features_1"], new_data["features_2"] = (self.no_label_encode(self.graphs[idx_1]),
                                                              self.no_label_encode(self.graphs[idx_2]))

        # 根据gid_pair后去图对的真实编辑距离
        real_ged = self.ged[idx_1][idx_2][0]
        ta_ged = self.ged[idx_1][idx_2][1:]
        new_data["ged"] = real_ged

        n1, m1 = (self.gn[idx_1], self.gm[idx_1])
        n2, m2 = (self.gn[idx_2], self.gm[idx_2])
        new_data["n1"] = n1
        new_data["n2"] = n2
        if self.args.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            new_data["avg_v"] = avg_v
            new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
            new_data["ta_ged"] = torch.exp(torch.tensor(ta_ged).float() / -avg_v).to(self.device)
        elif self.args.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            new_data["hb"] = higher_bound
            new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)
            new_data["ta_ged"] = (torch.tensor(ta_ged).float() / higher_bound).to(self.device)
        else:
            assert False

        return new_data

    def single_graph_encode(self, graph, labels_map):
        features = torch.zeros(graph['n'], self.onehot_dim, device=self.device)
        for i, label in enumerate(graph["labels"]):
            index = labels_map[label]
            features[i][index] = 1
        return features

    def global_encode(self, graph1, graph2):
        return self.single_graph_encode(graph1, self.labels), self.single_graph_encode(graph2, self.labels)

    def no_label_encode(self, graph):
        """
        生成无标签图各个节点的初始嵌入
        :param graph:
        :return:
        """
        features = torch.ones(graph["n"], self.onehot_dim, device=self.device)
        return features

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

    def score(self, testing_graph_set='test'):
        """
        评估模型表现
        :param testing_graph_set: 在哪个数据集上
        :return:
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for idx_1, idx_2_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for idx_2 in idx_2_list:
                data = self.pack_graph_pair((idx_1, idx_2))
                # 预期的相似度分数和GED
                target, gt_ged = data["target"].item(), data["ged"]
                # 模型预测的相似度分数和GED
                if self.args.model_name == 'GedGNN':
                    prediction, pre_ged, _ = self.model(data)
                else:
                    prediction, pre_ged = self.model(data)
                # 四舍五入
                round_pre_ged = round(pre_ged)
                num += 1

                mse.append((prediction.item() - target) ** 2)
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1

            t2 = time.time()
            time_usage.append(t2 - t1)
            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(float(np.mean(time_usage)), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(float(np.mean(mae)), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(float(np.mean(rho)), 3)
        tau = round(float(np.mean(tau)), 3)
        pk10 = round(float(np.mean(pk10)), 3)
        pk20 = round(float(np.mean(pk20)), 3)

        table = Texttable()
        table.add_row(["model_name", "dataset", "graph_set", "testing_pairs", "time_usage(s/100p)",
                       "mse", "mae", "acc", "fea", "rho", "tau", "pk10", "pk20"])
        table.add_row([self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage,
                       mse, mae, acc, fea, rho, tau, pk10, pk20])
        table.set_max_width(1000)
        print(table.draw())

        self.append_result_to_file("Testing", table)

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def save(self, epoch):
        """
        保存模型
        :param epoch:
        :return:
        """
        # 检查目录是否存在，如果不存在则创建
        models_path = f'{self.args.model_path}/{self.args.model_name}/{self.args.dataset}/models_dir/'
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        torch.save(self.model.state_dict(), f'{models_path}{str(epoch)}')

    def load(self, epoch):
        """
        加载模型
        :param epoch:
        :return:
        """
        self.model.load_state_dict(
            torch.load(f'{self.args.model_path}/{self.args.model_name}/{self.args.dataset}/models_dir/{str(epoch)}'))
