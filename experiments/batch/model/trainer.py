import os
import random
import sys
import time

import numpy as np
import torch.cuda
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from texttable import Texttable
from tqdm import tqdm

from experiments.batch.model.cgsn.CGSN import CGSN
from experiments.batch.model.cgsn.CGSNImproved import CGSNImproved
from experiments.batch.model.cgsn.CGSNGCN import CGSNGCN
from experiments.batch.model.cgsn.CGSNOneAtt import CGSNOneAtt
from experiments.batch.model.simgnn.SimGNN import SimGNN
from experiments.batch.model.simgnn.SimGNNGConv import SimGNNGConv
from utils.Dataset import Dataset


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.cur_epoch = args.epoch_start
        self.use_gpu = args.use_gpu
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.dataset = Dataset(args.data_location, args.dataset, args.onehot, args.num_testing_graphs, args.target_mode,
                               args.batch_size, self.device)
        self.setup_model()

    def setup_model(self):
        if self.args.model_name == "CGSN":
            self.model = CGSN(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == "CGSNGCN":
            self.model = CGSNGCN(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == "CGSNOneAtt":
            self.model = CGSNOneAtt(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == 'CGSNImproved':
            self.model = CGSNImproved(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == "SimGNN":
            self.model = SimGNN(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == "SimGNNGConv":
            self.model = SimGNNGConv(self.args, self.dataset.onehot_dim).to(self.device)
        else:
            assert False

    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()

        with tqdm(total=len(self.dataset.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",
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

    def prediction_analysis(self, values, info_str=''):
        """
        Analyze the performance of value prediction.
        :param values: an array of (pre_ged - gt_ged); Note that there is no abs function.
        :param info_str:
        :return:
        """
        neg_num = 0
        pos_num = 0
        pos_error = 0.
        neg_error = 0.
        for v in values:
            if v >= 0:
                pos_num += 1
                pos_error += v
            else:
                neg_num += 1
                neg_error += v

        tot_num = neg_num + pos_num
        tot_error = pos_error - neg_error

        pos_error = round(pos_error / pos_num, 3) if pos_num > 0 else None
        neg_error = round(neg_error / neg_num, 3) if neg_num > 0 else None
        tot_error = round(tot_error / tot_num, 3) if tot_num > 0 else None

        with open(f'{self.args.model_path}/{self.args.model_name}/{self.args.dataset}/analysis.txt', 'a') as f:
            print("prediction_analysis", info_str, sep='\t', file=f)
            print("num", pos_num, neg_num, tot_num, sep='\t', file=f)
            print("err", pos_error, neg_error, tot_error, sep='\t', file=f)
            print("--------------------", file=f)

    def create_batches(self):
        random.shuffle(self.dataset.training_graphs)
        batches = []
        for graph in range(0, len(self.dataset.training_graphs), self.args.batch_size):
            batches.append(self.dataset.training_graphs[graph: graph + self.args.batch_size])
        return batches

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        (batch_feat_1, batch_adj_1, batch_mask_1,
         batch_feat_2, batch_adj_2, batch_mask_2,
         batch_avg_v, batch_hb, batch_target_similarity, _) = self.dataset.get_batch_data(batch)
        pre_similarity, _ = self.model(batch_feat_1, batch_feat_2, batch_adj_1, batch_adj_2,
                                       batch_mask_1, batch_mask_2, batch_avg_v, batch_hb)
        losses = losses + len(batch) * F.mse_loss(batch_target_similarity, pre_similarity)

        losses.backward()
        self.optimizer.step()
        return losses.item()

    def score(self, testing_graph_set='test'):
        """
        评估模型表现
        :param testing_graph_set: 在哪个数据集上
        :return:
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.dataset.testing_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.dataset.val_graphs
        else:
            assert False

        self.model.eval()

        # total testing number
        num = 0
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, idx_1, idx_2_list in tqdm(testing_graphs, file=sys.stdout):
            t1 = time.time()

            num += len(idx_2_list)

            batch = [(pair_type, idx_1, idx_2) for idx_2 in idx_2_list]
            (batch_feat_1, batch_adj_1, batch_mask_1,
             batch_feat_2, batch_adj_2, batch_mask_2,
             batch_avg_v, batch_hb, batch_target_similarity, batch_target_ged) = self.dataset.get_batch_data(batch)
            pre_similarity, pre_ged = self.model(batch_feat_1, batch_feat_2, batch_adj_1, batch_adj_2,
                                                 batch_mask_1, batch_mask_2, batch_avg_v, batch_hb)
            # 四舍五入
            round_pre_ged = torch.round(pre_ged)
            # 统计GED 准确命中/feasible 的个数
            num_acc += torch.sum(round_pre_ged == batch_target_ged).item()
            num_fea += torch.sum(round_pre_ged >= batch_target_ged).item()

            mae.append(torch.abs(round_pre_ged - batch_target_ged))
            mse.append((pre_similarity - batch_target_similarity) ** 2)

            t2 = time.time()
            time_usage.append(t2 - t1)
            rho.append(spearmanr(pre_ged.detach().numpy(), batch_target_ged.detach().numpy())[0])
            tau.append(kendalltau(pre_ged.detach().numpy(), batch_target_ged.detach().numpy())[0])
            pk10.append(self.cal_pk(10, pre_ged, batch_target_ged))
            pk20.append(self.cal_pk(20, pre_ged, batch_target_ged))

        time_usage = round(float(np.mean(time_usage)), 3)
        mse = round(torch.mean(torch.stack(mse)).detach().numpy() * 1000, 3)
        mae = round(float(torch.mean(torch.stack(mae)).detach().numpy()), 3)
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
