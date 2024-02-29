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

from experiments.program_sim.model.cgsn.CGSN import CGSN
from experiments.program_sim.model.funcgnn.funcGNN import funcGNN
from utils.ProgramDataset import ProgramDataset


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.cur_epoch = args.epoch_start
        self.use_gpu = args.use_gpu
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.dataset = ProgramDataset(args.data_location, args.dataset, args.onehot, 
                                      args.target_mode, args.batch_size, self.device)
        self.setup_model()

    def setup_model(self):
        if self.args.model_name == "CGSN":
            self.model = CGSN(self.args, self.dataset.onehot_dim).to(self.device)
        elif self.args.model_name == "funcGNN":
            self.model = funcGNN(self.args, self.dataset.onehot_dim).to(self.device)
        else:
            assert False

    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()

        with tqdm(total=len(self.dataset.training_pairs), unit="graph_pairs", leave=True, desc="Epoch",
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
        random.shuffle(self.dataset.training_pairs)
        batches = []
        for graph in range(0, len(self.dataset.training_pairs), self.args.batch_size):
            batches.append(self.dataset.training_pairs[graph: graph + self.args.batch_size])
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
        testing_pairs = self.dataset.testing_pairs

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

        test_batch_size = 100

        test_batches = []
        for graph in range(0, len(testing_pairs), test_batch_size):
            test_batches.append(testing_pairs[graph: graph + test_batch_size])

        for test_batch in tqdm(test_batches, file=sys.stdout):
            t1 = time.time()

            num += test_batch_size

            (batch_feat_1, batch_adj_1, batch_mask_1,
             batch_feat_2, batch_adj_2, batch_mask_2,
             batch_avg_v, batch_hb, batch_target_similarity, batch_target_ged) = self.dataset.get_batch_data(test_batch)
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

        time_usage = round(float(np.mean(time_usage)), 3)
        mse = round(torch.mean(torch.stack(mse)).detach().numpy() * 1000, 3)
        mae = round(float(torch.mean(torch.stack(mae)).detach().numpy()), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(float(np.mean(rho)), 3)
        tau = round(float(np.mean(tau)), 3)

        table = Texttable()
        table.add_row(["model_name", "dataset", "graph_set", "testing_pairs", "time_usage(s/100p)",
                       "mse", "mae", "acc", "fea", "rho", "tau"])
        table.add_row([self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage,
                       mse, mae, acc, fea, rho, tau])
        table.set_max_width(1000)
        print(table.draw())

        self.append_result_to_file("Testing", table)

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
