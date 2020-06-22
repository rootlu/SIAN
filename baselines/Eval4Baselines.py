# coding: utf-8
# author: yf lu
# create date: 2019/7/16 11:40

import heapq
import math
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch import nn
from sklearn.neural_network import MLPClassifier


class Evaluation4Baselines:
    def __init__(self, u_num, i_num, ranking_num, topk, train_file, valid_file, test_file, u_p, i_p):
        self.data = defaultdict(int)  # more efficient
        self.user_num = u_num
        self.item_num = i_num
        self.ranking_neg_num = ranking_num
        self.top_k = topk
        self.train_file = train_file
        self.valild_file = valid_file
        self.test_file = test_file
        self.user_profile = u_p
        self.item_profile = i_p
        self.id_emb = {}

        self.user_profile = np.load(self.user_profile)
        self.item_profile = np.load(self.item_profile)

    def get_precision(self, rank_item, gt_item):
        p = 0
        for item in rank_item:
            if item in gt_item:
                p += 1
        return p * 1.0 / len(rank_item)

    def get_recall(self, rank_item, gt_item):
        r = 0
        for item in rank_item:
            if item in gt_item:
                r += 1
        return r * 1.0 / len(gt_item)

    def get_dcg(self, rank_item, gt_item):
        dcg = 0.0
        for i in range(len(rank_item)):
            item = rank_item[i]
            if item in gt_item:
                dcg += 1.0 / math.log(i + 2)
        return dcg

    def get_idcg(self, rank_item, gt_item):
        idcg = 0.0
        i = 0
        for item in rank_item:
            if item in gt_item:
                idcg += 1.0 / math.log(i + 2)
                i += 1
        return idcg

    def get_hit(self, rank_item, gt_item):
        for item in rank_item:
            if item in gt_item:
                return 1
        return 0

    def get_ndcg(self, rank_item, gt_item):
        dcg = self.get_dcg(rank_item, gt_item)
        idcg = self.get_idcg(rank_item, gt_item)
        if idcg == 0:
            return 0
        return dcg / idcg

    def evaluate_classification(self, ground_truth, prediction, thr=0.5):
        auc = roc_auc_score(ground_truth, prediction)
        tmp_pred = np.where(prediction > thr, np.full_like(prediction, 1), prediction)
        prediction = np.where(tmp_pred <= thr, np.full_like(prediction, 0), tmp_pred).tolist()

        pre, rec, f1, _ = precision_recall_fscore_support(ground_truth, prediction, average="binary")
        acc = accuracy_score(ground_truth, prediction)
        print('auc: {}, f1: {}, acc: {}, pre: {}, rec: {}'.format(auc, f1, acc, pre, rec))
        # pre = precision_score(ground_truth, prediction)
        # rec = recall_score(ground_truth, prediction)
        # f1 = f1_score(ground_truth, prediction)
        # print('auc: {}, f1: {}, acc: {}, pre: {}, rec: {}'.format(auc, f1, acc, pre, rec))

    def evaluate_ranking(self, prediction, user_list, item_list):
        precision = []
        recall = []
        hit = []
        ndcg = []
        ground_truth_u_items = defaultdict(list)
        for i in range(0, len(user_list), self.ranking_neg_num+1):
            ground_truth_u_items[user_list[i]].append(item_list[i])
        # print(len(ground_truth_u_items))
        for i in range(0, len(user_list), self.ranking_neg_num+1):
            item_score = {}
            for j in range(i, i+self.ranking_neg_num+1):
                item_score[item_list[j]] = prediction[j]

            rank_item = heapq.nlargest(self.top_k, item_score, key=item_score.get)

            gt_item = ground_truth_u_items[user_list[i]]
            precision.append(self.get_precision(rank_item, gt_item))
            recall.append(self.get_recall(rank_item, gt_item))
            hit.append(self.get_hit(rank_item, gt_item))
            ndcg.append(self.get_ndcg(rank_item, gt_item))
        print('precision: {}, recall: {}, hit: {} ndcg: {}'.
              format(np.array(precision).mean(), np.array(recall).mean(), np.array(hit).mean(), np.array(ndcg).mean()))

    def load_embs(self, e_file, baseline):
        print('loading embeddings...')
        if baseline == 'dw':
            with open(e_file) as e_f:
                for line in e_f:
                    token = line.strip().split(' ')
                    if len(token) > 2:
                        self.id_emb[int(token[0])] = list(map(lambda x: float(x), token[1:]))
        elif baseline == 'm2v':
            # for f in emb_file:
            with open(e_file) as e_f:
                for line in e_f:
                    token = line.strip().split(' ')
                    if len(token) > 2 and token[0] != '</s>':
                        self.id_emb[int(token[0][1:])] = list(map(lambda x: float(x), token[1:]))

    def dw(self, thr, feature_flag):
        """
        directly use the output of dw as prediction score
        thr for wechat = 0.2
        :param thr:
        :param feature_flag
        :return:
        """
        print('loading data...')
        user_list, item_list, label_list = [], [], []
        with open(self.test_file) as f:
            for line in f:
                token = line.split('\t')
                if int(token[0]) in self.id_emb and int(token[1])+self.user_num in self.id_emb:
                    user_list.append(int(token[0]))
                    item_list.append(int(token[1]))
                    label_list.append(int(token[2]))
        print ('test size: {}'.format(len(label_list)))

        print('calculating similarity...')
        prediction_score = []
        if feature_flag:
            for i in range(len(user_list)):
                v1 = np.concatenate((np.array(self.id_emb[user_list[i]]), self.user_profile[user_list[i]]))
                v2 = np.concatenate((np.array(self.id_emb[item_list[i]+self.user_num]), self.item_profile[item_list[i]]))  # TODO: new index of item!!!
                prediction_score.append(np.dot(v1, v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2))))
        else:
            for i in range(len(user_list)):
                v1 = np.array(self.id_emb[user_list[i]])
                v2 = np.array(self.id_emb[item_list[i] + self.user_num])  # TODO: new index of item!!!
                prediction_score.append(np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2))))
        print('evaluating performance...')
        self.evaluate_classification(np.array(label_list), np.array(prediction_score), thr)

    def pre_data4LR(self, data_file, pro_or_emb):
        x = []
        y = []
        if pro_or_emb == 'pro_ui':
            with open(data_file) as f:
                for line in f:
                    token = line.split('\t')
                    x.append(np.concatenate((self.user_profile[int(token[0])], self.item_profile[int(token[1])])))  # con. as edge feature
                    # x.append(self.user_profile[int(token[0])] * self.item_profile[int(token[1])])  # elementwise as edge feature
                    y.append(int(token[2]))
        if pro_or_emb == 'pro_uia':
            with open(data_file) as f:
                for line in f:
                    token = line.split('\t')
                    act_list = list(map(lambda x: int(x), token[3].strip().split(' ')))  # act list
                    mean_act_profile = np.mean(self.user_profile[np.array(act_list)],axis=0)
                    # x.append(mean_act_profile)  # con. as edge feature
                    x.append(np.concatenate((self.user_profile[int(token[0])],
                                             self.item_profile[int(token[1])],
                                             mean_act_profile)))  # con. as edge feature
                    # x.append(self.user_profile[int(token[0])]*self.item_profile[int(token[1])]*mean_act_profile)  # con. as edge feature
                    y.append(int(token[2]))
        elif pro_or_emb == 'emb':
            with open(data_file) as f:
                for line in f:
                    token = line.split('\t')
                    x.append(np.concatenate((self.id_emb[int(token[0])], self.id_emb[int(token[1])+self.user_num])))  # con. as edge feature
                    # x.append(self.user_profile[int(token[0])] * self.item_profile[int(token[1])])  # elementwise as edge feature
                    y.append(int(token[2]))

        return x, y

    def log_reg(self, thr, pro_emb_type):
        """
        lr takes nodes' profile as input
        :param thr:
        :param pro_emb
        :return:
        """
        print ('loading data...')
        if pro_emb_type == 'pro_ui':
            print ('taking ui profile as input...')
            x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
            # x_valid, y_valid = self.pre_data4LR(self.valild_file)
            x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)
        if pro_emb_type == 'pro_uia':
            print ('taking uia profile as input...')
            x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
            # x_valid, y_valid = self.pre_data4LR(self.valild_file)
            x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)

        elif pro_emb_type == 'emb':
            print ('taking embedding as input...')
            x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
            # x_valid, y_valid = self.pre_data4LR(self.valild_file)
            x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)

        print ('lr...')
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        # print ('evaluate train...')
        # train_pred = lr.predict(x_train)
        # self.evaluate_classification(np.array(y_train), np.array(train_pred), thr)
        # print ('evaluate test...')
        # test_pred = lr.predict_proba(x_test)[:,1]
        test_pred = lr.predict(x_test)
        self.evaluate_classification(np.array(y_test), np.array(test_pred), thr)
        # print ('mlp...')
        # mlp = MLPClassifier(hidden_layer_sizes=(64, ),random_state=1,max_iter=100)
        # mlp.fit(x_train,y_train)
        # test_pred = mlp.predict(x_test)
        # self.evaluate_classification(np.array(y_test), np.array(test_pred), thr)

    # def mlp(self, thr, prb_emb_type):
    #     w1 = nn.Linear(self.emb_size, self.emb_size)
    #     w2 = nn.Linear(self.emb_size, 1)
    #     print ('loading data...')
    #     if pro_emb_type == 'pro_ui':
    #         print ('taking ui profile as input...')
    #         x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
    #         x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)
    #     if pro_emb_type == 'pro_uia':
    #         print ('taking uia profile as input...')
    #         x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
    #         x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)
    #
    #     elif pro_emb_type == 'emb':
    #         print ('taking embedding as input...')
    #         x_train, y_train = self.pre_data4LR(self.train_file, pro_emb_type)
    #         x_test, y_test = self.pre_data4LR(self.test_file, pro_emb_type)
    #
    #     x_1 = F.relu(self.w_1(x_train))
    #     x_2 = F.relu(self.w_2(x_1))


if __name__ == '__main__':
    # yelp
    train_data = '../data/yelp/yelp.train.rating.712'
    valid_data = '../data/yelp/yelp.val.rating.712'
    test_data = '../data/yelp/yelp.test.rating.712'
    user_profile = '../data/yelp/user_profile.npy'
    item_profile = '../data/yelp/item_profile.npy'
    user_num = 8163
    item_num = 7900
    thr = 0.5

    # # wxt
    # train_data = '../data/wxt/wxt.train.rating.712'
    # valid_data = '../data/wxt/wxt.val.rating.712'
    # test_data = '../data/wxt/wxt.test.rating.712'
    # user_profile = '../data/wxt/user_profile.npy'
    # item_profile = '../data/wxt/item_profile.npy'
    # user_num = 72371
    # item_num = 22218
    # thr = 0.2

    ranking_neg_num = 100
    top_k = 10
    evaluator = Evaluation4Baselines(user_num, item_num, ranking_neg_num, top_k,
                                     train_data, valid_data, test_data,
                                     user_profile, item_profile)

    # emb_file = '../baselines/wxt.dw.embeddings.64'
    emb_file = '../baselines/yelp.dw.embeddings.64'
    evaluator.load_embs(emb_file, 'dw')

    # emb_file = '../baselines/wxt.m2v.uiu.10.50.embs.txt'
    # emb_file = '../baselines/wxt.m2v.uibiu.10.50.embs.txt'
    # emb_file = '../baselines/yelp.m2v.uiu.10.50.embs.txt'
    # evaluator.load_embs(emb_file, 'm2v')

    print ('logistic regression...')
    evaluator.log_reg(thr, 'pro_ui')
    evaluator.log_reg(thr, 'pro_uia')
    evaluator.log_reg(thr, 'emb')
    print ('embedding methods...')
    evaluator.dw(thr, feature_flag=False)  # dw
    print ('embedding methods with profile...')
    evaluator.dw(thr, feature_flag=True)  # dw+feature
