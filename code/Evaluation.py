# coding: utf-8
# author: yf lu
# create date: 2019/7/10 20:35
import heapq
import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch


class Evaluation:
    def __init__(self, args):
        self.args = args

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

    def evaluate_classification(self, ground_truth, prediction, thr, return_best_thr):
        y_pred = torch.where(prediction > 0.5, torch.full_like(prediction, 1), prediction)
        y_pred = torch.where(y_pred <= 0.5, torch.full_like(prediction, 0), y_pred)
        y_score = np.array(prediction)
        if thr is not None:
            y_pred = np.zeros_like(y_score)
            y_pred[y_score > thr] = 1

        precision, rec, f1, _ = precision_recall_fscore_support(ground_truth, y_pred, average="binary")
        auc = roc_auc_score(np.array(ground_truth), np.array(prediction))
        rmse = math.sqrt(mean_squared_error(np.array(prediction), np.array(ground_truth)))
        mae = mean_absolute_error(np.array(prediction), np.array(ground_truth))

        if return_best_thr:
            precs, recs, thrs = precision_recall_curve(np.array(ground_truth), y_score)
            f1s = 2 * precs * recs / (precs + recs)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            best_thr = thrs[np.argmax(f1s)]
            return rmse, mae, auc, f1, best_thr
        else:
            return rmse, mae, auc, f1

    def evaluate_ranking(self, users, items, pos_prediction, neg_prediction):
        pos_u = list(np.array(users))
        pos_i = list(map(lambda x: np.array(x[0]).tolist(), items))
        neg_i = list(map(lambda x: np.array(x[1:]).tolist(), items))

        precision = []
        recall = []
        hit = []
        ndcg = []
        ground_truth_u_items = defaultdict(list)
        for idx, u in enumerate(pos_u):
            ground_truth_u_items[u].append(pos_i[idx])

        for idx, u in enumerate(pos_u):
            item_score = {}
            real_score = pos_prediction[idx]
            pred_scores = neg_prediction[idx]

            item_score[pos_i[idx]] = real_score
            for jdx, n_i in enumerate(neg_i[idx]):
                item_score[n_i] = pred_scores[jdx]

            rank_item = heapq.nlargest(self.args.top_k, item_score, key=item_score.get)

            gt_item = ground_truth_u_items[u]
            precision.append(self.get_precision(rank_item, gt_item))
            recall.append(self.get_recall(rank_item, gt_item))
            hit.append(self.get_hit(rank_item, gt_item))
            ndcg.append(self.get_ndcg(rank_item, gt_item))
        return np.array(precision).mean(), np.array(recall).mean(), np.array(hit).mean(), np.array(ndcg).mean()
