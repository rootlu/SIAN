# coding: utf-8
# author: yf lu
# create date: 2019/7/12 17:10

import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset


class DataUtil(Dataset):
    def __init__(self, arg, root_dir, data_type):
        self.arg = arg
        self.root_dir = root_dir
        self.data_type = data_type
        self.u_items = defaultdict(set)
        self.i_users = defaultdict(set)
        self.data_dir = os.path.join(self.root_dir + '/data/', self.arg.dataset_name)  # ../data/yelp/
        self.data_size = 0
        self.filename = "%s/%s.%s.rating.712" % (self.data_dir, self.arg.dataset_name, self.data_type)
        self.data = defaultdict(int)  # more efficient
        self.neg_num = self.arg.eval_num
        if data_type == 'eval':
            self.get_data_4ranking()
        elif data_type == 'att_analysis':
            self.filename = "%s/%s.att.analysis" % (self.data_dir, self.arg.dataset_name)
            # self.filename = "%s/%s.interaction.graph" % (self.data_dir, self.arg.dataset_name)
            self.load_data()
        else:
            self.load_data()

    def load_data(self):
        with open(self.filename) as f:
            for line in f:
                token = line.split('\t')  # user_id \t item_id \t label \t act_list
                user = int(token[0])
                item = int(token[1])
                label = int(token[2])
                act = token[3].strip()
                self.data[(user, item, label, act)] = 1
                if label == 1:  # positive
                    self.u_items[user].add(item)
                    self.i_users[item].add(user)
                self.data_size += 1

        self.idx2user = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2item = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2label = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2act = np.zeros((self.data_size,), dtype=object)
        for idx, (u, i, l, a) in enumerate(self.data):
            self.idx2user[idx] = u
            self.idx2item[idx] = i
            self.idx2label[idx] = l
            self.idx2act[idx] = list(map(lambda x: int(x), a.split(' ')))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """
        for wechat dataset
        :param idx:
        :return:
        """
        user = self.idx2user[idx]
        item = self.idx2item[idx]
        label = self.idx2label[idx]
        act = self.idx2act[idx]

        sample = {
            'user': user,
            'item': item,
            'label': label,
            'act': act,
        }

        return sample

    def get_data_4ranking(self):
        user_list = []
        item_list = []
        test_ranking_filename = "%s/%s.test.negative.%s" % (self.data_dir, self.arg.dataset_name, self.neg_num)
        with open(test_ranking_filename) as f:
            for line in f:
                token = line.split('\t')
                user_list.append([int(token[0])])
                item_list.append(list(map(lambda _: int(_), token[1:])))
        self.ranking_user = np.array(user_list)
        self.ranking_item = np.array(item_list)

    def load_social_data(self):
        social_graph_filename = "%s/%s.social.graph" % (self.data_dir, self.arg.dataset_name)
        social_relation = defaultdict(list)
        with open(social_graph_filename) as f:
            for line in f:
                token = line.split('\t')  # user_id \t user_id
                social_relation[int(token[0])].append(int(token[1]))
                social_relation[int(token[1])].append(int(token[0]))
        return social_relation

    def load_biz_data(self):
        user_biz_file = "%s/%s.user.biz" % (self.data_dir, self.arg.dataset_name)
        item_biz_file = "%s/%s.item.biz" % (self.data_dir, self.arg.dataset_name)
        user_bizs = defaultdict(list)
        item_bizs = defaultdict(list)
        with open(user_biz_file) as f:
            for line in f:
                token = line.split('\t')  # user_id \t biz_id
                user_bizs[int(token[0])].append(int(token[1]))
        with open(item_biz_file) as f:
            for line in f:
                token = line.split('\t')  # item_id \t biz_id
                item_bizs[int(token[0])].append(int(token[1]))
        return user_bizs, item_bizs
