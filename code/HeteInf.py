# coding: utf-8
# author: yf lu
# create date: 2019/7/8 15:01
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from FeatureAgg import FeatureAgg
from InfluenceProp import InfluenceProp
from Fusion import Fusion
from Attention import Attention


class HeteInf(nn.Module):
    def __init__(self, data_set, user_num, item_num, biz_num,
                 user_profile, item_profile, biz_profile,
                 user_items, item_users, social_rel, user_bizs, item_bizs,
                 emb_size=64, profile_size=64, device='cpu'):
        super(HeteInf, self).__init__()

        self.emb_size = emb_size
        self.profile_size = profile_size
        self.device = device
        self.user_embedding = nn.Embedding(user_num, self.emb_size).to(device)
        self.item_embedding = nn.Embedding(item_num, self.emb_size).to(device)
        self.biz_embedding = nn.Embedding(biz_num, self.emb_size).to(device)
        self.user_profile = user_profile
        self.item_profile = item_profile
        self.biz_profile = biz_profile
        self.user_items = user_items
        self.item_users = item_users
        self.social_rel = social_rel
        self.user_bizs = user_bizs
        self.item_bizs = item_bizs

        self.fusion = Fusion(self.emb_size, self.profile_size)
        self.att = Attention(self.emb_size)
        # -------- Yelp data
        if data_set == 'yelp':
            self.item_feat_agg = FeatureAgg(self.item_embedding, {'user': self.user_embedding},
                                            self.item_profile, {'user': self.user_profile},
                                            {'user': self.item_users}, ['user'], self.emb_size, self.device,
                                            self.fusion, self.att)
            self.user_feat_social_agg = FeatureAgg(self.user_embedding, {'item': self.item_embedding, 'user': self.user_embedding},
                                                   self.user_profile, {'item': self.item_profile, 'user': self.user_profile},
                                                   {'item': self.user_items, 'user': self.social_rel},
                                                   ['item', 'user'], self.emb_size, self.device, self.fusion, self.att)
        elif data_set == 'wxt':
            self.item_feat_agg = FeatureAgg(self.item_embedding, {'user': self.user_embedding, 'biz': self.biz_embedding},
                                            self.item_profile, {'user': self.user_profile, 'biz': self.biz_profile},
                                            {'user': self.item_users, 'biz': self.item_bizs}, ['user','biz'], self.emb_size, self.device,
                                            self.fusion, self.att)
            self.user_feat_social_agg = FeatureAgg(self.user_embedding, {'item': self.item_embedding, 'user': self.user_embedding, 'biz': self.biz_embedding},
                                                   self.user_profile, {'item': self.item_profile, 'user': self.user_profile, 'biz': self.biz_profile},
                                                   {'item': self.user_items, 'user': self.social_rel, 'biz': self.user_bizs},
                                                   ['item', 'user', 'biz'], self.emb_size, self.device, self.fusion, self.att)
        # elif data_set == 'wxt':  # no bizs
        #     self.item_feat_agg = FeatureAgg(self.item_embedding, {'user': self.user_embedding},
        #                                     self.item_profile, {'user': self.user_profile},
        #                                     {'user': self.item_users}, ['user'], self.emb_size, self.device,
        #                                     self.fusion, self.att)
        #     self.user_feat_social_agg = FeatureAgg(self.user_embedding, {'item': self.item_embedding, 'user': self.user_embedding},
        #                                            self.user_profile, {'item': self.item_profile, 'user': self.user_profile},
        #                                            {'item': self.user_items, 'user': self.social_rel},
        #                                            ['item', 'user'], self.emb_size, self.device, self.fusion, self.att)

        self.social_inf_prop = InfluenceProp(self.social_rel, self.emb_size,
                                             self.user_embedding, self.user_profile,
                                             self.fusion, self.att)

        self.w_u1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_u2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_f1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_f2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_i1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_i2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_ui1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_ui2 = nn.Linear(self.emb_size, 16)
        self.w_ui3 = nn.Linear(16, 1)
        self.w_ufi1 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.w_ufi2 = nn.Linear(self.emb_size, 16)
        self.w_ufi3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bnf = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.item_fea_att_analysis = []
        self.user_fea_att_analysis = []
        self.inf_att_analysis = []

    def forward(self, users, items, act_users):
        # aggregate features from neighbors over interaction graph
        # u_embs = self.user_embedding.weight[users]
        # i_embs = self.item_embedding.weight[items]
        i_embs, item_fea_att_list = self.item_feat_agg(items)
        u_embs, user_fea_att_list = self.user_feat_social_agg(users)
        u_inf, inf_att_list = self.social_inf_prop(users, u_embs, items, i_embs, act_users)
        # self.item_fea_att_analysis.append((items.cpu().data.numpy(),
        #                                    map(lambda x: x.reshape(1,-1).cpu().data.numpy(), item_fea_att_list)))
        # self.user_fea_att_analysis.append((users.cpu().data.numpy(),
        #                                    map(lambda x: x.reshape(1,-1).cpu().data.numpy(), user_fea_att_list)))
        # self.inf_att_analysis.append((users.cpu().data.numpy(), items.cpu().data.numpy(), act_users,
        #                               map(lambda x: x.reshape(1,-1).cpu().data.numpy(), inf_att_list)))

        # way-1: w1*u_embs+w2*u_inf+w3*i_embs
        x_u = F.relu(self.bn1(self.w_u1(u_embs)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_u2(x_u)

        x_f = F.relu(self.bnf(self.w_f1(u_inf)))
        x_f = F.dropout(x_f, training=self.training)
        x_f = self.w_f2(x_f)

        x_i = F.relu(self.bn2(self.w_i1(i_embs)))
        x_i = F.dropout(x_i, training=self.training)
        x_i = self.w_i2(x_i)

        x_ufi = torch.cat((x_u, x_f, x_i), 1)
        x = F.relu(self.bn3(self.w_ufi1(x_ufi)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_ufi2(x)))
        x = F.dropout(x, training=self.training)
        scores = torch.sigmoid(self.w_ufi3(x))

        return scores.squeeze()

