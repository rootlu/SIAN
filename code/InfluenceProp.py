# coding: utf-8
# author: yf lu
# create date: 2019/7/10 11:21

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfluenceProp(nn.Module):
    def __init__(self, social_rel, emb_size, user_embs, user_profiles, fusion, att):
        super(InfluenceProp, self).__init__()
        self.social_rel = social_rel
        self.emb_size = emb_size
        self.user_embs = user_embs
        self.user_profiles = user_profiles
        self.linear = nn.Linear(2 * self.emb_size, self.emb_size)
        self.fusion = fusion
        self.att = att
        self.w_c1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_c2 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, users, u_embs, items, i_embs, act_users):
        """
        :param users:
        :param u_embs:
        :param items:
        :param i_embs:
        :param act_users: [np.array(), np.array(),...]
        :param batch_size
        :return:
        """
        batch_size = len(users)
        act_u_fusion = list(map(lambda x: self.fusion(self.user_embs.weight[x], self.user_profiles[x]), act_users))

        # # single influence
        # attention_list = list(map(lambda idx: self.att(u_embs[idx], act_u_fusion[idx], len(act_users[idx])),
        #                           range(batch_size)))
        # neigh_feature_matrix_single = torch.Tensor(
        #     list(map(lambda idx: torch.mm(act_u_fusion[idx].t(), attention_list[idx]).data.cpu().numpy(),
        #              range(batch_size)))).reshape(batch_size, self.emb_size).to(users.device)

        # coupling influence
        coupling_fea = list(map(lambda idx: torch.cat((act_u_fusion[idx],
                                                       i_embs[idx].repeat(len(act_u_fusion[idx]), 1)), 1), range(batch_size)))
        # coupling_fea = list(map(lambda idx: act_u_fusion[idx] + i_embs[idx].repeat(len(act_u_fusion[idx]), 1), range(batch_size)))

        coupling_fea = list(map(lambda idx: F.relu(self.w_c1(coupling_fea[idx])), range(batch_size)))
        coupling_fea = list(map(lambda idx: F.relu(self.w_c2(coupling_fea[idx])), range(batch_size)))
        attention_list = list(map(lambda idx: self.att(u_embs[idx], coupling_fea[idx], len(act_users[idx])),
                                  range(batch_size)))
        neigh_feature_matrix_coupling = torch.Tensor(
            list(map(lambda idx: torch.mm(coupling_fea[idx].t(), attention_list[idx]).data.cpu().numpy(),
                     range(batch_size)))).reshape(batch_size, self.emb_size).to(users.device)

        # # # self-connection could be considered.
        # # combined_feature = torch.cat([u_embs, neigh_feature_matrix], dim=1)
        # # combined_feature = F.relu(self.linear(combined_feature))

        combined_feature = neigh_feature_matrix_coupling

        return combined_feature, attention_list


