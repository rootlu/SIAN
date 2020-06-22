# coding: utf-8
# author: yf lu
# create date: 2019/7/15 17:19

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, embedding_dims, profile_dim):
        super(Fusion, self).__init__()
        self.embed_dim = embedding_dims
        self.profile_dim = profile_dim
        self.w_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear = nn.Linear(profile_dim, self.embed_dim)

    def forward(self, embedding, profile):
        # reduced_profile = self.linear(profile)
        # feature = torch.cat((embedding, reduced_profile), 1)
        # fusion_feature = F.relu(self.w_1(feature))
        # fusion_feature = F.relu(self.w_2(fusion_feature))
        # return fusion_feature

        # no profile
        return embedding
