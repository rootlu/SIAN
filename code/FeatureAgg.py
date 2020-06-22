# coding: utf-8
# author: yf lu
# create date: 2019/7/11 14:50
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce


class FeatureAgg(nn.Module):
    def __init__(self, nodes_embeddings, neighbors_embeddings,
                 nodes_profiles, neighbors_profiles,
                 nodes_neighbors, neighbor_types,
                 emb_size, device, fusion, att):
        super(FeatureAgg, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.nodes_embeddings = nodes_embeddings  # Embedding
        self.neighbors_embeddings_dict = neighbors_embeddings  # {type: Embedding}
        self.nodes_profiles = nodes_profiles  # Embedding
        self.neighbors_profiles_dict = neighbors_profiles  # {type: Embedding}
        self.nodes_neighbors_dict = nodes_neighbors  # {type: {node: [neigh]}}
        self.neighbor_types_set = set(neighbor_types)  # [type]
        self.num_neigh_type = len(self.neighbor_types_set)
        self.linear_1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_2 = nn.Linear(self.emb_size, self.emb_size)
        self.linear = nn.Linear(self.emb_size*2, self.emb_size)
        self.fusion = fusion
        self.att = att

        self.w_type_att = nn.Linear(self.emb_size * self.num_neigh_type, self.num_neigh_type, bias=False)

    def forward(self, nodes):
        batch_size = len(nodes)
        nodes_emb = self.nodes_embeddings.weight[nodes]  # (#node, dim)
        nodes_profile = self.nodes_profiles[nodes]
        nodes_fusion = self.fusion(nodes_emb, nodes_profile)

        neigh_type_agg = []  # (len(neigh_type_set), batch, dim)
        for idx, neigh_type in enumerate(self.neighbor_types_set):
            node_neighs = self.nodes_neighbors_dict[neigh_type]
            neighs = list(map(lambda y: list(y), map(lambda x: node_neighs[int(x)], nodes)))
            neigh_agg = self.aggregation(nodes_fusion, neighs, neigh_type, batch_size)  # (batch, dim)
            neigh_type_agg.append(neigh_agg)

        # # aggregate different type neighbor with mlp
        # type_agg = torch.Tensor(reduce(lambda x, y: torch.cat((x, y), 1), neigh_type_agg).cpu()).to(nodes.device)  # (batch, len(neigh_type_set)*dim)
        # neigh_agg_final = F.relu(self.w(type_agg))

        # aggregate different type neighbor with attention
        type_agg = torch.Tensor(reduce(lambda x, y: torch.cat((x, y), 1), neigh_type_agg).cpu()).to(nodes.device)  # (batch, len(neigh_type_set)*dim)
        map_type_agg = self.w_type_att(type_agg)  # (batch, #type)
        att = F.softmax(map_type_agg, dim=1).view(batch_size, self.num_neigh_type, 1)  # (b, #type)
        neigh_agg_final = torch.matmul(torch.transpose(type_agg.view(batch_size, -1, self.emb_size), dim0=1, dim1=2), att).squeeze()  # (batch, dim)
        neigh_agg_final = F.relu(self.linear_2(neigh_agg_final))

        # self-connection could be considered.
        combined_feature = torch.cat([nodes_fusion, neigh_agg_final], dim=1)
        combined_feature = F.relu(self.linear(combined_feature))

        return combined_feature, att

    def aggregation(self, nodes_fusion, nodes_neighbors, neigh_type, batch_size):
        neighbors_embeddings = self.neighbors_embeddings_dict[neigh_type]
        neighbors_profiles = self.neighbors_profiles_dict[neigh_type]

        neighs_fusion = list(map(lambda x: self.fusion(neighbors_embeddings.weight[x], neighbors_profiles[x]),
                                 nodes_neighbors))

        attention_list = list(map(lambda idx: self.att(nodes_fusion[idx], neighs_fusion[idx], len(nodes_neighbors[idx])),
                                  range(batch_size)))
        neigh_feature_matrix = torch.Tensor(list(map(lambda idx: torch.mm(neighs_fusion[idx].t(), attention_list[idx]).data.cpu().numpy(),
                                            range(batch_size)))).reshape(batch_size,self.emb_size).to(nodes_fusion.device)

        # # self-connection could be considered.
        # combined_feature = torch.cat([nodes_fusion, neigh_feature_matrix], dim=1)
        # combined_feature = F.relu(self.linear(combined_feature))

        combined_feature = F.relu(self.linear_1(neigh_feature_matrix))
        # combined_feature = neigh_feature_matrix

        return combined_feature  # (#node, emb_size)
