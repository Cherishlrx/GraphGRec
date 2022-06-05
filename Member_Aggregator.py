import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention


class Member_Aggregator(nn.Module):
    """
    Member Aggregator: for aggregating embeddings of group members.
    """

    def __init__(self, features, u2e, g2e, embed_dim, cuda="cpu"):
        super(Member_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.g2e = g2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            e_u = self.u2e.weight[list(tmp_adj)] # fast: member user embedding  # [N, C] 
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_u = torch.t(feature_neigbhors)

            g_rep = self.g2e.weight[nodes[i]] #[C]

            att_w = self.att(e_u, g_rep, num_neighs)
            att_history = torch.mm(e_u.t(), att_w).t() # [C,N] * [N, 1] = [C, 1]
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        # to_feats = self.g2e.weight[to_neighs].squeeze(1) # [B, C]

        return to_feats
