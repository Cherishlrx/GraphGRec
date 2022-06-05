import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from Attention import Attention


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, g2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.g2e = g2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        # embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        # for i in range(len(nodes)):
        #     tmp_adj = to_neighs[i]
        #     num_neighs = len(tmp_adj)
        #     # 
        #     e_u = self.g2e.weight[list(tmp_adj)] # fast: user embedding  # [N, C] 
        #     #slow: item-space user latent factor (item aggregation)
        #     #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
        #     #e_u = torch.t(feature_neigbhors)

        #     u_rep = self.u2e.weight[nodes[i]] #[C]

        #     att_w = self.att(e_u, u_rep, num_neighs)
        #     att_history = torch.mm(e_u.t(), att_w).t() # [C,N] * [N, 1] = [C, 1]
        #     embed_matrix[i] = att_history
        # to_feats = embed_matrix
        # neighs_list = [g for g_list in to_neighs for g in g_list] # flatten the nested list to_neighs
        neighs_df = pd.DataFrame(to_neighs).fillna(value=0)

        neighs_list = neighs_df.values.flatten().tolist() # [B]

        mask = [1 if len(n)>0 else 0 for n in to_neighs]  # [B]

        neighs_list, mask = torch.Tensor(neighs_list).long().to(self.device),\
                            torch.Tensor(mask).float().to(self.device)

        to_feats = self.g2e.weight[neighs_list] # [B, C]
        to_feats *= mask.unsqueeze(-1)          # [B, C] * [B, 1]

        # to_feats = self.g2e.weight[neighs_list].squeeze(1) # [B, C]

        return to_feats
