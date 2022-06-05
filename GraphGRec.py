import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AGREELoss, BPRLoss

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""
class GraphGRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, drop_ratio):
        super(GraphGRec, self).__init__()
        self.enc_u = enc_u
        # self.enc_gro = enc_gro
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.drop_ratio = drop_ratio

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        # self.predictlayer = PredictLayer(3 * self.embed_dim, drop_ratio)
        # self.criterion = nn.MSELoss()
        self.criterion = AGREELoss()

    def forward(self, nodes_u, nodes_v, type_m):
        if type_m == 'group':
            embeds_gro = self.enc_gro(nodes_u) # User Modeling: Item aggregation & Social aggregation
            embeds_v = self.enc_v_history(nodes_v) # Item Modeling: User aggregation

            x_gro = F.relu(self.bn1(self.w_ur1(embeds_gro)))
            x_gro = F.dropout(x_gro, training=self.training)
            x_gro = self.w_ur2(x_gro)
            x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
            x_v = F.dropout(x_v, training=self.training)
            x_v = self.w_vr2(x_v)

            # Rating Prediction
            element_embeds = torch.mul(x_gro, x_v) 
            x_uv = torch.cat((element_embeds, x_gro, x_v), dim=1)
            x = F.relu(self.bn3(self.w_uv1(x_uv)))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.bn4(self.w_uv2(x)))
            x = F.dropout(x, training=self.training)
            scores = self.w_uv3(x)
            preds_gro = torch.sigmoid(scores)

            return preds_gro.squeeze()

        elif type_m == 'user':
            embeds_u = self.enc_u(nodes_u) # User Modeling: Item aggregation & Social aggregation
            embeds_v = self.enc_v_history(nodes_v) # Item Modeling: User aggregation
            # element_embeds = torch.mul(embeds_u, embeds_v)  # Element-wise product
            # new_embeds = torch.cat((element_embeds, embeds_u, embeds_v), dim=1)
            # preds_user = torch.sigmoid(self.predictlayer(new_embeds))
            x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
            x_u = F.dropout(x_u, training=self.training)
            x_u = self.w_ur2(x_u)
            x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
            x_v = F.dropout(x_v, training=self.training)
            x_v = self.w_vr2(x_v)
            # Prediction -->> [0, 1]
            element_embeds = torch.mul(x_u, x_v) 
            x_uv = torch.cat((element_embeds, x_u, x_v), dim=1)
            x = F.relu(self.bn3(self.w_uv1(x_uv)))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.bn4(self.w_uv2(x)))
            x = F.dropout(x, training=self.training)
            scores = self.w_uv3(x)
            preds_user = torch.sigmoid(scores)

            return preds_user.squeeze()

class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out