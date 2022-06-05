import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
from time import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
from Member_Aggregator import Member_Aggregator
from G_Aggregator import G_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
# import matplotlib.pyplot as plt
from utils import Helper, AGREELoss, BPRLoss
from dataset import GDataset

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
class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, enc_gro, enc_grov_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.enc_gro = enc_gro
        self.enc_grov_history = enc_grov_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v, type_m):
        if type_m == 'user':
            embeds_u = self.enc_u(nodes_u) # User Modeling: Item aggregation & Social aggregation
            embeds_v = self.enc_v_history(nodes_v) # Item Modeling: User aggregation
        elif type_m == 'group':
            embeds_u = self.enc_gro(nodes_u) # Group preference Modeling: Item aggregation & Member aggregation
            embeds_v = self.enc_grov_history(nodes_v) # Item Modeling: Group aggregation

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        # Rating Prediction
        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, type_m, labels_list):
        scores = self.forward(nodes_u, nodes_v, type_m)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, type_m):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), type_m, labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return running_loss


def test(model, device, test_loader, type_m):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v, type_m)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def main(): 
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--path', type=str, default='./CAMRa2011/sparser_datagenerate/t3result_user')
    parser.add_argument('--user_dataset', type=str, default= './CAMRa2011/sparser_datagenerate/' + 'userRating')
    parser.add_argument('--group_dataset', type=str, default= './CAMRa2011/sparser_datagenerate/' + 't3_groupRating')
    parser.add_argument('--user_in_group_path', type=str, default= './CAMRa2011/sparser_datagenerate/groupMember')
    
    parser.add_argument('--batch_size_list', type=list, default=[64], metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim_list', type=list, default=[40], metavar='N', help='embedding size')
    parser.add_argument('--lr_list', type=list, default=[0.005], metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=20, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=80, metavar='N', help='number of epochs to train')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = True
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # device = torch.device('cpu')

    print('Load data...')
    torch.random.manual_seed(1314)
    # initial helper
    helper = Helper()

    # get the dict of users in group: group_id: [member1, member2,...]
    g_m_d = helper.gen_group_member_dict2(args.user_in_group_path)

    # initial dataset class
    dataset = GDataset(args.user_dataset, args.group_dataset, args.user_in_group_path)

    # get train user data
    history_u_lists, history_ur_lists = dataset.u_items_dict, dataset.u_rates_dict
    history_v_lists, history_vr_lists = dataset.i_users_dict, dataset.i_rates_dict
    train_u, train_v, train_r = dataset.train_userList, dataset.train_itemList, dataset.train_ratingList
    # get test user data
    test_u, test_v, test_r = dataset.test_userList, dataset.test_itemList, dataset.test_ratingList

    # get rating dict: initial_rating: [normalized rating (1-5)]
    ratings_list = dataset.rating_dict

    # get users' group information: user_id: [group1, group2,...]
    u_g_d = dataset.social_adj_dict

    # get train group data
    gro_items_dict, gro_rates_dict = dataset.gro_items_dict, dataset.gro_rates_dict
    i_groups_dict, groi_rates_dict = dataset.i_groups_dict, dataset.gro_i_rates_dict
    train_gro, train_grov, train_gror = dataset.train_groList, dataset.train_groitemList, dataset.train_groratingList
    # get test group data
    test_gro, test_grov, test_gror = dataset.test_groList, dataset.test_groitemList, dataset.test_groratingList

    num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()
    num_items = max(history_v_lists.keys()) + 1
    num_ratings = ratings_list.__len__()
    # get group number
    num_groups = len(g_m_d)
    

    print('Data prepare is over!')

    for batch_size in args.batch_size_list:
        for embed_dim in args.embed_dim_list:
            for lr in args.lr_list:
                for i in range(9):
                    # user dataloader
                    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                                            torch.FloatTensor(train_r))
                    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                                            torch.FloatTensor(test_r))
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
                    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
                    ## group dataloader
                    gro_trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_gro), torch.LongTensor(train_grov),
                                                            torch.FloatTensor(train_gror))
                    gro_testset = torch.utils.data.TensorDataset(torch.LongTensor(test_gro), torch.LongTensor(test_grov),
                                                            torch.FloatTensor(test_gror))
                    gro_train_loader = torch.utils.data.DataLoader(gro_trainset, batch_size=batch_size, shuffle=True, drop_last=True)
                    gro_test_loader = torch.utils.data.DataLoader(gro_testset, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

                    u2e = nn.Embedding(num_users, embed_dim).to(device)
                    v2e = nn.Embedding(num_items, embed_dim).to(device)
                    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)

                    g2e = nn.Embedding(num_groups, embed_dim).to(device)

                    # User feature
                    # features: item * rating
                    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
                    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
                    # neighobrs
                    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, g2e, embed_dim, cuda=device)
                    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, u_g_d, agg_u_social,
                                        base_model=enc_u_history, cuda=device)

                    # item feature: user * rating
                    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
                    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

                    # Group feature
                    # features: item * rating
                    agg_gro_history = UV_Aggregator(v2e, r2e, g2e, embed_dim, cuda=device, uv=True)
                    enc_gro_history = UV_Encoder(g2e, embed_dim, gro_items_dict, gro_rates_dict, agg_gro_history, cuda=device, uv=True)

                    # group members
                    agg_gro_member = Member_Aggregator(lambda nodes: enc_gro_history(nodes).t(), u2e, g2e, embed_dim, cuda=device)
                    enc_gro = Social_Encoder(lambda nodes: enc_gro_history(nodes).t(), embed_dim, g_m_d, agg_gro_member,
                                        base_model=enc_gro_history, cuda=device)

                    # item feature: user * rating
                    agg_grov_history = G_Aggregator(v2e, r2e, g2e, embed_dim, cuda=device)
                    enc_grov_history = UV_Encoder(v2e, embed_dim, i_groups_dict, groi_rates_dict, agg_grov_history, cuda=device, uv=False)

                    
                    # model
                    graphrec = GraphRec(enc_u, enc_v_history, enc_gro, enc_grov_history, r2e).to(device)
                    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=lr, alpha=0.9)

                    print('grahrec at train batch_size %d, embed_dim %d and lr at %1.5f'%(batch_size, embed_dim, lr))

                    best_rmse = 9999.0
                    best_mae = 9999.0
                    endure_count = 0
                    u_RMSE = []
                    u_MAE = []
                    gro_RMSE = []
                    gro_MAE = []
                    cost_train_u = []
                    cost_train_gro = []
                    for epoch in range(1, args.epochs + 1):
                        # begin to train
                        t1 = time()
                        # train the user
                        running_loss_u = train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, 'user')
                        cost_train_u.append(running_loss_u/len(train_loader)*batch_size)
                        
                        # train the group
                        # running_loss_gro = train(graphrec, device, gro_train_loader, optimizer, epoch, best_rmse, best_mae, 'group')
                        # cost_train_gro.append(running_loss_gro/len(train_loader)*batch_size)
                        print("user and group training time is: [%.1f s]" % (time()-t1))

                        # test the user
                        expected_rmse, mae = test(graphrec, device, test_loader, 'user')

                        # early stopping (no validation set in toy dataset)
                        if best_rmse > expected_rmse:
                            best_rmse = expected_rmse
                            best_mae = mae
                            endure_count = 0
                        else:
                            endure_count += 1
                        print("User rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
                        u_RMSE.append(expected_rmse)
                        u_MAE.append(mae)

                        # test the group
                        # gro_expected_rmse, gro_mae = test(graphrec, device, gro_test_loader, 'group')
                        
                        # # early stopping (no validation set in toy dataset)
                        # if best_rmse > gro_expected_rmse:
                        #     best_rmse = gro_expected_rmse
                        #     best_mae = gro_mae
                        #     endure_count = 0
                        # else:
                        #     endure_count += 1
                        # print("Group rmse: %.4f, mae:%.4f " % (gro_expected_rmse, gro_mae))
                        # gro_RMSE.append(gro_expected_rmse)
                        # gro_MAE.append(gro_mae)

                        if endure_count > 150:
                            break
                        
                    # EVA_data = np.column_stack((u_RMSE, u_MAE, gro_RMSE, gro_MAE, cost_train_u))
                    EVA_user = np.column_stack((u_RMSE, u_MAE))
                    # EVA_gro = np.column_stack((gro_RMSE, gro_MAE, cost_train_gro))

                    print("save to file...")
                    
                    if not os.path.exists(args.path):
                        os.makedirs(args.path)

                    # filename = "EVA_UserRMSE_MAE_GroRMSE_MAE_cost_bat%d_emb%d_lr%1.5f"%(batch_size, embed_dim, lr)
                    filename = "EVA_UserRMSE_MAE_bat%d_emb%d_lr%1.5f_%d"%(batch_size, embed_dim, lr, i)
                    # filename = "EVA_groRMSE_MAE_maeTrain_bat%d_emb%d_lr%1.5f_%d"%(batch_size, embed_dim, lr, i)
                    
                    filename = os.path.join(args.path, filename)

                    np.savetxt(filename, EVA_user, fmt='%1.4f', delimiter=' ')

                    # plot
                    # x = np.arange(1,len(EVA_data)+1,1)
                    # uRMSE = EVA_data[:,0]
                    # uMAE = EVA_data[:,1]
                    # gRMSE = EVA_data[:,2]
                    # gMAE = EVA_data[:,3]
                    # plt.plot(x, uRMSE,label='uRMSE')
                    # plt.plot(x, uMAE,label='uMAE')
                    # plt.plot(x, gRMSE,label='gRMSE')
                    # plt.plot(x, gMAE,label='gMAE')

                    # plt.legend(loc='upper right')
                    # plt.savefig(os.path.join(args.path, "loss.png"))
                    # plt.show()

                    print("Done!")

if __name__ == "__main__":
    main()
