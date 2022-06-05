'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import heapq
from collections import defaultdict


def LSFs(x, f):
    a = 1.4
    b = 0.8
    c = 0.8
    t = 3
    if f == 1:
        out = (x)/(2*t)
    elif f == 2:
        if (0<= x) and (x <= t):
            out = (a^t-a^(t-x))/(2*a^t-2)
        elif (t<x) and (x<=2*t):
            out = (a^t+a^(x-t)-2)/(2*a^t-2)
    elif f == 3:
        if (0<=x) and (x<=t):
            out = (t^b-(t-x)^b)/(2*t^b)
        elif (t<x) and (x<=2*t):
            out =(t^c+(x-t)^c)/(2*t^c)

    return out


class AGREELoss(nn.Module):
    def __init__(self):
        super(AGREELoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        
        loss = torch.mean((pos_preds - neg_preds - 1).clone().pow(2))

        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        # https://github.com/guoyang9/BPR-pytorch/blob/master/main.py
        # loss = - (pos_preds - neg_preds).sigmoid().log().sum().clone()
        loss = - (pos_preds - neg_preds).sigmoid().log().mean().clone()
        return loss


class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def gen_group_member_dict2(self, path):
        g_m_d = defaultdict(list)
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g, m = int(a[0]), int(a[1])
                g_m_d[g].append(m)
                line = f.readline().strip()
        return g_m_d


    # The following functions are used to evaluate NCF_trans and group recommendation performance
    def evaluate_model(self, model, testRatings, testNegatives, K, type_m, device):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx, device)
            hits.append(hr)
            ndcgs.append(ndcg)

        return (hits, ndcgs)


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx, device):
        p = 0
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users).long().to(device)
        items_var = torch.LongTensor(items).to(device)

        # get the predictions from the trained model
        if type_m == 'group':
            predictions = model(users_var, items_var, 'group')

        if type_m == 'group_fixed_agg':
            predictions = model(users_var, items_var, 'group_fixed_agg')   
            
        elif type_m == 'sa_group':
            predictions = model(users_var, items_var, 'sa_group')  
        elif type_m == 'H-fixed-agg-GR':
            predictions = model(users_var, items_var, 'H-fixed-agg-GR')
        elif type_m == 'target_user_multi':
            predictions = model(users_var, items_var, 'target_user_multi')
        elif type_m == 'target_user_HA':
            predictions = model(users_var, items_var, 'target_user_HA')    
        elif type_m == 'user':
            predictions = model(users_var, items_var, 'user')

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.cpu().data.numpy()[i]
        items.pop() # delete the last item in the list items

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0