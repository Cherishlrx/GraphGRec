'''
Created on Oct. 24 2020
Processing datasets.

@author: Ruxia Liang (1248824850@qq.com)
'''
from collections import defaultdict, Counter
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

class GDataset(object):

    def __init__(self, user_path, group_path, user_in_group_path):
        '''
        Constructor
        '''
        # user data
        self.train_userList, self.train_itemList, self.train_ratingList = self.load_rating_file_as_list(user_path + "Train.txt")
        self.test_userList, self.test_itemList, self.test_ratingList = self.load_rating_file_as_list(user_path + "Test.txt")
        
        self.u_items_dict, self.u_rates_dict = self.get_u_ir_dict(user_path + "Train.txt")
        self.i_users_dict, self.i_rates_dict = self.get_i_ur_dict(user_path + "Train.txt")
        
        self.rating_dict = self.get_rating_dict(self.train_ratingList)

        self.social_adj_dict = self.get_social_adj_dict(user_in_group_path)
        # group data
        self.train_groList, self.train_groitemList, self.train_groratingList = self.load_rating_file_as_list(group_path + "Train.txt")
        self.test_groList, self.test_groitemList, self.test_groratingList = self.load_rating_file_as_list(group_path + "Test.txt")

        self.gro_items_dict, self.gro_rates_dict = self.get_u_ir_dict(group_path + "Train.txt")
        self.i_groups_dict, self.gro_i_rates_dict = self.get_i_ur_dict(group_path + "Train.txt")


    def load_rating_file_as_list(self, filename):
        userList, itemList, ratingList = [], [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                # user, item, rating = map(int, arr)
                userList.extend([user])
                itemList.extend([item])
                ratingList.extend([rating])
                line = f.readline()
        return userList, itemList, ratingList

    def get_u_ir_dict(self, filename):
        # user_i -> items: 1: [10, 23]
        with open(filename, 'r') as reader:
            u_item = defaultdict(list)
            u_rate = defaultdict(list)
            for line in reader:
                # 162540,32,4,...
                user_id, item_id, rate = map(int, line.split(' ')[:3]) # ',' or '\t'
                u_item[user_id].append(item_id)
                u_rate[user_id].append(rate)
        return u_item, u_rate


    def get_i_ur_dict(self, filename):
            # item_i -> users: 1: [10, 23]
            with open(filename, 'r') as reader:
                i_users = defaultdict(list)
                i_rates = defaultdict(list)
                for line in reader:
                    user_id, item_id, rate = map(int, line.split(' ')[:3]) # ',' or '\t'
                    i_users[item_id].append(user_id)
                    i_rates[item_id].append(rate)
            return i_users, i_rates

    def get_social_adj_dict(self, filename):
        # user --> groups that the user has joined in: 1: [22, 13]
        with open(filename, 'r') as reader:
            social_adj_dict = defaultdict(list)
            for line in reader:
                arr = line.split(' ')
                group_id, member_ids = int(arr[0]), map(int, arr[1].split(','))
                for member_id in member_ids:
                    social_adj_dict[member_id].append(group_id)

            return social_adj_dict


    def get_rating_dict(self, train_ratingList):
        unique_rating = list(set(self.test_ratingList))
        rating_dict = {}
        for r in unique_rating:
            rating_dict[r] = [] 
            if r >=0 and r<=20:
                rating_dict[r].append(1)
            elif r > 20 and r <= 40:
                rating_dict[r].append(2)
            elif r > 40 and r <= 60:
                rating_dict[r].append(3)
            elif r > 60 and r <= 85:
                rating_dict[r].append(4)
            else:
                rating_dict[r].append(5)

        return rating_dict

    



    


            















