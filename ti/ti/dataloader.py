# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_dataloader.ipynb (unless otherwise specified).

__all__ = ['getSTW', 'file_dir', 'splitData', 'DatasetTraj', 'zero_padding']

# Cell
import os
import pandas as pd
import numpy as np
import random
from random import randrange
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from .prep import Transformer

# Cell
file_dir = os.path.dirname(os.path.realpath(__file__)) if '__file__' in globals() else './'
def getSTW(mode='sim'):
    window_file = os.path.join(file_dir, '../data/%s/'%(mode), 'space_time_windows')
    if not os.path.exists(window_file):
        print('Space time window doesn\'t exist create one first!: ', window_file)
        raise NotADirectoryError("Data folder not found")
    with open (window_file, 'rb') as fp:
        space_time_window_list = pickle.load(fp)
    return space_time_window_list

# Cell
def splitData(size):
    train_size = int(size*0.8) # 80%
    val_size = int(size*0.1) # 10%
    test_size = size - (train_size+val_size) # 10%
    return (range(train_size),
            range(train_size, train_size+val_size),
            range(train_size+val_size, train_size+val_size+test_size))

# Cell
class DatasetTraj(Dataset):
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, list_ids, space_time_window_list, mode='sim'):
        self.list_ids = list_ids
        self.mode = mode
        self.space_time_window_list = space_time_window_list
        self.trasformer = Transformer()

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_ids)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        id = self.list_ids[index]
        is_positive = random.getrandbits(1) # label

        # Select sample
        if is_positive:
            # Load data and get label
            if self.mode == 'sim':
                data = pd.read_csv(f'{file_dir}/../data/sim/{str(id)}.csv')
            else:
                window = self.space_time_window_list[id]
                tid = random.choice(window)
                data = pd.read_csv(f'{file_dir}/../data/real/{str(int(tid))}.csv')
            x1, org= self.trasformer.transform(data)
            total_steps = len(x1)
            dst_idx = randrange(int(0.7*total_steps), total_steps - 1)
            dst = x1[dst_idx]
            c_range = randrange(int(.25*dst_idx), int(.9*dst_idx))#total_steps#
            x1 = x1[:c_range]
            org = org[:c_range]
            dst = [dst] * len(org)
            x2 = [org, dst]
            y = 1
        else:
            # Load data and get label
            window = self.space_time_window_list[id]
            ids = random.sample(window, 2)
            pid, nid = ids[0], ids[1]
            if self.mode == 'sim':
                pid, nid = ids[0], ids[1]
                pos_data = pd.read_csv(f'{file_dir}/../data/sim/{str(int(pid))}.csv')
                neg_data = pd.read_csv(f'{file_dir}/../data/sim/{str(int(nid))}.csv')
            else:
                pos_data = pd.read_csv(f'{file_dir}/../data/real/{str(int(pid))}.csv')
                neg_data = pd.read_csv(f'{file_dir}/../data/real/{str(int(nid))}.csv')
            pos_x1, pos_org = self.trasformer.transform(pos_data)
            neg_x1, neg_org = self.trasformer.transform(neg_data)

            neg_total_steps = len(neg_x1)
            pos_total_steps = len(pos_x1)
            dst_idx = randrange(int(0.7*pos_total_steps), pos_total_steps - 1)
            dst = pos_x1[dst_idx]
            c_range = randrange(int(.25*neg_total_steps), int(.9*neg_total_steps))
            x1 = neg_x1[:c_range]
            org = [neg_org[0]] * len(x1)
            dst = [dst] * len(x1)
            x2 = [org, dst]
            y = 0

        return x1, x2, y


# Cell
def zero_padding(batch):
    '''Pads batch of variable length with leading zeros'''
    x1 = [item[0] for item in batch]
    x2_org = [item[1][0] for item in batch]
    x2_dst = [item[1][1] for item in batch]
    y = [item[2] for item in batch]
    x_seq_lens = [len(item) for item in x1]
    max_seq_len = max(x_seq_lens)
    n_dim = len(x1[0][0])
    x1_pad = torch.FloatTensor([
        np.zeros((max_seq_len-len(item), n_dim)).tolist()+item
        for item in x1
    ])
    x2_org_pad = torch.FloatTensor([
        np.zeros((max_seq_len-len(item), n_dim)).tolist()+item
        for item in x2_org
    ])
    x2_dst_pad = torch.FloatTensor([
        np.zeros((max_seq_len-len(item), n_dim)).tolist()+item
        for item in x2_dst
    ])
    return x1_pad, (x2_org_pad, x2_dst_pad), y, x_seq_lens, max_seq_len