import os
import numpy as np
import SimpleITK as sitk
import random
import scipy.stats
import json
import time
import re
import copy

import torch
from torch.utils.data import Dataset


class Dataset_breast(Dataset):
    def __init__(self, args, mode='train', valid_fold=1):
        self.mode = mode
        self.valid_fold = valid_fold
        self.image_root = args['data']['image_root']

        with open("data/data.json", 'r', encoding='utf-8') as load_f:
            self.load_dict = json.load(load_f)
        
        f_imgs = np.load('data/f_img.npy')
        f_txts = np.load('data/f_txt.npy')
        with open(os.path.join(args['data']['data_list'], 'all.csv'), 'r') as f:
            strs = f.readlines()
            self.f_imgs = {}
            self.f_txts = {}
            for ai, aid in enumerate([i.split('\n')[0] for i in strs]):
                self.f_imgs[aid] = f_imgs[ai]
                self.f_txts[aid] = f_txts[ai]

        self.aid_list = []
        if self.mode=='test':
            with open(os.path.join(args['data']['data_list'], 'test.csv'), 'r') as f:
                strs = f.readlines()
                self.aid_list = [i.split('\n')[0] for i in strs]
        elif self.mode=='valid':
            with open(os.path.join(args['data']['data_list'], 'fold_{}.csv'.format(valid_fold)), 'r') as f:
                strs = f.readlines()
                self.aid_list = [i.split('\n')[0] for i in strs]
        elif self.mode=='train':
            for i in range(5):
                if i==valid_fold:
                    continue
                with open(os.path.join(args['data']['data_list'], 'fold_{}.csv'.format(i)), 'r') as f:
                    strs = f.readlines()
                    self.aid_list += [i.split('\n')[0] for i in strs]

        self.group_list = [self.load_dict['group'][aid] for aid in self.aid_list]
        self.meta_list = [self.load_dict['meta'][aid] for aid in self.aid_list]
        self.regimen_list = [self.load_dict['regimen'][aid] for aid in self.aid_list]
        self.radiology_list = [self.load_dict['radiology'][aid] for aid in self.aid_list]
        self.pathology_list = [self.load_dict['pathology'][aid] for aid in self.aid_list]
        self.record_list = [self.load_dict['record'][aid] for aid in self.aid_list]
        self.pcr_label_list = [self.load_dict['pcr'][aid] for aid in self.aid_list]
        self.followup_list = [self.load_dict['follwup'][aid] for aid in self.aid_list]
        self.follwoup_day_list = [self.load_dict['followupday'][aid] for aid in self.aid_list]


    def __len__(self):
        return len(self.aid_list)

    def __getitem__(self, index):
        aid = self.aid_list[index]
        meta = self.meta_list[index]
        input_regimen = self.regimen_list[index]
        pcr_label = self.pcr_label_list[index]
        followup_label = self.followup_list[index]
        followup_day = self.follwoup_day_list[index]
        
        output = {
            'path': [aid],
            'meta': torch.from_numpy(np.array(meta, dtype=np.float32)),
            'input_regimen': torch.from_numpy(np.array(input_regimen)),
            'f_img': torch.from_numpy(np.array(self.f_imgs[aid], dtype=np.float32)),
            'f_txt': torch.from_numpy(np.array(self.f_txts[aid], dtype=np.float32)),
            'pcr': pcr_label,
            'followup': followup_label,
            'followup_day': torch.from_numpy(np.array(followup_day)),
        }
        return output