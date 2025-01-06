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


    def __len__(self):
        return len(self.aid_list)
    
    def prep(self, path):
        arr1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, 'dce1.nii.gz')))
        arr2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, 'dce2.nii.gz')))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, 'seg.nii.gz')))
        
        amax = np.percentile(np.concatenate([arr1, arr2], axis=0), 99.5)
        arr1 = arr1/amax
        arr2 = arr2/amax
        arr = np.stack([arr1, arr2], axis=0)
        seg = np.expand_dims(seg, axis=0)
        return arr, seg

    def __getitem__(self, index):
        aid = self.aid_list[index]
        meta = self.meta_list[index]
        input_regimen = self.regimen_list[index]

        radiology = self.radiology_list[index]
        pathology = self.pathology_list[index]
        record = self.record_list[index]

        date = os.listdir(os.path.join(self.image_root, aid))[0]
        image, seg = self.prep(os.path.join(self.image_root, aid, date))
        
        output = {
            'path': [aid],
            'meta': torch.from_numpy(np.array(meta, dtype=np.float32)),
            'input_regimen': torch.from_numpy(np.array(input_regimen)),
            'image': torch.from_numpy(np.array(image, dtype=np.float32)),
            'seg': torch.from_numpy(np.array(seg, dtype=np.float32)),
            'radiology': radiology,
            'pathology': pathology,
            'record': record,
        }
        return output