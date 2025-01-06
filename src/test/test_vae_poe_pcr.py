import argparse
import os
import sys
import logging
import yaml
import random
from tqdm import tqdm
import numpy as np
import scipy
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from model.breastVAE import BreastVAE
from dataloader.loader_all_f import Dataset_breast
from utils.inference import load_weights


def train(args, net, device):
    np.random.seed(2)
    test_data = Dataset_breast(args, mode='test')

    n_test = len(test_data)

    valid_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    os.makedirs(args['vae_poe']['result'], exist_ok=True)
    with open(os.path.join(args['vae_poe']['result'], 'ROC_pcr_scores.csv'), 'w') as f:
        f.write('aid,label,poe\n')

    with torch.no_grad():
        for i in range(len(net)):
            net[i].eval()

        pcr_results_sub = {
            'label': [],
            'tab_img_txt': [],
        }

        for batch in valid_loader:
            f_img = batch['f_img'].to(device=device, dtype=torch.float32)
            f_txt = batch['f_txt'].to(device=device, dtype=torch.float32)
            meta = batch['meta'].to(device=device, dtype=torch.float32)
            input_regimen = batch['input_regimen'].to(device=device, dtype=torch.float32)
            pcr_label = batch['pcr'].to(device=device, dtype=torch.int64)
            aid = batch['path'][0][0]

            pcrs_tab_img_txt = 0
            for i in range(len(net)):
                output = net[i](input_regimen, meta, f_img, f_txt, sample=False)
                
                pcrs_tab_img_txt += (nn.Softmax(dim=1)(output['pcr_poe_all'])/len(net))
                
            pcr_results_sub['tab_img_txt'].append(pcrs_tab_img_txt[:,1].item())
            pcr_results_sub['label'].append(pcr_label.item())

            with open(os.path.join(args['vae_poe']['result'], 'ROC_pcr_scores.csv'), 'a+') as f:
                f.write('{},{},{}\n'.format(aid, pcr_label.item(), pcrs_tab_img_txt[:,1].item()))
           

def get_args():
    parser = argparse.ArgumentParser(description='Test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config/config.yaml',
                        help='config file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cuda',
                        help='cuda or cpu')
    
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    dir_checkpoint = config['vae_poe']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = config['vae_poe']['vis']
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = []
    for i in range(5):
        if not os.path.exists(os.path.join(config['vae_poe']['ckpt'], str(i), 'ckpt_pcr_best.pth')):
            print('Fold {} not found!'.format(i))
            continue
        net.append(BreastVAE(in_dim=config['vae_poe']['in_dim'], regimen_dim=5, hidden_dim=256, latent_dim=3))
        net[-1].to(device=device)
        ckpt_path = os.path.join(config['vae_poe']['ckpt'], str(i), 'ckpt_pcr_best.pth')
        net[-1] = load_weights(net[-1], ckpt_path, device=device)

    
    try:
        train(
            config,
            net=net,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)