import argparse
import os
import sys
import logging
import yaml
import random
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from collections import Counter
from lifelines.utils import concordance_index

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from model.breastVAE import BreastVAE
from dataloader.loader_all_f import Dataset_breast
from utils.record import Plotter, Recorder
from utils.inference import load_weights


def train(args, net, device):
    train_data = Dataset_breast(args, valid_fold=args['args']['fold'], mode='train')
    valid_data = Dataset_breast(args, valid_fold=args['args']['fold'], mode='valid')

    n_train = len(train_data)#len(dataset) - n_val
    n_valid = len(valid_data)#len(dataset) - n_val

    label_counts = Counter(train_data.group_list)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    weights = [class_weights[train_data.group_list[i]] for i in range(n_train)]
    weights = torch.DoubleTensor(weights)

    train_loader = DataLoader(train_data, batch_size=655, num_workers=0, pin_memory=True,
                              sampler=WeightedRandomSampler(weights, n_train))
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    n_copy = 1
    epochs = args['vae_poe']['epochs']
    lr = np.float32(args['vae_poe']['lr'])
    dir_visualize = os.path.join(config['vae_poe']['vis'], str(args['args']['fold']))
    dir_checkpoint = os.path.join(config['vae_poe']['ckpt'], str(args['args']['fold']))
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size:   {n_valid}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    
    recorder = Recorder(['loss', 'pcr_auc', 'followup_cindex'])
    plotter = Plotter(dir_visualize, keys1=['loss'], keys2=['pcr_auc', 'followup_cindex'])
    
    with open(os.path.join(dir_checkpoint, 'log_poe_pcr.csv'), 'w') as f:
        f.write('epoch,loss,pcr_auc,followup_cindex\n')
    with open(os.path.join(dir_checkpoint, 'log_poe_followup.csv'), 'w') as f:
        f.write('epoch,loss,pcr_auc,followup_cindex\n')

    best_pcr_auc = 0
    best_followup_cindex = 0

    for epoch in range(epochs):
        net.train()
        
        train_losses = []
        
        with tqdm(total=n_train*n_copy, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            
            for batch in train_loader:
                f_img = batch['f_img'].to(device=device, dtype=torch.float32)
                f_txt = batch['f_txt'].to(device=device, dtype=torch.float32)
                meta = batch['meta'].to(device=device, dtype=torch.float32)
                input_regimen = batch['input_regimen'].to(device=device, dtype=torch.float32)
                followup_label = batch['followup'].to(device=device, dtype=torch.float32)
                ldd = batch['followup_day'].to(device=device, dtype=torch.int64)
                pcr_label = batch['pcr'].to(device=device, dtype=torch.int64)
                
                for _ in range(n_copy):
                    output = net(input_regimen, meta, f_img, f_txt, sample=True)

                    loss_pcr = nn.CrossEntropyLoss()(output['pcr_poe_all'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_poe_img'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_poe_txt'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_poe_all0'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_poe_img0'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_poe_txt0'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_tab'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_img'], pcr_label) + \
                        nn.CrossEntropyLoss()(output['pcr_txt'], pcr_label)
                    loss_followup = net.cox_loss(output['hazard_poe_all'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_poe_img'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_poe_txt'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_poe_all0'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_poe_img0'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_poe_txt0'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_tab'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_img'], ldd, followup_label) + \
                        net.cox_loss(output['hazard_txt'], ldd, followup_label)
                    
                    loss_kl = output['kl_tab'] + output['kl_img'] + output['kl_txt']
                    
                    loss = loss_pcr + loss_followup + loss_kl
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                
                    train_losses.append(loss.item())
                    pbar.set_postfix(**{'loss': loss.item(), 'loss_pcr': loss_pcr.item(), 'loss_follow': loss_followup.item()})
                    pbar.update(meta.shape[0])
    
        train_losses = np.mean(train_losses)

        with torch.no_grad():
            net.eval()

            pcr_labels = []
            pcr_scores = []
            followup_labels = []
            followup_day = []
            followup_hazard = []

            for batch in valid_loader:
                f_img = batch['f_img'].to(device=device, dtype=torch.float32)
                f_txt = batch['f_txt'].to(device=device, dtype=torch.float32)
                meta = batch['meta'].to(device=device, dtype=torch.float32)
                input_regimen = batch['input_regimen'].to(device=device, dtype=torch.float32)
                pcr_label = batch['pcr'].to(device=device, dtype=torch.int64)
                followup_label = batch['followup'].to(device=device, dtype=torch.int64)
                ldd = batch['followup_day'].to(device=device, dtype=torch.int64)

            
                output = net(input_regimen, meta, f_img, f_txt, sample=False)
                pcr_scores.append(nn.Softmax(dim=1)(output['pcr_poe_all'])[:,1].item())
                pcr_labels.append(pcr_label.item())
                
                followup_hazard.append(-output['hazard_poe_all'].item())
                followup_labels.append(followup_label.item())
                followup_day.append(ldd.item())


        fpr, tpr, thresholds = metrics.roc_curve(np.array(pcr_labels), np.array(pcr_scores), pos_label=1)
        pcr_auc = metrics.auc(fpr, tpr)

        followup_cindex = concordance_index(np.array(followup_day), np.array(followup_hazard), np.array(followup_labels))
        
        recorder.update({'loss': train_losses, 'pcr_auc': pcr_auc, 'followup_cindex': followup_cindex})
        plotter.send(recorder.call())
        if best_pcr_auc <= pcr_auc:
            best_pcr_auc = pcr_auc
            with open(os.path.join(dir_checkpoint, 'log_poe_pcr.csv'), 'a+') as f:
                f.write('{},{},{}\n'.format(epoch+1, train_losses, pcr_auc, followup_cindex))
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_pcr_best.pth'))

        if best_followup_cindex <= followup_cindex:
            best_followup_cindex = followup_cindex
            with open(os.path.join(dir_checkpoint, 'log_poe_followup.csv'), 'a+') as f:
                f.write('{},{},{}\n'.format(epoch+1, train_losses, pcr_auc, followup_cindex))
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_followup_best.pth'))


def get_args():
    parser = argparse.ArgumentParser(description='Train PoE model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config/config.yaml',
                        help='config file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('-f', '--fold', dest='fold', type=int, default=0,
                        help='fold')
    
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['args'] = {
        'fold': args.fold,
    }

    dir_checkpoint = os.path.join(config['vae_poe']['ckpt'], str(args.fold))
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = os.path.join(config['vae_poe']['vis'], str(args.fold))
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = BreastVAE(in_dim=config['vae_poe']['in_dim'], regimen_dim=5, hidden_dim=256, latent_dim=3)
    net.to(device=device)
    ckpt_path = os.path.join(config['vae_pretrain']['ckpt'], 'ckpt_best.pth')
    net = load_weights(net, ckpt_path, device=device)
    
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