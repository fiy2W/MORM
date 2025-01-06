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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from model.breastVAE import BreastVAE
from dataloader.loader_all import Dataset_breast
from loss.contrastive_loss import SupConLoss
from utils.record import Plotter, Recorder
from utils.inference import load_weights


def train(args, net, device):
    train_data = Dataset_breast(args, valid_fold=-1, mode='train')
    
    n_train = len(train_data)#len(dataset) - n_val

    label_counts = Counter(train_data.group_list)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    weights = [class_weights[train_data.group_list[i]] for i in range(n_train)]
    weights = torch.DoubleTensor(weights)

    train_loader = DataLoader(train_data, batch_size=16, num_workers=2, pin_memory=True, sampler=WeightedRandomSampler(weights, n_train))
    
    n_copy = 1
    batch_size=args['vae_pretrain']['batchsize']
    epochs = 100
    lr = np.float32(args['vae_pretrain']['lr'])
    dir_visualize = os.path.join(config['vae_pretrain']['vis'])
    dir_checkpoint = os.path.join(config['vae_pretrain']['ckpt'])
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)

    contrast = SupConLoss()
    
    recorder = Recorder(['loss'])
    plotter = Plotter(dir_visualize, keys1=['loss'])
    
    with open(os.path.join(dir_checkpoint, 'log_poe_clip.csv'), 'w') as f:
        f.write('epoch,loss\n')

    total_step = 0
    best_pcr_auc = 100
    for epoch in range(epochs):
        if epoch > args['vae_pretrain']['early_stop']:
            break
        net.train()
        
        train_losses = []
        with tqdm(total=n_train*n_copy, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                image = batch['image'].to(device=device, dtype=torch.float32)
                seg = batch['seg'].to(device=device, dtype=torch.float32)
                meta_ori = batch['meta'].to(device=device, dtype=torch.float32)
                input_regimen = batch['input_regimen'].to(device=device, dtype=torch.float32)
                radiology = batch['radiology']
                pathology = batch['pathology']
                record = batch['record']

                for _ in range(n_copy):
                    f_tab, f_img, f_txt = net.encoder(meta_ori, image, seg, radiology, pathology, record)

                    output = net(input_regimen, meta_ori, f_img, f_txt, sample=True)

                    f_tab = f_tab / f_tab.norm(dim=1, keepdim=True)
                    f_img = f_img / f_img.norm(dim=1, keepdim=True)
                    f_txt = f_txt / f_txt.norm(dim=1, keepdim=True)
                    
                    loss_clip = contrast(torch.stack([f_img, f_txt], dim=1)) + contrast(torch.stack([f_txt, f_img], dim=1)) + \
                        contrast(torch.stack([f_img, f_tab], dim=1)) + contrast(torch.stack([f_tab, f_img], dim=1)) + \
                        contrast(torch.stack([f_tab, f_txt], dim=1)) + contrast(torch.stack([f_txt, f_tab], dim=1))

                    loss_kl = output['kl_tab'] + output['kl_img'] + output['kl_txt']
                    
                    loss = loss_clip + loss_kl
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                
                    train_losses.append(loss.item())
                    pbar.set_postfix(**{'clip': loss_clip.item(), 'kl': loss_kl.item()})
                    pbar.update(meta_ori.shape[0])
                
                
            total_step += 1
        
        train_losses = np.mean(train_losses)


        recorder.update({'loss': train_losses})
        plotter.send(recorder.call())
        if best_pcr_auc > train_losses:
            best_pcr_auc = train_losses
            with open(os.path.join(dir_checkpoint, 'log_poe_clip.csv'), 'a+') as f:
                f.write('{},{}\n'.format(epoch+1, train_losses))
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))


def get_args():
    parser = argparse.ArgumentParser(description='Pretrain clip model',
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

    dir_checkpoint = os.path.join(config['vae_pretrain']['ckpt'], str(args.fold))
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = os.path.join(config['vae_pretrain']['vis'], str(args.fold))
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = BreastVAE(in_dim=config['vae_pretrain']['in_dim'], regimen_dim=5, hidden_dim=256, latent_dim=3)
    net.to(device=device)
    ckpt_path = os.path.join(config['vae_pretrain']['ckpt'], 'ckpt_best.pth')
    if os.path.exists(ckpt_path):
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