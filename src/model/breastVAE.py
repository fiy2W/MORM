import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

import numpy as np
import random

from transformers import RobertaModel

from model.resnet import resnet10


class BreastVAE(nn.Module):
    def __init__(self, in_dim, regimen_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # tab
        self.tab_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tab_phi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU())
        self.tab_p_quant = nn.Sequential(nn.Linear(hidden_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2))

        # img
        img_encoder = resnet10(cin=1)
        model_dict = img_encoder.state_dict()
        load_dict = torch.load("ckpt/resnet_pretrain/resnet_10_23dataset.pth")
        load_dict = {k.replace('module.', ''): v for k, v in load_dict['state_dict'].items() if 'conv_seg' not in k and k.replace('module.', '') in model_dict}
        model_dict.update(load_dict)
        img_encoder.load_state_dict(model_dict)
        img_encoder.conv1 = nn.Conv3d(
            2,
            64,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False)
        
        self.img_encoder = nn.Sequential(
            img_encoder,
            nn.BatchNorm3d(64*8),
            AttentionPool3d(6*6*3, 64*8, 8, hidden_dim),
        )
        self.img_phi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5))
        self.img_p_quant = nn.Sequential(nn.Linear(hidden_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2))

        # txt
        self.txt_bert = RobertaModel.from_pretrained('ckpt/radiobert_en_nl_30522', add_pooling_layer=False, output_attentions=True)
        self.txt_fc_merge = nn.Sequential(nn.Linear(768*3, hidden_dim))
        self.txt_phi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5))
        self.txt_p_quant = nn.Sequential(nn.Linear(hidden_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2))

        self.outcome_predictor = OutcomePredictor(latent_dim, hidden_dim=hidden_dim, regimen_dim=regimen_dim, in_dim=in_dim)
        
        self.fc_factor_pcr_poe = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        self.fc_factor_hazard_poe = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        
    def ProductOfExperts(self, mu_set, logvar_set, eps=1e-9):
        mu_tmp = 0.
        var_tmp = 1.
        for mu, logvar in zip(mu_set, logvar_set):
            tmp = 1./(torch.exp(logvar) + eps)
            var_tmp += tmp
            mu_tmp += mu*tmp
        poe_var = 1./var_tmp
        poe_mu = mu_tmp/var_tmp
        poe_logvar = torch.log(poe_var)
        return poe_mu, poe_logvar
    
    def reparameterize(self, mu, logvar, sample=True):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if sample:
            z = mu + eps*std
        else:
            z = mu
        return z
    
    def kl_loss(self, mu, logvar):
        return 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1.0 - logvar, dim=[1,]).mean()
    
    def cox_loss(self, risk_pred, y, e):
        time_value = torch.squeeze(y)
        event = torch.squeeze(e).type(torch.bool)
        score = torch.squeeze(risk_pred)

        ix = torch.where(event)[0]

        sel_time = time_value[ix]
        sel_mat = (sel_time.unsqueeze(1).expand(1, sel_time.size()[0],
                                                time_value.size()[0]).squeeze() <= time_value).float()

        p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))

        loss = -torch.mean(p_lik)

        return loss if len(ix)>0 else torch.tensor(0., device=y.device)
    
    def forward(self, regimen, x_tab, f_img=None, f_txt=None, sample=True):
        f_tab = self.tab_encoder(x_tab)
        f_phi_tab = self.tab_phi(f_tab)
        p_tab = self.tab_p_quant(f_phi_tab)
        mu_tab, logvar_tab = torch.chunk(p_tab, 2, dim=1)
        z_tab = self.reparameterize(mu_tab, logvar_tab, sample=sample)
        yout_tab = self.outcome_predictor(z_tab, regimen)

        output = {
            'pcr_tab': yout_tab[0], 'hazard_tab': yout_tab[1], 'recon_tab': yout_tab[2],
            'mu_tab': mu_tab, 'logvar_tab': logvar_tab, 'kl_tab': self.kl_loss(mu_tab, logvar_tab),
        }

        if f_img is not None:
            f_phi_img = self.img_phi(f_img)
            p_img = self.img_p_quant(f_phi_img)
            mu_img, logvar_img = torch.chunk(p_img, 2, dim=1)
            z_img = self.reparameterize(mu_img, logvar_img, sample=sample)
            yout_img = self.outcome_predictor(z_img, regimen)
            
            mu_poe_img, logvar_poe_img = self.ProductOfExperts([mu_tab.detach(), mu_img], [logvar_tab.detach(), logvar_img])
            z_poe_img = self.reparameterize(mu_poe_img, logvar_poe_img, sample=sample)
            yout_poe_img = self.outcome_predictor(z_poe_img, regimen)

            z_poe_img0 = self.reparameterize(mu_poe_img, logvar_poe_img, sample=False).detach()
            factor_pcr_poe_img = self.fc_factor_pcr_poe(z_poe_img0)
            factor_hazard_poe_img = self.fc_factor_hazard_poe(z_poe_img0)

            output['pcr_img'] = yout_img[0]
            output['hazard_img'] = yout_img[1]
            output['recon_img'] = yout_img[2]
            output['mu_img'] = mu_img
            output['logvar_img'] = logvar_img
            output['kl_img'] = self.kl_loss(mu_img, logvar_img)
            output['pcr_poe_img0'] = yout_poe_img[0]
            output['hazard_poe_img0'] = yout_poe_img[1]
            output['recon_poe_img0'] = yout_poe_img[2]
            output['pcr_poe_img'] = yout_poe_img[0].detach()*factor_pcr_poe_img+yout_tab[0].detach()*(1-factor_pcr_poe_img)
            output['hazard_poe_img'] = yout_poe_img[1].detach()*factor_hazard_poe_img+yout_tab[1].detach()*(1-factor_hazard_poe_img)
            output['mu_poe_img'] = mu_poe_img
            output['logvar_poe_img'] = logvar_poe_img
            output['kl_poe_img'] = self.kl_loss(mu_poe_img, logvar_poe_img)
            output['factor_pcr_poe_img'] = factor_pcr_poe_img
            output['factor_hazard_poe_img'] = factor_hazard_poe_img
        
        if f_txt is not None:
            f_phi_txt = self.txt_phi(f_txt)
            p_txt = self.txt_p_quant(f_phi_txt)
            mu_txt, logvar_txt = torch.chunk(p_txt, 2, dim=1)
            z_txt = self.reparameterize(mu_txt, logvar_txt, sample=sample)
            yout_txt = self.outcome_predictor(z_txt, regimen)

            mu_poe_txt, logvar_poe_txt = self.ProductOfExperts([mu_tab.detach(), mu_txt], [logvar_tab.detach(), logvar_txt])
            z_poe_txt = self.reparameterize(mu_poe_txt, logvar_poe_txt, sample=sample)
            yout_poe_txt = self.outcome_predictor(z_poe_txt, regimen)

            z_poe_txt0 = self.reparameterize(mu_poe_txt, logvar_poe_txt, sample=False).detach()
            factor_pcr_poe_txt = self.fc_factor_pcr_poe(z_poe_txt0)
            factor_hazard_poe_txt = self.fc_factor_hazard_poe(z_poe_txt0)

            output['pcr_txt'] = yout_txt[0]
            output['hazard_txt'] = yout_txt[1]
            output['recon_txt'] = yout_txt[2]
            output['mu_txt'] = mu_txt
            output['logvar_txt'] = logvar_txt
            output['kl_txt'] = self.kl_loss(mu_txt, logvar_txt)
            output['pcr_poe_txt0'] = yout_poe_txt[0]
            output['hazard_poe_txt0'] = yout_poe_txt[1]
            output['recon_poe_txt0'] = yout_poe_txt[2]
            output['pcr_poe_txt'] = yout_poe_txt[0].detach()*factor_pcr_poe_txt+yout_tab[0].detach()*(1-factor_pcr_poe_txt)
            output['hazard_poe_txt'] = yout_poe_txt[1].detach()*factor_hazard_poe_txt+yout_tab[1].detach()*(1-factor_hazard_poe_txt)
            output['mu_poe_txt'] = mu_poe_txt
            output['logvar_poe_txt'] = logvar_poe_txt
            output['kl_poe_txt'] = self.kl_loss(mu_poe_txt, logvar_poe_txt)
            output['factor_pcr_poe_txt'] = factor_pcr_poe_txt
            output['factor_hazard_poe_txt'] = factor_hazard_poe_txt
        
        if f_img is not None and f_txt is not None:
            mu_poe_all, logvar_poe_all = self.ProductOfExperts([mu_tab.detach(), mu_img, mu_txt], [logvar_tab.detach(), logvar_img, logvar_txt])
            z_poe_all = self.reparameterize(mu_poe_all, logvar_poe_all, sample=sample)
            yout_poe_all = self.outcome_predictor(z_poe_all, regimen)

            z_poe_all0 = self.reparameterize(mu_poe_all, logvar_poe_all, sample=False).detach()
            factor_pcr_poe_all = self.fc_factor_pcr_poe(z_poe_all0)
            factor_hazard_poe_all = self.fc_factor_hazard_poe(z_poe_all0)

            output['pcr_poe_all0'] = yout_poe_all[0]
            output['hazard_poe_all0'] = yout_poe_all[1]
            output['recon_poe_all0'] = yout_poe_all[2]
            output['pcr_poe_all'] = yout_poe_all[0].detach()*factor_pcr_poe_all+yout_tab[0].detach()*(1-factor_pcr_poe_all)
            output['hazard_poe_all'] = yout_poe_all[1].detach()*factor_hazard_poe_all+yout_tab[1].detach()*(1-factor_hazard_poe_all)
            output['mu_poe_all'] = mu_poe_all
            output['logvar_poe_all'] = logvar_poe_all
            output['kl_poe_all'] = self.kl_loss(mu_poe_all, logvar_poe_all)
            output['factor_pcr_poe_all'] = factor_pcr_poe_all
            output['factor_hazard_poe_all'] = factor_hazard_poe_all
        
        return output
    
    def txt_encoder(self, radiology, pathology, record, device):
        f_rad = self.txt_bert(input_ids=radiology['input_ids'][:,0].to(device), attention_mask=radiology['attention_mask'][:,0].to(device), return_dict=False)
        f_pat = self.txt_bert(input_ids=pathology['input_ids'][:,0].to(device), attention_mask=pathology['attention_mask'][:,0].to(device), return_dict=False)
        f_rec = self.txt_bert(input_ids=record['input_ids'][:,0].to(device), attention_mask=record['attention_mask'][:,0].to(device), return_dict=False)
        f_rad = f_rad[0][:,0]
        f_pat = f_pat[0][:,0]
        f_rec = f_rec[0][:,0]
        f_txt = self.txt_fc_merge(torch.cat([f_rad, f_pat, f_rec], dim=1))
        return f_txt
    
    def encoder(self, x_tab, x_img, md, radiology, pathology, record):
        f_tab = self.tab_encoder(x_tab)
        f_img = self.img_encoder(torch.cat([x_img, md], dim=1))
        f_txt = self.txt_encoder(radiology, pathology, record, x_tab.device)
        return f_tab, f_img, f_txt

    def encoder2(self, x_tab, x_img, md):
        f_tab = self.tab_encoder(x_tab)
        f_img = self.img_encoder(torch.cat([x_img, md], dim=1))
        return f_tab, f_img


class OutcomePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, regimen_dim, in_dim):
        super().__init__()
        self.hyper = nn.Sequential(
            nn.Linear(regimen_dim+latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_pcr = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )
        self.fc_hazard = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z, regimen):
        f = self.hyper(torch.cat([z, regimen], dim=1))
        pcr = self.fc_pcr(f)
        hazard = self.fc_hazard(f)
        return pcr, hazard


class AttentionPool3d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)