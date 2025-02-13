import torch
import numpy as np
import torch.nn as nn
import torch.distributions as td
import pandas as pd
from scipy.stats import norm
from pycox.evaluation import EvalSurv
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from model.base_model import Encoder, Decoder,Sigma
from utils.gumbel_min import Gumbelmin


class CVAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_dim, px, decoder_layer_sizes,sigma_learning = "joint"):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_dim) == int
        assert type(decoder_layer_sizes) == list
        self.latent_dim = latent_dim
        # dimension of x
        self.px = px
        # Encoder does not take censoring information
        self.encoder = Encoder(encoder_layer_sizes, latent_dim)
        self.decoder = Decoder(decoder_layer_sizes,input_size=latent_dim+px)
        self.primative = "gumbel"
        if sigma_learning== "joint":
            self.log_sigma =  nn.Parameter(torch.tensor(0,dtype=torch.float32), requires_grad=True) #parameter to be learned
        elif sigma_learning == "fixed":
            self.log_sigma = torch.tensor(0,dtype=torch.float32) #fixed at 0
        elif sigma_learning == "xdependent":
            self.sigmanet = Sigma([32,16,1],input_size=px)
        elif sigma_learning == "coordiante":
            self.log_sigma = torch.tensor(0,dtype=torch.float32) #intialized at 0
        else:
            raise(RuntimeError("the type of sigma learning not specified"))   
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x_train, y_train):
        # do not take censoring information for encoder
        mean, log_var = self.encoder(x_train, y_train)  # [batch_size, latent_dim] 
        z = self.reparameterize(mean, log_var)  # [batch_size, nrep] 
        recon = self.decoder(z,x_train)  
        return recon, mean, log_var #mu_y, mu_z, log_sigma_z
    
    def validate(self,x_val,y_val,censor_val, y_train,censor_train,event_quantile = True):
        if isinstance(y_val,pd.DataFrame):
            raise RuntimeError('pd.dataframe used, please parse')
        if isinstance(y_val,np.ndarray):
            x_val = torch.from_numpy(x_val).to(dtype=torch.FloatTensor, device = y_train.device)
            y_val = torch.from_numpy(y_val).to(dtype=torch.FloatTensor, device = y_train.device)
            y_censor = torch.from_numpy(censor_val).to(dtype=torch.FloatTensor, device = y_train.device)        
        z = torch.randn(size=[len(x_val),self.latent_dim],device = y_train.device)
        loc_val = self.decoder(z,x_val).detach().cpu()
        if event_quantile ==True:
            train_times = np.quantile(y_train[censor_train==1], np.linspace(0.1, 1, 10))
        else:
            train_times = np.unique(y_train.flatten().cpu())
        if self.primative == "normal":
            predictions= norm.logsf(train_times.numpy(), loc=loc_val.numpy(), scale=1)
        elif self.primative == "gumbel":
            gumbel = Gumbelmin(loc_val,torch.tensor([1],dtype=torch.float32))
            predictions= gumbel.logsf(torch.from_numpy(train_times))
        else: raise RuntimeError('primative distribution not speficied')
        ctd = EvalSurv(pd.DataFrame(predictions.T, index=train_times), y_val.flatten().cpu().numpy(), censor_val.flatten().cpu().numpy(), censor_surv='km')
        return ctd.concordance_td()
    
    def test(self,x_test,y_test,censor_test, y_train,censor_train,event_quantile = True):
        if isinstance(y_test,pd.DataFrame):
            raise RuntimeError('pd.dataframe used, please parse')
        if isinstance(y_test,np.ndarray):
            x_test = torch.from_numpy(x_test).to(dtype=torch.FloatTensor, device = y_train.device)
            y_test = torch.from_numpy(y_test).to(dtype=torch.FloatTensor, device = y_train.device)
            censor_test = torch.from_numpy(censor_test).to(dtype=torch.FloatTensor, device = y_train.device)        
        z = torch.randn(size=[len(x_test),self.latent_dim],device = y_train.device)
        loc_test = self.decoder(z,x_test).detach().cpu().numpy()
        if event_quantile ==True:
            train_times = np.quantile(y_train[censor_train==1], np.linspace(0.1, 1, 10))
        else:
            train_times = np.unique(y_train.flatten().cpu().numpy())
        predictions= norm.logsf(train_times, loc=loc_test, scale=1)  
        ctd = EvalSurv(pd.DataFrame(predictions.T, index=train_times),y_test.flatten().cpu().numpy(), censor_test.flatten().cpu().numpy(), censor_surv='km')
        return ctd.concordance_td()    