import torch
import numpy as np
import pandas as pd
import warnings
import os
import sys

from scipy.stats import norm, gamma
from scipy.optimize import fsolve
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import qth_survival_time

from sklearn.metrics import auc
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils.gumbel_min import Gumbelmin
from datasets.simulate import p_z_given_xy

def loss_fn(recon_y, y,delta, sigma, q_mean, q_log_var,primative = "normal",beta = 1):
    """
    Loss function for CIVIL-CVAE model 
    Input:
        recon_y (Tensor): Decoded mean parameter
        sigma (Tensor): Decoded log scale parameter
        y (Tensor): Response variable 
        delta (Tensor): Censoring indicator for each response (event = 1) 
        q_mean, q_log_var: Encoder mean and log_var
    Args:
        primative (String): type of primative distribution parameterized by recon_y as mean and sigma.exp() as scale 
        beta (Scalar): Censoring likelihood penality hyperparameter (beta = 1 means no penality) 
    Output:
        NLK: negative loglikelihood (reconstruction error)
        KLD: KL divergence between encoder and standard gaussian prior
    """
    loc = recon_y
    scale = (sigma* torch.ones_like(recon_y,device = recon_y.device)).exp()
    if primative != None:
        if primative == "gumbel":
            dist = Gumbelmin(loc,scale)
            NLK = -1*(dist.log_prob(y)[delta.squeeze()==True,:].sum() +beta* dist.logsf(y)[delta.squeeze()==False,:].sum()) # do not use 1-cdf for numeric stability
        elif primative == "normal":
            dist = torch.distributions.normal.Normal(loc,scale)
            NLK = -1*(dist.log_prob(y)[delta.squeeze()==True,:].sum() +beta*((1+1e-6-dist.cdf(y))[delta.squeeze()==False,:].log().sum())) # do not use 1-cdf for numeric stability
        else:
            raise NotImplementedError("primative distribution not implemented")
    
    KLD = -0.5 * torch.sum(1 + q_log_var - q_mean.pow(2) - q_log_var.exp())
    return [NLK + KLD, NLK,KLD]


def KL_divergence(mu1,log_var1,mu2,log_var2):
    """
    Compute the KL divergence between two multivariate Gaussian distributions.
    KL(N(mu1, cov1) || N(mu2, cov2))
    Args:
        mu1 (numpy.ndarray): Mean vector of the first Gaussian (shape: d,)
        log_var1 (numpy.ndarray): logarithm of the diagonal elements of cov1 (shape: d ,)
        mu2 (numpy.ndarray): Mean vector of the second Gaussian (shape: d,)
        log_var2 (numpy.ndarray): logarithm of the diagonal elements of cov2 (shape: d ,)
    Returns:
        float: The KL divergence.
    """
    KLD = -0.5* torch.sum(1+log_var1-log_var2-(mu1-mu2).pow(2)/(log_var2.exp())-(log_var1.exp()/log_var2.exp()))
    return KLD
    
def iwae_loss_fn(log_likelihood, log_w , mode='fast'):
    # loss computation (several ways possible from https://borea17.github.io/paper_summaries/iwae/)
    if mode == 'original':
        ####################### ORIGINAL IMPLEMENTAION #######################
        # numerical stability (found in original implementation)
        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        # compute normalized importance weights (no gradient)
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        # compute loss (negative IWAE objective)
        loss = -(w_tilde * log_w).sum(1).mean()
    elif mode == 'normalized weights':
        ######################## LOG-NORMALIZED TRICK ########################
        # copmute normalized importance weights (no gradient)
        log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w_tilde = log_w_tilde.exp().detach()
        # compute loss (negative IWAE objective)
        loss = -(w_tilde * log_w).sum(1).mean()
    elif mode == 'fast':
        ########################## SIMPLE AND FAST ###########################
        loss = -log_likelihood
    else:
        raise NotImplementedError("iwae loss mode not implemented")
    return loss
    

def compute_kld_gmm(x,y,loc_q=[0,0],scale_q= [1,1], num_points=1000):
    """Monte Carlo Estimation of the KL divergence between a mixture of two Gaussians and a single Gaussian with loc_q and scale_q."""
    # get the true posterior mean and std
    kls =[]
    for i in range(num_points):
        temp_z, pdf  = p_z_given_xy(x,y,ztype = "mixture",weight=0.5,label=False,prob = True)
        log_p = np.log(pdf)
        log_q =  np.log(np.maximum(norm.pdf(temp_z,loc=loc_q,scale= scale_q), 1e-10))    # Avoid division by zero or log(0)
        kls.append(log_p/log_q)

    return np.mean(kls)


