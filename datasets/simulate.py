#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simulating dataset that has a customized posterior function, using gibbs sampling and independent censoring
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os
import sys
import warnings

# Suppress only DeprecationWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils.gumbel_min import Gumbelmin

from scipy.stats import norm, multivariate_normal
#------------------------------

def censoring(u,clevel):
    """
    Conditional independent censoring for simulation dataset
    """
    level_dict = { # controls the percentage of censoring in the dataset
        "no_censor": 16,
        "very_low": 16,
        "low":8.5,
        "medium": 5.5,
        "high":0,
        "all_censor":16
    }
    rng = np.random.default_rng() 
    c_logsigma = 2
    c_mu = level_dict[clevel]
    c = c_mu+ rng.normal(0,np.exp(c_logsigma)) 
    if clevel == "no_censor" or clevel == "all_censor": # all/no censoring
        boolean_mapping = {"no_censor": 1, "all_censor": 0}
        event_ind = np.zeros_like(u)+ boolean_mapping[clevel]
    elif clevel in level_dict: # no censoring
        event_ind = c > u
    else:
        raise(RuntimeError("level of censoring not specified"))

    y = u*event_ind + (1-event_ind)*c
        

    return y, event_ind


def p_xydelta_given_z(z,log_sigma=0,ytype="normal",clevel="high"): 
    """
    Input:
        array z: latent variable 2d
    Args:
        scalr sigma: scale for noise in p(y|x,z)
        ytype: type of distribution of noise, can be gumbel or normal
    Output:
        ndarray x,y: one sample from distribution p(x|z) and p(y|x,z)
        ndarray u: uncensored survival time
        ndarray llk: partial log-likelihood for observing x,y,delta 
        boolean delta: if censoring < u
    """
    sigma= np.exp(log_sigma)
    rng = np.random.default_rng() 
    x = rng.normal(1,1) # x and z independent
    mu = x*z[0]+1*z[1]
    if ytype == "normal":
        u = mu+ rng.normal(0,sigma) 
        rv = norm(mu,sigma)
    elif ytype =="gumbel":
        dist = torch.distributions.normal.Normal(torch.tensor(mu),torch.tensor(sigma))
        u = mu+ dist.sample([1]).cpu().detach().numpy().astype(np.float64)[0]
    else:
        raise(RuntimeError("residual not implemented"))
    y,ind = censoring(u,clevel=clevel)
    if ytype == "normal":
        surv_llk = ind*rv.logpdf(u)+ (1-ind)*(rv.logsf(u))
    elif ytype =="gumbel":
        surv_llk = ind*dist.log_prob(torch.tensor(mu)).cpu().detach().numpy().astype(np.float64)+ (1-ind)*dist.log_prob(torch.tensor(mu)).cpu().detach().numpy().astype(np.float64)
    else:
        raise(RuntimeError("residual not implemented"))

    return x,y, ind, u, surv_llk



def p_z_given_xydelta(x,y,delta,prob = False):
    """
    True posterior distribution p(z|x,y,delta) to sample z and 
    Input:
        ndarray x: covariates
        ndarra y: survival time
        boolean delta: event indicator
    Args:
        ztype: type of posterior distribution. can be mixture of gaussian, isotropic gaussian or correlated gaussian.
        weight: covariance of correlated gaussian or weight of mixture of gaussian
        label: if True, return mixture gaussian with component label 0 or 1
        prob: if True, return the probability of sampling the output z from p(z|x,y)
    Output:
        ndarray z: one sample z from distribution p(z|x,y,delta)
        ndarray mu,cov: mean and covariance for posterior of distribution p(z|x,y,delta) 2d
    """
    rng = np.random.default_rng() 
    multiplier = int(delta)*2-1
    #if delta = 1, positive mean, if delta = 0, negative mean.
    signless = 3/(1 + np.exp(y+ x))
    mu = (multiplier*signless,multiplier*signless) 
    cov = [[1,0],[0,1]]
    z = rng.multivariate_normal(mu, cov)
    
    if prob == True:
        # return the probability of getting the current z
        rv = multivariate_normal(mean=mu, cov=cov)
        logpdf = rv.logpdf(z)
        return z,mu,logpdf
    else:
        return z,mu


def gibbs_sampler(num_of_data,sigma=1.0,ytype= "gumbel",clevel= "high"):
    """
    A Gibbs sampler for a joint distribution of triplets {x,y,z} where z is the latent variable.
    p(z|x,y) in particular can be a 2-component mixture of 2d gaussian.
    p(y,x|z) = p(y|x,z)p(x|z) is 1d normal and bernoulli
    (without the exact distribution of x,y,z)
    sigma: the noise variance of p(y|x,z)
    weight: the weight of two Gaussian p(z|y,x)
    Cesoring on y is independent of x,y,z
    censor_rate: the percentage of censoring rate of y
    """
    burnin = 25000
    # observable
    x_samples = [] 
    y_samples = []
    delta_samples = []

    # oracle information for comparison experiment
    z_samples =np.array([], dtype=np.float64).reshape(0,2)
    u_samples = []
    mus = np.array([], dtype=np.float64).reshape(0,2)
    llks = []
    postprobs = []
    #starting z value
    z = np.array([0,0])

    for _ in range(num_of_data):
        x,y,delta,u,llk =  p_xydelta_given_z(z,sigma,ytype,clevel)
        z,mu,poster_p = p_z_given_xydelta(x,y,delta,prob = True)
        if _ >= burnin:
            x_samples= np.append(x_samples,x)
            y_samples= np.append(y_samples,y)
            delta_samples= np.append(delta_samples,delta)
            z_samples= np.vstack((z_samples,z))
            mus= np.vstack((mus,mu))
            u_samples= np.append(u_samples,u)
            llks= np.append(llks,llk)
            postprobs= np.append(postprobs,poster_p)
    #showing the gibbs sampled 
    return x_samples,y_samples,delta_samples,z_samples,u_samples,mus,llks,postprobs # [num_of_data,2],[num_of_data,2] array float32

def save_object(obj, filename):
    file_path = os.path.join(current_dir, filename)
    with open(file_path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        print("simulation completed, data saved")

# gibbs_sampler usage example
size = 35000
sigma = 1
ytype = "normal"

clist = ["no_censor","very_low","low","medium","high","all_censor"]
#clist = []
#------------when simulate new dataset, uncomment the following
for clevel in clist:
    x_samples,y_samples,delta_samples,z_samples,u_samples,mus,llks,postprobs = gibbs_sampler(size,sigma,ytype,clevel)
    if clevel != "all_censor":
        print(f"simulation complete, event rate:{np.mean(delta_samples)*100:.1f}%, with mean {np.mean(y_samples[delta_samples==1]):.2f}, median {np.median(y_samples[delta_samples==1]):.2f}, with min {np.min(y_samples[delta_samples==1]):.2f}, with max {np.max(y_samples[delta_samples==1]):.2f}, with censored mean {np.mean(y_samples[delta_samples==0]):.2f}")
    else:
        print(f"simulation complete, event rate:{np.mean(delta_samples)*100:.1f}%, with mean {np.mean(y_samples[delta_samples==0]):.2f}")
    filename = "dataset_"+str(clevel)+".pkl"
    save_object([x_samples,y_samples,delta_samples,z_samples,u_samples,mus,llks,postprobs],filename)

"""some descriptive statistics:
    u 10000x1: max = 34.6136, min = -32.2747
    x 10000x1: max 3.01 min -0.98
    y 10000x1: max 21.53 min -32.274
    z 10000x2: max 6.44 min -6.64
    mu 10000x2: max 2.999999 min -2.99999
"""

# #------------
# def read_object(filename):
#     file_path = os.path.join(current_dir, filename)
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     return data

# x_samples,y_samples,delta_samples,z_samples,u_samples,mus,llks,postprobs = read_object('dataset.pkl')
# colors = {False:'red', True:'blue'}
# print(x_samples)
# print(np.mean(delta_samples))
# # # censoring example
# # print(np.mean(ind))
# # print(y.reshape(len(y),1))

# # #################################################################################
# # marginal distribution of x,y
# plt.scatter(x_samples,y_samples,c=np.vectorize(colors.get)(delta_samples))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
# counts,ybins,xbins,image = plt.hist2d(x_samples,y_samples,bins=50,norm=mcolors.PowerNorm(0.3))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# # marginal distribution of z
# scatter = plt.scatter(z_samples[:,0],z_samples[:,1], c=np.vectorize(colors.get)(delta_samples))
# plt.legend(handles=scatter.legend_elements()[0], labels=["Censored","Event"] )
# plt.xlabel("z1")
# plt.ylabel("z2")
# plt.colorbar()
# plt.show()

# # z |x,y for all events (x,y)
# scatter = plt.scatter(z_samples[delta_samples==1,0],z_samples[delta_samples==1,1],c="blue")
# plt.xlabel("z1")
# plt.ylabel("z2")
# plt.show()

# # z |x,y for all censored (x,y)
# scatter = plt.scatter(z_samples[delta_samples==0,0],z_samples[delta_samples==0,1],c="red")
# plt.xlabel("z1")
# plt.ylabel("z2")
# plt.show()

# # # posterior of z given x,y = 0,0
# # condi_z_label= np.array([p_z_given_xy(0,0,ztype= "mixture",weight=0.5,label=True) for i in range(20000)]) 
# # scatter = plt.scatter(condi_z_label[:,0],condi_z_label[:,1],c=condi_z_label[:,2],label=condi_z_label[:,2])
# # plt.legend(handles=scatter.legend_elements()[0], labels=["Part 1","Part 2"])
# # plt.show()
# # plt.hist2d(condi_z_label[:,0],condi_z_label[:,1],label=condi_z_label[:,2],bins=200,norm=mcolors.PowerNorm(0.3))
# # plt.legend(handles=scatter.legend_elements()[0], labels=["Part 1","Part 2"])
# # plt.colorbar()
# # plt.show()


# # # marginal distribution of x,censored_y
# # plt.scatter(x,y,c=ind,label=ind)
# # plt.xlabel("x")
# # plt.ylabel("y")
# # plt.show()
# # counts,ybins,xbins,image = plt.hist2d(x,y,bins=100,norm=mcolors.PowerNorm(0.3))
# # plt.xlabel("x")
# # plt.ylabel("y")
# # plt.show()