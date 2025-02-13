import torch
import numpy as np
import torch.nn as nn
import torch.distributions as td
import pandas as pd
from scipy.stats import norm
from pycox.evaluation import EvalSurv
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from model.cd_cvae import CDCVAE
from utils.gumbel_min import Gumbelmin
from utils.metrics import iwae_loss_fn
from utils.override_functions import survival_regression_metric_modified

class CDDIWAE(CDCVAE):
    """
    CDVI_IW: censor-dependent conditional CVAE for survival analysis with importance sampling
    """
    def __init__(self, K = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = K
    
    def compute_marginal_log_likelihood(self, x,y,delta, k=None, primative = "normal",beta=1):
        """computes the marginal log-likelihood in which the sampling
        distribution is exchanged to q_{\phi} (z|x),
        this function can also be used for the IWAE loss computation
        Args:
            k: number of importance weights
            primative: type of distribution of p(y|x,z)
            beta: censored likelihood penality (1 means no penalty)
        Returns:
            log_marginal_likelihood (torch tensor): scalar
            log_w (torch tensor): unnormalized log importance weights [batch, k]
        """        
        # get mean and log-var from encoder [batch,latent_dim]
        mu, log_var  = self.encoder(x,y,delta) 
        # upsample mu, log-var with k reptitions [batch,k,latent_dim]
        mu_up = mu.unsqueeze(1).repeat(1, k, 1) 
        log_var_up = log_var.unsqueeze(1).repeat(1, k, 1)
        # upsampled z with reparameterize [batch,k,latent_dim]
        z_up = self.reparameterize(mu_up, log_var_up) 
        # decode the recon_y from z_up and x_up 
        x_up =  x.unsqueeze(1).repeat(1, k, 1)
        recon_y_up = self.decoder(z_up, x_up)
        # compute logarithmic unnormalized importance weights log_w [batch, k]
        y_up  = y.unsqueeze(1).repeat(1, k, 1)
        if primative == "normal":
            dist = torch.distributions.normal.Normal(recon_y_up,(self.log_sigma*torch.ones_like(y_up,device = y_up.device)).exp())
            log_p_y_g_xz = torch.zeros(size =[len(delta.squeeze()), k], device = y_up.device)
            log_p_y_g_xz[delta.squeeze()==True,:]  +=  dist.log_prob(y_up)[delta.squeeze()==True,:,:].sum(2) 
            log_p_y_g_xz[delta.squeeze()==False,:] += beta*((1+1e-6-dist.cdf(y_up))[delta.squeeze()==False,:,:].log().sum(2))
        elif primative == "gumbel":
            dist = Gumbelmin(recon_y_up,(self.log_sigma*torch.ones_like(y_up,device = y_up.device)).exp())
            # use self defined logsf function for numerical stability
            log_p_y_g_xz= dist.log_prob(y_up)[delta.squeeze()==True,:,:].sum(2) + beta* dist.logsf(y_up)[delta.squeeze()==False,:,:].sum(2)
        else: raise RuntimeError("distribution not specified")
        log_prior_z = td.normal.Normal(0, 1).log_prob(z_up).sum(2)
        log_q_z_g_xy = td.normal.Normal(mu_up,  (0.5*log_var_up).exp()).log_prob(z_up).sum(2)
        log_w = log_p_y_g_xz + log_prior_z - log_q_z_g_xy 
        # compute marginal log-likelihood with bias
        w = torch.exp(log_w)
        w_mean = torch.mean(w,1,keepdim=True)
        w_var = torch.sum((w-w_mean).pow(2),1)/(k-1)
        bias = w_var/(2*k*w_mean.pow(2).sum())
        avg_delta_method_likelihood = (torch.logsumexp(log_w, 1)-  np.log(k) +bias).mean()
        return log_w,  avg_delta_method_likelihood
    
    def fit(self,train_data, batch_size, num_epochs, learning_rate, format="pre",temperature=1,criterion="ctdpycox",early_stopper=True, patience=30):
        """Fit the Censor-dependent Variational Autoencoders (CD-CVAE) series of models to a given dataset and select the best module by cross validation. 
        It is a fully parametric approach and improves on the Accelerated Failure Time model by modelling the event time distribution 
        as a infinite size mixture over Weibull or Log-Normal distributions.

        References
        -----------
        [1] 

        Input:
            train_data : lists
                A list with pd.dataframes for training and validation dataset
        Args: 
            batch_size: int
                Learning is performed on mini-batches of input data. This parameter
                specifies the size of each mini-batch.
            num_epochs: int
                total number of epochs on the training data.
            learning_rate: float
                learning rate for 'Adam' optimizer with weight decay 0.01
            criterion : str, default='ctd', see survival_regression_metric_modified.py
                metrics for selecting best model in the training process
            temperature : float, default=1.0
                The value with which to rescale the survival function of censored observation, similar to a censored time truncation.
            early_stopper: bool
                if True, the fit process will be killed if the converging rate is slow (possible due to posterior collapse)
            patience: int
                number of epochs of waiting to break the fit process when no better metric is obtained in validation dataset

        Returns:
            Trained instance of the model
        """
        name =  __name__.split('.')[-1] if '.' in __name__ else __name__
        save_path = os.path.join(current_dir, "best_"+str(name)+"_cv.pth")
        x_tr,y_tr,x_val,y_val = train_data
        x_train = torch.tensor(x_tr.to_numpy(), dtype=torch.float32)
        y_train  = torch.tensor(y_tr["time"].to_numpy(), dtype=torch.float32).unsqueeze(-1) # numpy ndarray n x 1
        e_train = torch.tensor(y_tr["event"].to_numpy(), dtype=torch.float32)

        self = self.to("cpu")
        self.train()

        n = len(x_train)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)
        x_ind = np.arange(n)
        loss_list =[0]
        dict_list =[0] 
        metric_list = []
        best_metric = float('-inf')
        for epoch in range(num_epochs):
            np.random.shuffle(x_ind)
            train_loss = 0
            for i in range(0, n, batch_size):
                batch_id = x_ind[i:(i + batch_size)]
                x = x_train[batch_id, :]
                #print(str(x.shape)) #WHAS:[982, 5]
                y = y_train[batch_id, :]
                #print(str(y.shape)) #WHAS:[982, 1]
                e = e_train[batch_id]
                #print(str(e.shape)) #WHAS:[982]
                optimizer.zero_grad()
                log_w, avgllk = self.compute_marginal_log_likelihood(x,y,e,k=self.k)
                loss = iwae_loss_fn(avgllk, log_w, mode="original") 
                loss.backward()                
                optimizer.step()
                train_loss += loss.item()
            loss_list.append(train_loss/n)
            if len(dict_list) >= 10:
                dict_list.pop(0)
            dict_list.append(self.state_dict())
            # stopping criteria using validation
            if epoch % 2 == 0:
                # original codes for validation
                # x_valid = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
                # y_valid = torch.tensor(y_val["time"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
                # e_valid =  torch.tensor(y_val["event"].to_numpy(), dtype=torch.float32)
                # if self.sigma_learning == "xdependent":
                #     self.log_sigma = self.sigmanet(x_valid)
                # valid_ctd = self.validate(x_valid,y_valid,e_valid, y_train,e_train,False) # replace with metrics functions
                # print("Epoch {:02d}/{:02d}, sigma{:9.4f} Loss(ELBO) {:9.4f}, MSE {:4.4f},KLD {:4.4f},CTD_VAL {:9.3f}"
                # .format(epoch, num_epochs, self.log_sigma.exp().sum().item()/len(x_valid), train_loss/n,train_NLK/n, train_KL/n,valid_ctd))    
                # new codes for validation using self.prediction() taking validation pd.dataframe as input
                times = np.quantile(y_tr.time[y_tr["event"]==1], np.linspace(0.1, 1, 10))
                predictions = self.predict(x_val,times,format=format)
                metric = survival_regression_metric_modified(criterion,y_val,predictions,times,y_tr)
                metric = np.mean(metric) # if the metric is a list return its mean
                print("Epoch {:02d}/{:02d}, sigma{:9.4f} Loss(ELBO) {:9.4f}, Metric_VAL {:9.4f}"
                      .format(epoch, num_epochs, self.log_sigma.exp(), train_loss/n, metric))   
                # early stopper for cold restart
                metric_list.append(metric)
                if early_stopper == True and len(metric_list)> patience:
                    last_four = metric_list[-4:]
                    changes = [last_four[i] - last_four[i - 1] for i in range(1, len(last_four))]
                    if all(change == changes[0] for change in changes):
                        print("bad intialization/slow training detected. Restarting needed")
                        return None
                    
            # select the best model
            if metric > best_metric:
                best_metric = metric
                best_dict = dict_list[-1]
                torch.save(best_dict,save_path)
                best_epoch = epoch
            elif epoch-best_epoch>patience: # early stopping avoid overfitting and saving computation time
                break

        if best_metric is not None:
            print("Best model in terms of "+criterion+" found, and the metric on validation dataset is " +str(round(best_metric, 4)))
            self.load_state_dict(torch.load(save_path))
        
        self.eval()  # Switch the model to evaluation mode
        return self
    
    def tune(self,train_data, batch_size , num_epochs, learning_rate=0.0001, format="pre",temperature=1,criterion="ctd",early_stopper=True, patience=10, device="cpu"):
        """Fine tune the Censor-dependent Variational Autoencoders (CD-CVAE) series of models to a given dataset. 
        It is a fully parametric approach and improves on the Accelerated Failure Time model by modelling the event time distribution 
        as a infinite size mixture over Weibull or Log-Normal distributions.

        References
        -----------
        [1] 

        Input:
            train_data : lists
                A list with pd.dataframes for training and validation dataset
        Args: arguments needed for training the model
            criterion : str, default='ctd', see survival_regression_metric_modified.py
                metrics for selecting best model in the training process
            temperature : float, default=1.0
                The value with which to rescale the logits for the gate.
            batch_size: int, default=100
                Learning is performed on mini-batches of input data. This parameter
                specifies the size of each mini-batch.
            learning_rate: float, default=1e-3
                Learning rate for the 'Adam' optimizer.
            num_epochs: int
                Number of complete passes through the training data.

        Returns:
            Trained instance of the model
        """
        name =  __name__.split('.')[-1] if '.' in __name__ else __name__
        x_tr,y_tr,x_val,y_val = train_data
        train_x = pd.concat([x_tr,x_val]) 
        train_y = pd.concat([y_tr,y_val])
        x_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32)
        y_train  = torch.tensor(train_y["time"].to_numpy(), dtype=torch.float32).unsqueeze(-1) # numpy ndarray n x 1
        e_train = torch.tensor(train_y["event"].to_numpy(), dtype=torch.float32)

        self = self.to(device)
        self.train()

        n = len(x_train)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        x_ind = np.arange(n)
        loss_list =[0]
        KLD_list =[0]
        dict_list =[0] 
        best_metric = float('-inf')
        for epoch in range(num_epochs):
            np.random.shuffle(x_ind)
            train_loss,train_NLK,train_KL = 0,0,0
            for i in range(0, n, batch_size):
                batch_id = x_ind[i:(i + batch_size)]
                x = x_train[batch_id, :]
                #print(str(x.shape)) #WHAS:[982, 5]
                y = y_train[batch_id, :]
                #print(str(y.shape)) #WHAS:[982, 1]
                e = e_train[batch_id]
                #print(str(e.shape)) #WHAS:[982]
                optimizer.zero_grad()
                log_w, avgllk = self.compute_marginal_log_likelihood(x,y,e,k=self.k)
                loss = iwae_loss_fn(avgllk, log_w, mode="fast") 
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            loss_list.append(train_loss/n)
            KLD_list.append(train_KL/n)
            current_dict = self.state_dict()
            # stopping criteria using validation
            times = np.quantile(train_y.time[train_y["event"]==1], np.linspace(0.1, 1, 10))
            predictions = self.predict(train_x,times,format=format)
            metric = survival_regression_metric_modified(criterion,train_y,predictions,times,train_y)
            metric = np.mean(metric) # if the metric is a list return its mean
            print("Epoch {:02d}/{:02d}, sigma{:9.4f} Loss(ELBO) {:9.4f}, Metric_VAL {:9.4f}"
                    .format(epoch, num_epochs, self.log_sigma.exp(), train_loss/n, metric))   
                # early stopper for cold restart
            if metric > best_metric:
                best_metric = metric
                best_dict = current_dict
                save_path_2 = os.path.join(current_dir, "best_"+name+"_tuned.pth")
                torch.save(best_dict,save_path_2)

        if best_metric is not None:
            print("Best model in terms of "+criterion+" found, and the metric on the whole training dataset is " +str(round(best_metric, 4)))
            self.load_state_dict(torch.load(save_path_2))
        
        self.eval()  # Switch the model to evaluation mode
        return self