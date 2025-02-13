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
from model.base_model import Decoder, Delta_encoder, Joint_encoder,Sigma
from utils.gumbel_min import Gumbelmin
from utils.metrics import loss_fn
from utils.override_functions import survival_regression_metric_modified
# xavierinitialization 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class CDCVAE(nn.Module):
    """
    CDVI: censor-dependent conditional CVAE for survival analysis
    """
    def __init__(self, encoder_layer_sizes, latent_dim, px, decoder_layer_sizes,sigma_learning = "joint",encoder_type = "dense",primative ="normal",dropout=0.5):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_dim) == int
        assert type(decoder_layer_sizes) == list
        self.latent_dim = latent_dim
        self.px = px
        self.primative = primative
        self.sigma_learning = sigma_learning
        self.encoder_type = encoder_type
        # intialize the network structure
        self.decoder = Decoder(decoder_layer_sizes,input_size=latent_dim+px,activation = "Tanh") #"LeakyReLU" is also suitable
        if encoder_type == "split":
            self.encoder = Delta_encoder(encoder_layer_sizes, latent_dim,activation = "Tanh",dropout=dropout)
        elif encoder_type == "dense":
            encoder_layer_sizes[0]+=1
            self.encoder = Joint_encoder(encoder_layer_sizes, latent_dim,activation = "Tanh",dropout=dropout)
        else:
            raise(RuntimeError("the type of sigma learning not specified"))
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        # handle the learning process of the sigma (scale parameter) of $U$
        if sigma_learning== "joint":
            self.log_sigma =  nn.Parameter(torch.tensor(1,dtype=torch.float32), requires_grad=True) #parameter to be learned
        elif sigma_learning == "fixed":
            self.log_sigma = torch.tensor(0,dtype=torch.float32) #fixed at 0
        elif sigma_learning == "xdependent":
            self.sigmanet = Sigma([32,16,1],input_size=px)
        elif sigma_learning == "coordiante":
            self.log_sigma = torch.tensor(0,dtype=torch.float32) #intialized at 0
        else:
            raise(RuntimeError("the type of sigma learning not specified"))
    

    def reparameterize(self, mean, log_var): #log_var is the diag var of encoder for reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x_train, y_train,e_train):
        mean, log_var = self.encoder(x_train,y_train,e_train)  # [batch_size, latent_dim]
        z = self.reparameterize(mean, log_var)  # [batch_size, nrep] 
        recon = self.decoder(z,x_train)  
        return recon, mean, log_var #mu_y, mu_z, log_sigma^2_z
    
    def validate(self,x_val,y_val,e_val, y_train,e_train,event_quantile = True):
        # build-in module of computing ctd for validation dataset 
        if isinstance(y_val,pd.DataFrame):
            raise RuntimeError('pd.dataframe used, please convert to tensor')
        if isinstance(y_val,np.ndarray):
            x_val = torch.from_numpy(x_val).to(dtype=torch.FloatTensor, device = y_train.device)
            y_val = torch.from_numpy(y_val).to(dtype=torch.FloatTensor, device = y_train.device)  
            e_val = torch.from_numpy(e_val).to(dtype=torch.FloatTensor, device = y_train.device)     
        z = torch.randn(size=[len(x_val),self.latent_dim],device = y_train.device)  # prior of z
        loc_val = self.decoder(z,x_val)
        if event_quantile ==True:
            train_times = np.quantile(y_train[e_train==1], np.linspace(0.1, 1, 10))
        else:
            train_times = np.unique(y_train.flatten().cpu())
        if self.primative == "normal":
            predictions= norm.logsf(train_times, loc=loc_val.detach().numpy(), scale=torch.exp(self.log_sigma).detach().numpy())
        elif self.primative == "gumbel":
            gumbel = Gumbelmin(loc_val,torch.exp(self.log_sigma))
            predictions= gumbel.logsf(torch.from_numpy(train_times)).detach().numpy()
        else: raise RuntimeError('primative distribution not speficied')
        print("prediction complete"+str(type(predictions))+str(predictions.shape))

        ctd = EvalSurv(pd.DataFrame(predictions.T, index=train_times), y_val.flatten().cpu().numpy(), e_val.flatten().cpu().numpy(), censor_surv='km')
        return ctd.concordance_td()
    
    def test(self,x_test,y_test,e_test, y_train,e_train,event_quantile = True):
        # build-in module of computing ctd for test dataset
        if isinstance(y_test,pd.DataFrame):
            raise RuntimeError('pd.dataframe used, please parse')
        if isinstance(y_test,np.ndarray):
            x_test = torch.from_numpy(x_test).to(dtype=torch.FloatTensor, device = y_train.device)
            y_test = torch.from_numpy(y_test).to(dtype=torch.FloatTensor, device = y_train.device)
            e_test = torch.from_numpy(e_test).to(dtype=torch.FloatTensor, device = y_train.device)        
        z = torch.randn(size=[len(x_test),self.latent_dim],device = y_train.device)
        loc_test = self.decoder(z,x_test)
        if event_quantile ==True:
            train_times = np.quantile(y_train[e_train==1], np.linspace(0.1, 1, 10))
        else:
            train_times = np.unique(y_train.flatten().cpu().numpy())
        if self.primative == "normal":
            predictions= norm.logsf(train_times, loc=loc_test.detach().cpu().numpy(), scale=torch.exp(self.log_sigma).detach().numpy())
        elif self.primative == "gumbel":
            gumbel = Gumbelmin(loc_test,torch.exp(self.log_sigma))
            predictions= gumbel.logsf(torch.from_numpy(train_times)).detach().numpy()
        ctd = EvalSurv(pd.DataFrame(predictions.T, index=train_times),y_test.flatten().detach().numpy(), e_test.flatten().detach().numpy(), censor_surv='km')
        return ctd.concordance_td()    
    
    def predict(self,x,times,format="pre",expo=True):
        """Predict survival at specified time(s) using the Censor-dependent Variational Autoencoders (CD-CVAE) series of models on CPU.
        
        Parameters
        x : pd.DataFrame
            A pandas dataframe with rows corresponding to individual samples and
            columns as covariates.
        times : float or list
            A float or list of the times at which to compute the survival probability.
        format: string
            determine how z is sampled when generating the individual survival distribution: 
            if "pre" then z is sampled from standartd normal distribution (the prior), without using the information of times.
            if "post" then z is sampled from the encoder distribution (approximation posterior) which take both (x,times) as input.
        exp: bool
            determine if the exponential-transform will be applied to output: 
            if False then the output is log-transformed. logsf will be used directly 
            if True then the output returns the survival probability (needed for metrics like brs)
        Returns
        predictions: np.array : An array of the survival probabilites at each time point in times. [len(x),len(times)]
        """       
        # predict the survival time
        x = torch.tensor(x.to_numpy(),dtype=torch.float32)
        if isinstance(times, (float,int)):
            times = [times]
        times = torch.tensor(times,dtype=torch.float32)
        if format == "pre":
            z = torch.randn(size=[len(x),self.latent_dim])
            u_mean = self.decoder(z,x)
            if self.primative == "normal":
                predictions= norm.logsf(times.T, loc=u_mean.detach().numpy(), scale=torch.exp(self.log_sigma).detach().numpy())  #[len(x),len(times)]
            elif self.primative == "gumbel":
                gumbel = Gumbelmin(u_mean,torch.exp(self.log_sigma))
                predictions= gumbel.logsf(times).detach().numpy() #[len(x),len(times)]
        elif format == "post":
            predictions= []
            for time in times:
                vtime = torch.ones(size=[len(x),1])*time 
                e =  torch.ones(size=[len(x),1])
                mean, log_var = self.encoder(x,vtime,e)
                z = self.reparameterize(mean, log_var) # [len(x),latent_dim]
                u_mean = self.decoder(z,x)
                if self.primative == "normal":
                    pred= norm.logsf(time, loc=u_mean.detach().numpy(), scale=torch.exp(self.log_sigma).detach().numpy()) #[len(x),1] #torch.exp(self.log_sigma).detach().numpy()
                elif self.primative == "gumbel":
                    gumbel = Gumbelmin(u_mean,torch.exp(self.log_sigma)) #torch.exp(self.log_sigma)
                    pred= gumbel.logsf(time).detach().numpy() #[len(x),1]
                predictions.append(pred) 
            predictions = np.hstack(predictions) #[len(x),len(times)] numpy.ndarray
        else:
            raise NotImplementedError("inference process is not correctly specified")
        
        if expo is True:
            return np.exp(predictions)
        else:
            return predictions
    
    def fit(self,train_data, batch_size, num_epochs, learning_rate, format="pre",temperature=1,criterion="ctdpycox",early_stopper=True, patience=30):
        """Fit the Censor-dependent Variational Autoencoders (CD-CVAE) series of models to a given dataset and select the best module by cross validation. 
        It is a fully parametric approach and improves on the Accelerated Failure Time model by modelling the event time distribution 
        as a infinite size mixture over Weibull or Log-Normal distributions.

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
                        
        # select the best model

        n = len(x_train)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,weight_decay=0.01)
        x_ind = np.arange(n)
        loss_list =[0]
        KLD_list =[0]
        dict_list =[0] 
        metric_list = []
        if criterion == "ctdpycox" or criterion == "ctd":
            best_metric = float('-inf')
        elif criterion == "brs":
            best_metric = float("inf")
        best_epoch = 0
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
                pred_y,mean, log_var = self.forward(x, y,e)  
                if self.sigma_learning == "xdependent":
                    log_sigma = self.sigmanet(x)
                    loss,NLK,KL = loss_fn(pred_y, y,e, log_sigma, mean, log_var,primative = self.primative,beta =temperature) 
                else:
                    loss,NLK,KL = loss_fn(pred_y, y,e, self.log_sigma, mean, log_var,primative = self.primative,beta =temperature) 
                loss.backward()
                train_loss += loss.item()
                train_NLK +=NLK.item()
                train_KL += KL.item()
                optimizer.step()
            loss_list.append(train_loss/n)
            KLD_list.append(train_KL/n)
            current_dict = self.state_dict()
            if epoch % 2 == 0:
                if criterion == "ctdpycox":
                    times = np.quantile(y_tr.time[y_tr["event"]==1], np.linspace(0.1, 1, 10))
                    predictions = self.predict(x_val,times,format=format)
                    metric = survival_regression_metric_modified(criterion,y_val,predictions,times,y_tr)
                    metric = np.mean(metric) # if the metric is a list return its mean
                elif criterion  == "ctd":
                    times = np.quantile(y_tr.time[y_tr["event"]==1], 0.75)
                    predictions = self.predict(x_val,times,format=format)
                    metric = survival_regression_metric_modified(criterion,y_val,predictions,times,y_tr)
                    metric = np.mean(metric)
                elif criterion == "brs":
                    times = np.quantile(y_tr.time[y_tr["event"]==1], np.linspace(0.1, 1, 10))
                    predictions = self.predict(x_val,times,format=format,expo=True)
                    metric = survival_regression_metric_modified(criterion,y_val,predictions,times,y_tr)
                    metric = np.mean(metric) # if the metric is a list return its mean
                print("Epoch {:02d}/{:02d}, sigma{:9.4f} Loss(ELBO) {:9.4f}, Metric_VAL {:9.4f}"
                      .format(epoch, num_epochs, self.log_sigma.exp(), train_loss/n, metric))   
                # early stopper for cold restart
                
                if len(dict_list) >= 10:
                    dict_list.pop(0)
                dict_list.append(self.state_dict())
                metric_list.append(metric)
                if early_stopper == True and len(metric_list)> patience:
                    last_four = metric_list[-4:]
                    changes = [last_four[i] - last_four[i - 1] for i in range(1, len(last_four))]
                    if all(change == changes[0] for change in changes):
                        print("bad intialization/slow training detected. Restarting needed")
                        return None

            if criterion == "ctypycox" or criterion == "ctd": # largeer is better
                if metric > best_metric:
                    best_metric = metric
                    best_dict = dict_list[-1]
                    torch.save(best_dict,save_path)
                    best_epoch = epoch
                elif epoch-best_epoch>patience: # early stopping avoid overfitting and saving computation time
                    break
            elif criterion == "brs": #smaller is better
                if metric < best_metric:
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
                y = y_train[batch_id, :]
                e = e_train[batch_id]
                optimizer.zero_grad()
                pred_y,mean, log_var = self.forward(x, y,e)  
                if self.sigma_learning == "xdependent":
                    log_sigma = self.sigmanet(x)
                    loss,NLK,KL = loss_fn(pred_y, y,e, log_sigma, mean, log_var,primative = self.primative,beta =temperature) 
                else:
                    loss,NLK,KL = loss_fn(pred_y, y,e, self.log_sigma, mean, log_var,primative = self.primative,beta =temperature) 
                loss.backward()
                train_loss += loss.item()
                train_NLK +=NLK.item()
                train_KL += KL.item()
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

            if metric > best_metric:
                best_metric = metric
                best_dict = current_dict
                save_path_2 = os.path.join(current_dir, "best_"+name+"_tuned.pth")
                torch.save(best_dict,save_path_2)
            if criterion == "ctypycox" or criterion == "ctd": # largeer is better
                if metric > best_metric:
                    best_metric = metric
                    best_dict = current_dict
                    save_path_2 = os.path.join(current_dir, "best_"+name+"_tuned.pth")
                torch.save(best_dict,save_path_2)
            elif criterion == "brs": #smaller is better
                if metric < best_metric:
                    best_metric = metric
                    best_dict = current_dict
                    save_path_2 = os.path.join(current_dir, "best_"+name+"_tuned.pth")
                    torch.save(best_dict,save_path_2)
    
        if best_metric is not None:
            print("Best model in terms of "+criterion+" found, and the metric on the whole training dataset is " +str(round(best_metric, 4)))
            self.load_state_dict(torch.load(save_path_2))
        
        self.eval()  # Switch the model to evaluation mode
        return self
   