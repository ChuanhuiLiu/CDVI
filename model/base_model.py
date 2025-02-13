import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim,activation= "Tanh",dropout = 0.5):  # layer_sizes=[px, h1, ..., hl]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.Tanh()
        if activation == "Sigmoid":
            act_func =  nn.Sigmoid()
        if activation == "LeakyReLU":
            act_func =  nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.MLP = nn.Sequential()  # [px+py, h1, ..., hl]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.1))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))     
            self.MLP.add_module(name="A{:d}".format(i), module=act_func)       
        self.linear_mean = nn.Linear(layer_sizes[-1], latent_dim)  # [hl, pz]
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)
    def forward(self, x, y=None):
        x = torch.cat((x, y), dim=-1)
        x = self.MLP(x)
        x = self.dropout(x)
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mean, log_var  # [batch_size, pz]

class Delta_encoder(nn.Module):
    """
    splits network without shared weights for the encoder of CDVI: censoring indicator leads to different networks instead of inputs
    """ 
    def __init__(self, layer_sizes, latent_dim,activation= "Tanh",dropout = 0.5):  # layer_sizes=[px, h1, ..., hl]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.Tanh()
        if activation == "Sigmoid":
            act_func =  nn.Sigmoid()
        if activation == "LeakyReLU":
            act_func =  nn.LeakyReLU(0.1)
        # Split encoder for censor
        self.dropout = nn.Dropout(p=dropout)
        self.MLP_cns = nn.Sequential()  # [px+py, h1, ..., hl]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.1))
            self.MLP_cns.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))     
            self.MLP_cns.add_module(name="A{:d}".format(i), module=act_func)       
        self.linear_mean_cns = nn.Linear(layer_sizes[-1], latent_dim)  # [hl, pz]
        self.linear_log_var_cns = nn.Linear(layer_sizes[-1], latent_dim)
        # Split encoder for event
        self.MLP_evn= nn.Sequential()  # [px+py, h1, ..., hl]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.1))
            self.MLP_evn.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))     
            self.MLP_evn.add_module(name="A{:d}".format(i), module=act_func)       
        self.linear_mean_evn = nn.Linear(layer_sizes[-1], latent_dim)  # [hl, pz]
        self.linear_log_var_evn = nn.Linear(layer_sizes[-1], latent_dim)

    def mask_cat(self, mean_cns, log_var_cns,mean_evn,log_var_evn, mask_evn, mask_cns):
        dim_1 = mean_cns.shape[0]+mean_evn.shape[0]
        dim_2 = mean_cns.shape[1]
        
        joint_mean = torch.zeros([dim_1,dim_2],device=mean_cns.device, requires_grad=False)
        joint_log_var = torch.zeros([dim_1,dim_2],device=mean_cns.device, requires_grad=False)
        joint_mean[mask_evn,:], joint_mean[mask_cns,:] = mean_evn,mean_cns
        joint_log_var[mask_evn,:], joint_log_var[mask_cns,:] = log_var_evn,log_var_cns
        return  joint_mean,joint_log_var
    
    def forward(self, x, y,delta):
        # reshape 1d time/index array to [len(time),1]
        if len(y.shape)==1:
            y= y.unsqueeze(-1)
        if len(delta.shape)==1:
            delta= delta.unsqueeze(-1)
        # mask the event and censor index
        mask_evn = delta.squeeze()==True
        mask_cns = delta.squeeze()==False
        # forward pass
        x = torch.cat((x, y), dim=-1)
        mlp_cns = self.MLP_cns(x[mask_cns,:])
        mean_cns = self.linear_mean_cns(mlp_cns)
        log_var_cns = self.linear_log_var_cns(mlp_cns)
        mlp_evn = self.MLP_evn(x[mask_evn,:])
        mean_evn = self.linear_mean_evn(mlp_evn)
        log_var_evn = self.linear_log_var_evn(mlp_evn)
        joint_mean,joint_log_var = self.mask_cat(mean_cns, log_var_cns,mean_evn,log_var_evn,mask_evn, mask_cns)
        return joint_mean,joint_log_var  # [batch_size, dim_of_z]

class Joint_encoder(nn.Module):
    """
    dense network with shared weights for the encoder of CDVI: censoring indicator leads to different networks instead of inputs
    """ 
    def __init__(self, layer_sizes, latent_dim,activation= "Tanh",dropout=0.5):  # layer_sizes=[px, h1, ..., hl]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.Tanh()
        if activation == "Sigmoid":
            act_func =  nn.Sigmoid()
        if activation == "LeakyReLU":
            act_func =  nn.LeakyReLU(0.1)
        # Joint encoder
        self.dropout = nn.Dropout(p=dropout)
        self.MLP = nn.Sequential()  # [px+py, h1, ..., hl]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.1))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))     
            self.MLP.add_module(name="A{:d}".format(i), module=act_func)       
        self.linear_mean = nn.Linear(layer_sizes[-1], latent_dim)  # [hl, pz]
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, y,delta):
        # reshape 1d time/index array to [len(time),1]
        if len(y.shape)==1:
            y= y.unsqueeze(-1)
        if len(delta.shape)==1:
            delta= delta.unsqueeze(-1)
        # forward pass
        x = self.MLP(torch.cat((x, y,delta), dim=-1))
        joint_mean,joint_log_var = self.linear_mean(x),self.linear_log_var(x)
        return joint_mean,joint_log_var  # [batch_size, dim_of_z]

class Decoder(nn.Module):
    def __init__(self, layer_sizes, input_size,activation="ReLU"):  # layer_sizes=[hl, ..., h1, (px+py)*nrep]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.Tanh()
        if activation == "Sigmoid":
            act_func =  nn.Sigmoid()
        if activation == "LeakyReLU":
            act_func =  nn.LeakyReLU(0.05)
        self.MLP = nn.Sequential()  #
        input_size = input_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.2))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=act_func)
            self.linear = nn.Linear(out_size, out_size)
    def forward(self, z,x):
        z = torch.cat((z, x), dim=-1)
        u = self.MLP(z)
        u = self.linear(u)
        return u
    

    
class Sigma(nn.Module):
    def __init__(self, layer_sizes, input_size,activation="ReLU"):  # layer_sizes=[hl, ..., h1, (px+py)*nrep]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.Tanh()
        if activation == "Sigmoid":
            act_func =  nn.Sigmoid()
        if activation == "LeakyReLU":
            act_func =  nn.LeakyReLU(0.05)
        self.MLP = nn.Sequential()  #
        input_size = input_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            #self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.2))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=act_func)
            self.linear = nn.Linear(out_size, out_size)
    def forward(self, x):
        sigma = self.MLP(x)
        sigma = self.linear(sigma)
        return sigma