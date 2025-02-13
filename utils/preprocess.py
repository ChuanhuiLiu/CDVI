import torch
import numpy as np
from scipy.stats import norm, gamma
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils.gumbel_min import Gumbelmin

def pre_process(features,outcomes,dataset=None, to_numpy=True,to_tensor=True, ptest=0.2, pval=0.25,random_state=0, log = True, imputation = True):
    """
    preprocess module of CIVIL-CVAE for the raw survival datasets, 
    including train-test split and standard preprocess functions.
    """

    if not isinstance(features,pd.DataFrame) or not isinstance(outcomes,pd.DataFrame):
        raise(RuntimeError('features and outcomes must be pd.dataframe'))
    
    if log == False: # no need for log transform (e.g. simulated dataset)
        outcomes['time'] = outcomes['time'] 
    else:
        outcomes['time'] = np.log(outcomes['time'] )
    
    # train_test_split on pd.df
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=ptest, random_state=random_state)

    # train_valid_split on pd.df
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=pval, random_state=random_state) 

    if imputation ==False: # skip imputation and normalization for simulated dataset
        train_x,train_y,train_event = x_tr.to_numpy(),y_tr["time"].to_numpy(),y_tr["event"].to_numpy()
        valid_x,valid_y,valid_event =  x_te.to_numpy(),y_te["time"].to_numpy(),y_te["event"].to_numpy()
        test_x,test_y,test_event = x_val.to_numpy(),y_val["time"].to_numpy(),y_val["event"].to_numpy()
        return train_x,train_y,train_event, valid_x,valid_y,valid_event , test_x,test_y,test_event

    # preprocess for each benchmark dataset
    if dataset =="SUPPORT":
        cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
        num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
                'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                'glucose', 'bun', 'urine', 'adlp', 'adls']
    elif dataset == 'WHAS':
        cat_feats = ['sex', 'chf','miord']
        num_feats = ['age', 'bmi']        
    elif dataset == 'PBC':
        cat_feats = ['drug', 'sex', 'ascites', 'hepatomegaly',
                'spiders', 'edema', 'histologic']
        num_feats = ['age','serBilir', 'serChol', 'albumin', 'alkaline',
                'SGOT', 'platelets', 'prothrombin']   
    elif dataset == 'METABRIC':
        # already cleaned
        cat_feats = None
        num_feats = None
    elif dataset == 'FLCHAIN':
        cat_feats = ['sex','flc.grp','mgus']
        num_feats = ['age','sample.yr', 'kappa', 'lambda', 'creatinine'] 
    elif dataset == 'MNIST':
        # already cleaned
        cat_feats = None
        num_feats = None
    elif dataset == 'GBSG':
        # already cleaned from pycox
        cat_feats = None
        num_feats = None
    elif dataset == 'NWTCO':
        cat_feats = ['instit', 'histol', 'stage', 'study','in.subcohort']
        num_feats = ['age']
    else:
        raise NotImplementedError('Preprocessing for Dataset '+dataset+' not implemented.')
    
    if cat_feats and num_feats is not None:
        from auton_survival.preprocessing import Preprocessor
        #print(x_tr)
        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat= 'mean') 
        transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,one_hot=True, fill_value=-1)
        x_tr = transformer.transform(x_tr)
        #print("after preprocess")
        #print(x_tr)
        x_val = transformer.transform(x_val) #still pd.dataframe object
        x_te = transformer.transform(x_te)
    
    if to_numpy is True:
        train_x,train_y,train_event = x_tr.to_numpy(),y_tr["time"].to_numpy(),y_tr["event"].to_numpy() # numpy ndarray
        valid_x,valid_y,valid_event =  x_te.to_numpy(),y_te["time"].to_numpy(),y_te["event"].to_numpy()
        test_x,test_y,test_event = x_val.to_numpy(),y_val["time"].to_numpy(),y_val["event"].to_numpy()
        if to_tensor is True:
            train_x  = torch.tensor(train_x, dtype=torch.float32)
            train_y  = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1) # numpy ndarray n x 1
            train_event = torch.tensor(train_event, dtype=torch.float32)
            valid_x = torch.tensor(valid_x, dtype=torch.float32)
            valid_y = torch.tensor(valid_y, dtype=torch.float32).unsqueeze(-1)
            valid_event =  torch.tensor(valid_event, dtype=torch.float32)
            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1)
            test_event = torch.tensor(test_event, dtype=torch.float32)
        # return numpys or tensors
        return train_x,train_y,train_event, valid_x,valid_y,valid_event,test_x,test_y,test_event
    else:
        # return pd.dataframes where y contain both time and event column
        return x_tr,y_tr,x_val,y_val,x_te,y_te
    

# list = ['SUPPORT','WHAS','PBC','METABRIC','FLCHAIN','MNIST','GBSG','NWTCO']
#from utils.data_load import data_loader
#name = "FLCHAIN"
#outcomes,features = data_loader(name)
#print('============= '+name+" dataset loaded sucessfully =====================")
#train_x,train_y,train_event, valid_x,valid_y,valid_event , test_x,test_y,test_event=pre_process(features,outcomes,dataset=name)

def pre_process_simulate(features,outcomes,mus, to_tensor=True,**kwargs):
    """
    preprocess module for the simulated datasets
    """
    # train_test_split on pd.df
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes,mus, test_size=0.2, random_state=1)
    # train_valid_split on pd.df
    #print(x_tr)

    train_x,train_y,train_event = x_tr.to_numpy(),y_tr["time"].to_numpy(),y_tr["event"].to_numpy()
    test_x,test_y,test_event = x_te.to_numpy(),y_te["time"].to_numpy(),y_te["event"].to_numpy()
    if to_tensor is True:
        train_x  = torch.tensor(train_x, dtype=torch.float32)
        train_y  = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)
        train_event = torch.tensor(train_event, dtype=torch.float32)
        valid_x = torch.tensor(valid_x, dtype=torch.float32)
        valid_y = torch.tensor(valid_y, dtype=torch.float32).unsqueeze(-1)
        valid_event =  torch.tensor(valid_event, dtype=torch.float32)
        test_x = torch.tensor(test_x, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1)
        test_event = torch.tensor(test_event, dtype=torch.float32)

    return train_x,train_y,train_event,test_x,test_y,test_event
