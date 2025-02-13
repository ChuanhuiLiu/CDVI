import numpy as np
import pandas as pd
import torch
import os
from auton_survival.datasets import load_dataset
from sklearn.model_selection import train_test_split
import pycox
import torchvision
import pickle

def data_loader(dataset= "SUPPORT",  **kwargs):
    """
    load datasets to test Survival Analysis models (modified and extended from auto_survival.datasets.load_dataset) 
    Args:
        dataset (str): Name of dataset to be loaded ["SUPPORT",'WHAS','PBC,'METABRIC','FLCHAIN','MNIST','GBSB','NWTCO']
        **kwargs (dict): Dataset specific keyword arguments.
    Returns:
        A tuple (pd.dataframe) of the form of [outcomes, features] 
        where outcomes = [censoring indicator, time-to-event]
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset == 'SUPPORT':
        file_path = os.path.join(root_dir, "datasets/support2.csv")
        #contain NaN, require preprocessing
        return _load_support(file_path)
    if dataset == 'WHAS':
        file_path = os.path.join(root_dir, "datasets/whas.csv")
        return _load_whas(file_path)
    if dataset == 'PBC':
        #contain NaN, require preprocessing
        file_path = os.path.join(root_dir, "datasets/pbc2.csv")
        return _load_pbc2(file_path)    
    if dataset == 'METABRIC':
        feature_path = os.path.join(root_dir, "datasets/metabric_feat.csv")
        label_path = os.path.join(root_dir, "datasets/metabric_label.csv")
        return _load_metabric(feature_path,label_path)    
        
    if dataset == 'SIMULATE':
        filename = "datasets/dataset_"+kwargs["clevel"]+".pkl"
        file_path = os.path.join(root_dir, filename)
        return _load_simulate(file_path, full= True)  

    # USING EXISTING DATASETS
    if dataset == 'FLCHAIN':
        #contain NaN, require preprocessing
        return _load_flchain()  
    if dataset == 'MNIST':
        return _load_mnist()
    if dataset == 'GBSG':
        return _load_gbsg()
    if dataset == 'NWTCO':
        return _load_nwtco()

    else:
        raise NotImplementedError('Dataset '+dataset+' not implemented.')

def _load_support(file_path):
    df = pd.read_csv(file_path)
    outcomes = df.copy()
    outcomes['event'] =  df['death']
    outcomes['time'] = df['d.time']
    outcomes = outcomes[['event', 'time']]

    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp',
                 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph',
                 'glucose', 'bun', 'urine', 'adlp', 'adls']
    features = df[cat_feats+num_feats]
    return outcomes, features

def _load_whas(file_path):
    df = pd.read_csv(file_path, sep=',')
    label = ['event', 'time']
    outcomes = df[label]
    features = df.drop(columns=label)
    return outcomes, features

def _load_pbc2(file_path):
    df = pd.read_csv(file_path) 

    df['histologic'] = df['histologic'].astype(str)
    df['age'] = df['age'].values + df['years'].values

    t = (df['years'] - df['year']).values
    e = df['status2'].values

    cat_feats= ['drug', 'sex', 'ascites', 'hepatomegaly',
                'spiders', 'edema', 'histologic']
    num_feats = ['age','serBilir', 'serChol', 'albumin', 'alkaline',
                'SGOT', 'platelets', 'prothrombin'] 
    outcomes = pd.DataFrame({"event":e,"time":t})
    features = df[cat_feats+num_feats]
    return outcomes, features

def _load_metabric(feature_path,label_path):
    df = pd.read_csv(feature_path, sep=',')
    surv_df = pd.read_csv(label_path, sep=',')
    outcomes = surv_df.copy()
    outcomes["event"] = surv_df['label']
    outcomes["time"] = surv_df['event_time']
    outcomes = outcomes[['event', 'time']]
    selected_columns = ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'stage', 'lymph_nodes_removed','grade', 'grade.1','grade.2']
    #outcomes =  pd.DataFrame(data={"event":e,"time":t},index=[0])
    # This column is redundant
    df = df[selected_columns ]
    #df = df.drop(labels=['NPI'], axis=1)  
    # Remove ER_IHC_status.1, ER_Expr.1, PR_Expz.1
    #df = df.drop(labels=['ER_IHC_status', 'ER_Expr', 'PR_Expz', 'Her2_Expr', 'inf_men_status'], axis=1)
    return outcomes, df

def _load_simulate(file_path, full = True):
    with open(file_path, "rb") as f:
        x_samples,y_samples,delta_samples,z_samples,u_samples,mus,llks,postprobs = pickle.load(f)
    # print(len(y_samples))  10000
    outcomes = pd.DataFrame({'event':delta_samples,"time":y_samples})
    features = pd.DataFrame({"x":x_samples})
    latent_dict = {"z":z_samples,"u":u_samples,"mu":mus,"llk":llks,"ppb":postprobs}
    if full == True:
        return outcomes, features,latent_dict
    else: 
        return outcomes, features

def _load_flchain():
    df = pycox.datasets.flchain.read_df(processed=False)  # Processed is set to False because there is a bug in Pycox library.
    outcomes = df.copy() 
    outcomes['event'] = df[['death']]
    outcomes['time'] =  df.apply(lambda y : 1e-6 if y['futime'] ==0 else y['futime'], axis=1)  
    outcomes = outcomes[['event', 'time']]

    cat_feats= ['sex','sample.yr','flc.grp','mgus']
    num_feats = ['age','kappa', 'lambda', 'creatinine'] 
    features = df[cat_feats+num_feats]

    return outcomes, features

def _load_gbsg():
    df = pycox.datasets.gbsg.read_df()  # Processed is set to False because there is a bug in Pycox library.
    df = df.astype(float)
    outcomes = df.copy()
    outcomes['event'] = df[['event']]
    outcomes['time'] = df[['duration']]
    outcomes = outcomes[['event', 'time']]

    features = df.drop(labels=['event', 'duration'], axis=1)
    return outcomes, features

def _load_nwtco():
    df = pycox.datasets.nwtco.read_df(processed=False)  
    outcomes = pd.DataFrame({'event':df['rel'],'time': df['edrel']})
    features = df.drop(labels=['rel', 'edrel','rownames', 'seqno'], axis=1)

    return outcomes, features


# for creating survival MNIST - modified from auton-survival.dataset
def increase_censoring(e, t, p, random_seed=0):

  np.random.seed(random_seed)
  uncens = np.where(e == 1)[0]
  # censoring e=1 at prob. of 1-p
  mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
  toswitch = uncens[mask]

  e[toswitch] = np.array([0])
  t_ = t[toswitch]
  # replace event time by random censored time
  newt = []
  for t__ in t_:
    newt.append(np.random.uniform(1, t__))
  t[toswitch] = newt

  return e, t

def _load_mnist():
  """Helper function to load the MNIST and tranform label to survival outcome
  """
  train = torchvision.datasets.MNIST(root='datasets/',
                                     train=True, download=True)
  x = train.data.numpy() #(60000, 28, 28)
  x = x.reshape(60000, -1).astype(float) #(60000, 784) 
  t = train.targets.numpy().astype(float) + 1 
  event, time = increase_censoring(np.ones(t.shape), t, p=.5)
  outcomes = pd.DataFrame({"event":pd.Series(event).astype(float),"time":pd.Series(time)})
  feature = pd.DataFrame(x)
  return outcomes, feature



# list = ['SUPPORT','WHAS','PBC','METABRIC','FLCHAIN','MNIST','GBSB','NWTCO']
# for name in list: 
#    print('============= loading '+name+' dataset  =====================')
#    outcomes,feature = data_loader(name)
#    print(outcomes.head(5))
#    print(feature.head(5))
#    print('============= '+name+" dataset loaded sucessfully =====================")

