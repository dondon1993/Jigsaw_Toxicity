
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os


# In[2]:


import import_ipynb
from utils.Seed import seed_config, seed_everything
from utils.Preprocess import preproc_config, prepare_train_text
from utils.Bert import get_tokenized_samples, resort_index, clip_to_max_len, train_config


# In[3]:


from apex import amp
import apex
import sys


# In[4]:


from tqdm import tqdm_notebook
pd.options.display.precision = 6

import time
import warnings
warnings.filterwarnings("ignore")

from scipy import stats


# In[5]:


from transformers import BertTokenizer,BertForSequenceClassification
from pytorch_pretrained_bert import BertAdam


# In[6]:


import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F


# In[7]:


identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# In[8]:


def train_bert_cased(t_config, p_config, s_config):
    
    device=torch.device('cuda')
    seed_everything(s_config.seed)
    
    train = pd.read_csv('./input/train.csv').sample(t_config.num_to_load+t_config.valid_size,
                                                    random_state=s_config.seed)
    train = prepare_train_text(train, p_config)
    train = train.fillna(0)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_processed = get_tokenized_samples(t_config.MAX_SEQUENCE_LENGTH, tokenizer, train['text_proc'])
    
    sequences = train_processed
    lengths = np.argmax(sequences == 0, axis=1)
    lengths[lengths == 0] = sequences.shape[1]
    
    MyModel = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=t_config.num_labels)
    MyModel.to(device)
    
    # Possbile way to refactor the code below is to write pipeline for train and validation data
    # Prepare target
    target_train = train['target'].values[:t_config.num_to_load]
    target_train_aux = train[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values[:t_config.num_to_load]
    target_train_identity = train[identity_columns].values[:t_config.num_to_load]
    target_val = train['target'].values[t_config.num_to_load:]
    target_val_aux = train[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values[t_config.num_to_load:]
    target_val_identity = train[identity_columns].values[t_config.num_to_load:]
    
    # Prepare training data 
    inputs_train = train_processed[:t_config.num_to_load]
    inputs_val = train_processed[t_config.num_to_load:]
    weight_train = train['weight'].values[:t_config.num_to_load]
    weight_val = train['weight'].values[t_config.num_to_load:]
    lengths_train = lengths[:t_config.num_to_load]
    lengths_val = lengths[t_config.num_to_load:]
    
    inputs_train = torch.tensor(inputs_train,dtype=torch.int64)
    Target_train = torch.Tensor(target_train)
    Target_train_aux = torch.Tensor(target_train_aux)
    Target_train_identity = torch.Tensor(target_train_identity)
    weight_train = torch.Tensor(weight_train)
    Lengths_train = torch.tensor(lengths_train,dtype=torch.int64)
    
    inputs_val = torch.tensor(inputs_val,dtype=torch.int64)
    Target_val = torch.Tensor(target_val)
    Target_val_aux = torch.Tensor(target_val_aux)
    Target_val_identity = torch.Tensor(target_val_identity)
    weight_val = torch.Tensor(weight_val)
    Lengths_val = torch.tensor(lengths_val,dtype=torch.int64)
    
    # Prepare dataset
    train_dataset = data.TensorDataset(inputs_train, Target_train, Target_train_aux, Target_train_identity, weight_train, Lengths_train)
    val_dataset = data.TensorDataset(inputs_val, Target_val, Target_val_aux, Target_val_identity, weight_val, Lengths_val)
    
    ids_train = lengths_train.argsort(kind="stable")
    ids_train_new = resort_index(ids_train, t_config.num_of_bucket, s_config.seed)
    
    train_loader = torch.utils.data.DataLoader(data.Subset(train_dataset, ids_train_new), batch_size=t_config.batch_size, collate_fn=clip_to_max_len, shuffle=False)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in list(MyModel.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in list(MyModel.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr = t_config.learning_rate, betas = [0.9, 0.999], warmup=t_config.warmup, 
                         t_total = t_config.num_epoch * len(train_loader) // t_config.accumulation_steps)
    
    i = 0
    for n, p in list(MyModel.named_parameters()):
        if i < 10:
            p.requires_grad = False
        i += 1
        
    p = train['target'].mean()
    likelihood = np.log(p/(1-p))
    model_bias = torch.tensor(likelihood).type(torch.float)
    MyModel.classifier.bias = nn.Parameter(model_bias.to(device))
    
    MyModel, optimizer = amp.initialize(MyModel, optimizer, opt_level="O1",verbosity=0)

    for epoch in range(t_config.num_epoch):
        i = 0

        print('Training start')

        optimizer.zero_grad()
        MyModel.train()
        for batch_idx, (input, target, target_aux, target_identity, sample_weight) in tqdm_notebook(enumerate(train_loader),total=len(train_loader)):

            y_pred = MyModel(input.to(device), 
                             attention_mask = (input>0).to(device),
                            )
            loss =  F.binary_cross_entropy_with_logits(y_pred[0][:,0], target.to(device), reduction='none')
            loss = (loss * sample_weight.to(device)).sum()/(sample_weight.sum().to(device))
            loss_aux = F.binary_cross_entropy_with_logits(y_pred[0][:,1:6],target_aux.to(device), reduction='none').mean(axis=1)
            loss_aux = (loss_aux * sample_weight.to(device)).sum()/(sample_weight.sum().to(device))
            loss += loss_aux
            if t_config.num_labels == 15:
                loss_identity = F.binary_cross_entropy_with_logits(y_pred[0][:,6:],target_identity.to(device), reduction='none').mean(axis=1)
                loss_identity = (loss_identity * sample_weight.to(device)).sum()/(sample_weight.sum().to(device))
                loss += loss_identity

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i+1) % t_config.accumulation_steps == 0:             
                optimizer.step()                            
                optimizer.zero_grad()

            i += 1

        torch.save({
            'model_state_dict': MyModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'{t_config.PATH}')


# In[ ]:


if __name__ == "__main__":
    
    config_path = sys.argv[1]
    with open(config_path) as json_file:
        configs = json.load(json_file)
        
    t_config = train_config(
        num_to_load = configs['train_config']['num_to_load'], 
        valid_size = configs['train_config']['valid_size'],
        num_labels = configs['train_config']['num_labels'],
        MAX_SEQUENCE_LENGTH = configs['train_config']['MAX_SEQUENCE_LENGTH'], 
        num_of_bucket = configs['train_config']['num_of_bucket'],
        num_epoch = configs['train_config']['num_epoch'],
        batch_size = configs['train_config']['batch_size'],
        learning_rate = configs['train_config']['learning_rate'],
        accumulation_steps = configs['train_config']['accumulation_steps'], 
        warmup = configs['train_config']['warmup'],
        PATH = configs['train_config']['PATH']
    )
    p_config = preproc_config(
        lower_case = configs['preproc_config']['lower_case'], 
        replace_at = configs['preproc_config']['replace_at'],
        clean_tweets = configs['preproc_config']['clean_tweets'],
        replace_misspell = configs['preproc_config']['replace_misspell'],
        replace_star = configs['preproc_config']['replace_star'],
        replace_puncts = configs['preproc_config']['replace_puncts'],
    )
    s_config = seed_config(
        SEED = configs['seed_config']['SEED']
    )
    
    train_bert_cased(t_config, p_config, s_config)

