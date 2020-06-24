
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os


# In[ ]:


import import_ipynb
from utils.Seed import seed_config, seed_everything
from utils.Preprocess import preproc_config, prepare_train_text
from utils.LSTM import train_config, create_embeddings_matrix, tokenize, pad_text, NeuralNet


# In[3]:


from tqdm import tqdm_notebook
pd.options.display.precision = 6

import gc
import warnings
warnings.filterwarnings("ignore")


# In[4]:


import pickle


# In[5]:


import torch
from torch.utils import data
from torch.nn import functional as F
from torch.optim import Adam


# In[6]:


from apex import amp


# In[7]:


identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# In[17]:


def train_LSTM(t_config, p_config, s_config):
    
    device=torch.device('cuda')
    seed_everything(s_config.seed)
    
    train = pd.read_csv('./input/train.csv').sample(t_config.num_to_load+t_config.valid_size,
                                                    random_state=s_config.seed)
    train = prepare_train_text(train, p_config)
    train = train.fillna(0)
    
    with open('./vocab/embeddings_fast.pickle', 'rb') as handle:
        embeddings_index_fast = pickle.load(handle)
    
    with open('./voab/embeddings_glove.pickle', 'rb') as handle:
        embeddings_index_glove = pickle.load(handle)

    with open('./vocab/vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    
    texts_train = tokenize(train, vocab['token2id'], t_config.MAX_SEQUENCE_LENGTH, 'text_proc')
    
    embeddings_matrix_fast = create_embeddings_matrix(embeddings_index_fast, vocab['token2id'], t_config.Max_Num_Words)
    embeddings_matrix_glove = create_embeddings_matrix(embeddings_index_glove, vocab['token2id'], t_config.Max_Num_Words)
    
    texts_train = pad_text(texts_train, t_config.MAX_SEQUENCE_LENGTH)
    
    target_train = train['target'].values[:t_config.num_to_load]
    target_train_aux = train[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values[:t_config.num_to_load]
    target_train_identity = train[identity_columns].values[:t_config.num_to_load]
    inputs_train = texts_train[:t_config.num_to_load]
    weight_train = train['weight'].values[:t_config.num_to_load]
    
    inputs_train = torch.tensor(inputs_train, dtype=torch.int64)
    Target_train = torch.Tensor(target_train)
    Target_train_aux = torch.Tensor(target_train_aux)
    Target_train_identity = torch.Tensor(target_train_identity)
    weight_train = torch.Tensor(weight_train)
    
    train_dataset = data.TensorDataset(inputs_train, Target_train, Target_train_aux, Target_train_identity, weight_train)
    
    embeddings_matrix = np.concatenate((embeddings_matrix_fast, embeddings_matrix_glove), axis=1)
    MyModel = NeuralNet(embeddings_matrix, 14, t_config.LSTM_UNITS)
    MyModel.to(device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=t_config.batch_size, shuffle=False)
    optimizer_grouped_parameters = [
        {'params': [p for p in list(MyModel.parameters())]},
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr = t_config.learning_rate, betas = [0.9, 0.999])
    
    MyModel, optimizer = amp.initialize(MyModel, optimizer, opt_level="O1",verbosity=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.7 ** epoch)
    
    for epoch in range(t_config.num_epoch):
        
        i = 0
        scheduler.step()

        # Training step
        print('Training start')

        optimizer.zero_grad()
        MyModel.train()
        for batch_idx, (input, target, target_aux, target_identity, sample_weight) in tqdm_notebook(enumerate(train_loader),total=len(train_loader)):

            y_pred = MyModel(input.to(device)
                            )
            loss =  F.binary_cross_entropy_with_logits(y_pred[:,0], target.to(device), reduction='none')
            loss = (loss * sample_weight.to(device)).sum()/(sample_weight.sum().to(device))
            loss_aux = F.binary_cross_entropy_with_logits(y_pred[:,1:6],target_aux.to(device), reduction='none').mean(axis=1)
            loss_aux = (loss_aux * sample_weight.to(device)).sum()/(sample_weight.sum().to(device))
            loss += loss_aux

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i+1) % t_config.accumulation_steps == 0:             
                optimizer.step()                            
                optimizer.zero_grad()

            i += 1


        torch.save({
            'model_state_dict': MyModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'{t_config.PATH}_{epoch}')


# In[ ]:


if __name__ == "__main__":
    
    t_config = train_config(
        num_to_load = 1500000,
        valid_size = 100000,
        num_aux_targets = 14,
        Max_Num_Words = 30000,
        MAX_SEQUENCE_LENGTH = 250,
        LSTM_UNITS = 400,
        num_epoch = 3,
        batch_size = 128,
        learning_rate = 5e-4,
        accumulation_steps = 4,
        self.PATH = './checkpoint/model-LSTM.pth'
    )
    p_config = preproc_config(
        lower_case = True, 
        replace_at = True,
        clean_tweets = True,
        replace_misspell = True,
        replace_star = True,
        replace_puncts = True,
    )
    s_config = seed_config(
        SEED = 7529
    )
    
    train_LSTM(t_config, p_config, s_config)

