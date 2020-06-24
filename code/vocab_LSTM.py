
# coding: utf-8

# This script is used to generate the vocab file that is used in all LSTM models

# In[1]:


import numpy as np
import pandas as pd

import os


# In[ ]:


import import_ipynb
from utils.Seed import seed_config, seed_everything
from utils.Preprocess import preproc_config, prepare_train_text
from utils.LSTM import train_config, create_embedding_index, build_vocab


# In[3]:


import pickle


# In[4]:


def generate_vocab(t_config, p_config, s_config):
    
    seed_everything(s_config.seed)
    train = pd.read_csv('./input/train.csv').sample(t_config.num_to_load+t_config.valid_size,
                                                    random_state=s_config.seed)
    train = prepare_train_text(train, p_config)
    train = train.fillna(0)
    
    Path_fast = './word_vector/cc.en.300.vec'
    embeddings_index_fast = create_embedding_index(Path_fast,'y')
    
    Path_glove = './word_vector/glove.6B.300d.txt'
    embeddings_index_glove = create_embedding_index(Path_glove,'no')
    
    with open('embeddings_fast.pickle', 'wb') as handle:
        pickle.dump(embeddings_index_fast, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    with open('embeddings_glove.pickle', 'wb') as handle:
        pickle.dump(embeddings_index_glove, handle, protocol = pickle.HIGHEST_PROTOCOL)

    vocab = build_vocab(train, t_config.Max_Num_Words, 'text_proc')
    
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol = pickle.HIGHEST_PROTOCOL)


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
        PATH = './checkpoint/model-LSTM.pth'
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
        SEED = 641
    )
    
    generate_vocab(t_config, p_config, s_config)

