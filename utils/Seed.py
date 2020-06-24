
# coding: utf-8

# In[11]:


import random
import numpy as np
import torch
import pickle


# In[12]:


class seed_config:
    def __init__(self, SEED):
        self.seed = SEED


# In[13]:


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    with open('torch_seed.pickle', 'rb') as handle:
        l = pickle.load(handle)

    torch.cuda.set_rng_state(l)

