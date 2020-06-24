import numpy as np
import pandas as pd
from collections import Counter

import tokenization
from wordcloud import STOPWORDS
import string
import re
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

import torch
from torch import nn
from torch.nn import functional as F

def create_embedding_index(Path, head = 'y'):
    
    embeddings_index = {}
    with open(Path, encoding="utf8") as f:
        i = 0
        for line in f:
            if (i == 0)&(head=='y'):
                i = i+1
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            i = i + 1
            
    return embeddings_index

def build_vocab(df, Max_Num_Words = 100000, TEXT_COLUMN = 'comment_text'):

    counter = Counter()
    for text in df[TEXT_COLUMN].values:
        counter.update(text.split())

    vocab = {
        'token2id': {'<PAD>': 0, '<UNK>': Max_Num_Words + 1},
        'id2token': {}
    }
    
    vocab['token2id'].update(
        {token: _id + 1 for _id, (token, count) in
         enumerate(counter.most_common(Max_Num_Words))})
    
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    
    return vocab

def text2ids(text, token2id, MAX_SEQUENCE_LENGTH):
    return [
        token2id.get(token, len(token2id) - 1)
        for token in text.split()[:MAX_SEQUENCE_LENGTH]]

def tokenize(df, token2id, MAX_SEQUENCE_LENGTH, Comment_Column = 'text_proc'):
    
    texts = df[Comment_Column].values
    texts = [text2ids(text, token2id, MAX_SEQUENCE_LENGTH) for text in texts]
    
    return texts

ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer('english')

def create_embeddings_matrix(embeddings_index, word_index, MAX_NUM_WORDS = 30000):
    Embedding_dimension = len(embeddings_index['the'])
    nb_words = min(MAX_NUM_WORDS + 2, len(word_index))
    
    embedding_matrix = np.zeros((nb_words,
                                 Embedding_dimension))
    
    for key, i in word_index.items():
        word = key
        embedding_vector = embeddings_index.get(word)
            
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
            
    return embedding_matrix

def pad_text(texts, MAX_SEQUENCE_LENGTH):
    
    all_tokens = []
    
    for text in texts:
        
        pad_len = max(0,MAX_SEQUENCE_LENGTH - len(text))
        text = text[:MAX_SEQUENCE_LENGTH]
        
        text = [0] * pad_len + text
        
        all_tokens.append(text)
        
    return np.array(all_tokens)

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, LSTM_UNITS):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        #self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(4 * LSTM_UNITS, 4 * LSTM_UNITS)
        self.linear2 = nn.Linear(4 * LSTM_UNITS, 4 * LSTM_UNITS)
        
        self.linear_out = nn.Linear(4 * LSTM_UNITS, 1)
        self.linear_aux_out = nn.Linear(4 * LSTM_UNITS, num_aux_targets)
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        #h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out

class train_config:
    
    def __init__(self, num_to_load, valid_size, num_aux_targets, Max_Num_Words,
                 MAX_SEQUENCE_LENGTH, LSTM_UNITS, num_epoch, 
                 batch_size, learning_rate, accumulation_steps, PATH='./'):
        
        self.num_to_load = num_to_load
        self.valid_size = valid_size
        self.num_aux_targets = num_aux_targets
        self.Max_Num_Words = Max_Num_Words
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.LSTM_UNITS = LSTM_UNITS
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        self.PATH = PATH
