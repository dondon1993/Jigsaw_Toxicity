import numpy as np
import pandas as pd
import sys

from utils.Seed import seed_config, seed_everything
from utils.Preprocess import preproc_config, prepare_train_text
from utils.LSTM import train_config, create_embeddings_matrix, tokenize, pad_text, NeuralNet

from tqdm import tqdm_notebook

import gc
import pickle

import torch
from torch.utils import data
from torch.nn import functional as F
from torch.optim import Adam
from apex import amp

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

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

if __name__ == "__main__":
    
    config_path = sys.argv[1]
    with open(config_path) as json_file:
        configs = json.load(json_file)

    t_config = train_config(
        num_to_load = configs['train_config']['num_to_load'],
        valid_size = configs['train_config']['valid_size'],
        num_aux_targets = configs['train_config']['num_aux_targets'],
        Max_Num_Words = configs['train_config']['Max_Num_Words'],
        MAX_SEQUENCE_LENGTH = configs['train_config']['MAX_SEQUENCE_LENGTH'],
        LSTM_UNITS = configs['train_config']['LSTM_UNITS'],
        num_epoch = configs['train_config']['num_epoch'],
        batch_size = configs['train_config']['batch_size'],
        learning_rate = configs['train_config']['learning_rate'],
        accumulation_steps = configs['train_config']['accumulation_steps'],
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
    
    train_LSTM(t_config, p_config, s_config)

