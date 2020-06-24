import torch
import numpy as np
from transformers import BertTokenizer
import random

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

def get_tokenized_samples(max_seq_length, tokenizer, texts):

    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_seq_length - 2]
        input_sequence = ['[CLS]'] + text + ['[SEP]']
        pad_len = max_seq_length - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        
        all_tokens.append(tokens)
    return np.array(all_tokens)

def resort_index(ids_train, num_of_bucket, seed):
    
    num_of_bucket = 2
    num_of_element = int(len(ids_train)/num_of_bucket)
    ids_train_new = ids_train.copy()
    
    for i in range(num_of_bucket):
        
        copy = ids_train[i*num_of_element:(i+1)*num_of_element].copy()
        random.Random(seed).shuffle(copy)
        ids_train_new[i*num_of_element:(i+1)*num_of_element] = copy

    return ids_train_new

def clip_to_max_len(batch):
    
    inputs, Target, Target_aux, Target_identity, weight, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    
    return inputs[:, :max_len], Target, Target_aux, Target_identity, weight

class train_config:
    
    def __init__(self, num_to_load, valid_size, num_labels, 
                 MAX_SEQUENCE_LENGTH, num_of_bucket, num_epoch, 
                 batch_size, learning_rate, accumulation_steps, warmup, PATH='./'):
        self.num_to_load = num_to_load
        self.valid_size = valid_size
        self.num_labels = num_labels
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.num_of_bucket = num_of_bucket
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        self.warmup = warmup
        self.PATH = PATH

