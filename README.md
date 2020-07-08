# Jigsaw_Toxicity Competition Overview

Background: When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce unintended bias in machine learning models, and you'll help the Conversation AI team, and the entire industry, build models that work well for a wide range of conversations.

# Preprocessing

* Remove URLs
* Translate slangs and abbreviations (For LSTM only)
* Remove punctuations
* Translate words with '*'

# Loss

Some of my models predict 6 targets (1 main target and 5 toxicity labels). Some of my models predict 15 targets (1 main target + 5 toxicity labels + 9 indentity labels). All targets are soft targets instead of binary. Different targets have different weights. However, it is not clear how much improvement comes from the weighted target setting.

# Bert based models

Transfer learning from [huggingface](https://github.com/huggingface/transformers)'s pretrained BertForSequenceClassification model. Tried both bert-base-cased and bert-base-uncased models. Freeze the first 10 layers to reduce the total parameter size and speed up the training. Each model is trained for 2 epochs. Main hyperparameter changed for different models is the random seed.

The most important things learned during the implementation of Bert models are:

* Use [NVIDIA Apex](https://github.com/NVIDIA/apex) for more efficient training
* Bucket sequencing to clip sentences to longest sentence in the same batch. This speeds up both training and inference process.

For more details about the above operations, please refer to the [blog](https://medium.com/swlh/bridge-the-gap-between-online-course-and-kaggle-experience-from-jigsaw-unintended-toxicity-bias-6d4f638c4375#6561-f7a80a335237) here.

# LSTM models

Simple architecture which has been used in the kaggle community during the competition. Two consecutive birectional LSTMs followed by a couple of dense layers. We used word based embeddings here. From my experience this type of model requires more careful preprocessing so that more words in a sentence can be recognized. The vocab is constructed by concatenating cc.en.300.vec from fasttext and glove.6B.300d.txt from global vectors. The code is below:
```
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
```

# Final Solution

* 3 bert-base-cased models with 6-target classification head
* 1 bert-base-cased model with 15-target classification head
* 2 bert-base-uncased models with 6-target classification head
* 2 bert-base-uncased models with 15-target classification head
* 2 LSTM models with 200 LSTM hidden units
* 1 LSTM models with 400 LSTM hidden units

Final score is 0.94276 ranking 145/3165
