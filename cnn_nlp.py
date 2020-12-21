# followed the tuturial from https://towardsdatascience.com/cnn-sentiment-analysis-9b1771e7cdd6
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim

import spacy
nlp = spacy.load('en')

import time
import matplotlib.pyplot as plt


import model



# Load Data

data = np.array(pd.read_csv('./preprocessed/withPunc.csv', header = None))

# X = data[:,0]
# Y = data[:,1]

train_data = torch.tensor(X[:7*len(Y)//10])

X_test = torch.tensor(X[7*len(Y)//10:])
Y_test = torch.tensor(Y[7*len(Y)//10:])



train_data, valid_data = train_data.split()



# Load embeddings

MAX_VOCAB_SIZE = 30_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create torch dataset and dataloader with batch size 50
trainset = TensorDataset(X_train, Y_train) 
trainloader = DataLoader(trainset, batch_size=50, shuffle=True) 

valset = TensorDataset(X_val, Y_val) 
valloader = DataLoader(valset,batch_size=50, shuffle=True) 




 # create 2-D matrices for word vs word embedding corresponding to that word

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(len(TEXT.vocab), 100, 100, [1, 2, 3, 4, 5], 2, 0.5, PAD_IDX)

model.embedding.weight.data.copy_(TEXT.vocab.vectors)

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model = model.to(device)


# function for accuracy
def accuracy(preds, y):
    correct = (torch.round(torch.sigmoid(preds)) == y).float() 
    return correct.sum() / len(correct)


## train the model
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



# define the optimizer
optimizer = optim.SGD(model.parameters())

# define the loss function
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)


# evaluate for the testing data 
test_loss, test_acc = evaluate(model, test_iterator, criterion)

