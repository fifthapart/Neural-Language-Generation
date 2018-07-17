#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:33:48 2018

@author: ethan
"""

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import pickle
import spacy
import argparse

def train_epoch(model, tokenized):  
    losses = []
    for sent in tokenized:
        sent = numpy.array(sent)
        sent_x = sent[None, :-1]
        sent_y = sent[None, 1:, None]
        loss = model.train_on_batch(x=sent_x, y=sent_y)
        losses.append(loss)
    loss = numpy.mean(losses)
    return loss

parser = argparse.ArgumentParser(description='Train RNN language model')
parser.add_argument('--train-file', default='your_text_here.txt',
                    help='text file to train LSTM')
parser.add_argument('--embedding-dim', default=300, type=int)
parser.add_argument('--hidden-dim', default=300, type=int)
parser.add_argument('--epochs', default=1, type=int)
args = parser.parse_args()

nlp = spacy.load('en', disable=['tagger', 'ner', 'textcat'])

START = '<s> '
END = ' <e>'
FILTERS = '’!”#$%&()*+,-./:;=?@[\\]^`{|}~\t\n’'

with open(args.train_file) as f:
    text = f.read()

doc = nlp(text)
sents = [START + sent.text + END for sent in doc.sents]

tokenizer = Tokenizer(lower=True, filters=FILTERS)
tokenizer.fit_on_texts(sents)
tokenized = tokenizer.texts_to_sequences(sents)

vocab_size = len(tokenizer.word_index)
batch_size = 1
n_timesteps = None
rnn = Sequential()

embedding_layer = Embedding(batch_input_shape=(batch_size, n_timesteps),
                            input_dim=vocab_size + 1, #add 1 because word indices start at 1, not 0
                            output_dim=args.embedding_dim, 
                            mask_zero=True)
rnn.add(embedding_layer)

recurrent_layer1 = LSTM(units=args.hidden_dim, return_sequences=True)
rnn.add(recurrent_layer1)

recurrent_layer2 = LSTM(units=args.hidden_dim, return_sequences=True)
rnn.add(recurrent_layer2)

pred_layer = TimeDistributed(Dense(vocab_size + 1, activation="softmax"))
rnn.add(pred_layer)

rnn.compile(loss="sparse_categorical_crossentropy", 
            optimizer='adam')

print("Training RNN on", len(tokenized), "sentences for", args.epochs, "epochs...")
for epoch in range(args.epochs):
    loss = train_epoch(rnn, tokenized)
    print("epoch {} loss: {:.3f}".format(epoch + 1, loss))
    
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
rnn.save('rnn_lm.h5')

