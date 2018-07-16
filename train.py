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
import numpy
import pickle
import spacy
import argparse
              
def clean(text):
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines]
    poem = ' '.join(cleaned)
    sents = poem.split('.')
    return [s for s in sents if len(s) != 0]

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
parser.add_argument('--train-file', default='/home/ethan/the_last_answer.txt',
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

lexicon_size = len(tokenizer.word_index)
batch_size = 1
n_timesteps = None
rnn = Sequential()

embedding_layer = Embedding(batch_input_shape=(batch_size, n_timesteps),
                            input_dim=lexicon_size + 1, #add 1 because word indices start at 1, not 0
                            output_dim=args.embedding_dim, 
                            mask_zero=True)
rnn.add(embedding_layer)

recurrent_layer1 = LSTM(units=args.hidden_dim, return_sequences=True)
rnn.add(recurrent_layer1)

recurrent_layer2 = LSTM(units=args.hidden_dim, return_sequences=True)
rnn.add(recurrent_layer2)

pred_layer = TimeDistributed(Dense(lexicon_size + 1, activation="softmax"))
#add 1 because word indices start at 1, not 0
rnn.add(pred_layer)

rnn.compile(loss="sparse_categorical_crossentropy", 
            optimizer='adam')

print("Training RNN on", len(tokenized), "sentences for", args.epochs, "epochs...")
for epoch in range(args.epochs):
    loss = train_epoch(rnn, tokenized)
    print("epoch {} loss: {:.3f}".format(epoch + 1, loss))
    
with open('/home/ethan/newest_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
rnn.save('/home/ethan/rnn_lm.h5')

