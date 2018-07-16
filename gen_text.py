#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 20:40:20 2018

@author: ethan
"""

import argparse
import pickle
import numpy as np
from keras.models import load_model

def predict(rnn, init_text, nwords, mode='random'):     
    encoded_seq = tokenizer.texts_to_sequences(init_text)[0]
    for idx in range(nwords):
        encoded = np.array([encoded_seq])
        p_next_word = rnn.predict(encoded)[0][0]
        if mode == 'max':
            next_word = np.argmax(p_next_word)
        elif mode == 'random':
            next_word = np.random.choice(a=p_next_word.shape[-1], p=p_next_word)
        encoded_seq.append(next_word)
        if word_lookup[next_word] == '<e>':
            break
    pred_ending = [word_lookup[word] for word in encoded_seq]
    return ' '.join(pred_ending)


parser = argparse.ArgumentParser(description='Generate text from language model')

parser.add_argument('--model')
parser.add_argument('--tokenizer')
parser.add_argument('--nwords', default=30, type=int)
parser.add_argument('--nsents', default=4, type=int)
parser.add_argument('--init_text', default='kitty kisses the', type=str)
parser.add_argument('--mode', default='max', choices=['max', 'random'])
args = parser.parse_args()

rnn = load_model(args.model)
with open(args.tokenizer, 'wb') as handle:
    tokenizer = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)

word_lookup = {idx: word for word, idx in tokenizer.word_index.items()}

for _ in range(args.nsents):
    print(predict(rnn, ['<s> ' + args.init_text], nwords=args.nwords, mode=args.mode))
    print('\n')
