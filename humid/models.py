# coding: utf-8

import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from tensorflow.keras.layers import TimeDistributed, Masking, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import pickle
from datetime import datetime
import sys
import logging
import os

def difference(x1, x2, mode='simple'):
    '''Difference Layer: supports 3 types (simple, abs, square)'''
    if mode == 'simple':
        return x1 - x2
    elif mode == 'abs':
        return tf.abs(x1 - x2)
    elif mode == 'square':
        return tf.square(x1 - x2)
    else:
        raise NotImplementedError

def seq_lstm(size):
    lstm = Sequential()
    if size == 'big':
        lstm.add(LSTM(200, return_sequences=True))
        lstm.add(LSTM(100))
        lstm.add(Dense(48, activation='relu'))    
    elif size == 'medium':
        lstm.add(LSTM(128, return_sequences=True))
        lstm.add(LSTM(64))
        lstm.add(Dense(32, activation='relu'))
    else:
        lstm.add(LSTM(80, return_sequences=True))
        lstm.add(LSTM(40))
        lstm.add(Dense(16, activation='relu'))
    return lstm

def seq_embedding():
    embed = Sequential()
    embed.add(TimeDistributed(Dense(16, activation='relu')))
    return embed

def seq_profile():
    # model profile features representation nn
    profile = Sequential()
    profile.add(Dense(64,activation='relu',name='dense_profile_1'))
    profile.add(Dropout(0.3))
    profile.add(Dense(32,activation='relu',name='dense_profile_2'))
    profile.add(Dropout(0.2))
    profile.add(Dense(8,activation='relu',name='dense_profile_3'))
    return profile

def seq_similarity(size):
    prediction = Sequential()
    if size != 'small':
        prediction.add(Dense(64,activation='relu'))
        prediction.add(Dropout(0.1))
        prediction.add(Dense(32,activation='relu'))
        prediction.add(Dropout(0.1))
        prediction.add(Dense(8,activation='relu'))
    else:
        prediction.add(Dense(16,activation='relu',name='dense_sim_1'))
        prediction.add(Dropout(0.1))
        prediction.add(Dense(8,activation='relu',name='dense_sim_2'))
    prediction.add(Dense(1,activation='sigmoid',name='output'))
    return prediction

def build_model(with_speed, with_profile, num_days, mode='abs', size='small'):
    if with_speed:
        inputs1_d1s = [Input((None,4)) for _ in range(num_days)] 
        inputs1_d1d = [Input((None,4)) for _ in range(num_days)] 
        inputs1_d2s = [Input((None,4)) for _ in range(num_days)] 
        inputs1_d2d = [Input((None,4)) for _ in range(num_days)] 
    else:
        inputs1_d1s = [Input((None,3)) for _ in range(num_days)] 
        inputs1_d1d = [Input((None,3)) for _ in range(num_days)] 
        inputs1_d2s = [Input((None,3)) for _ in range(num_days)] 
        inputs1_d2d = [Input((None,3)) for _ in range(num_days)]
    
    seq_masking = Masking(mask_value=0.)

    # embed
    seq_embed = seq_embedding()
    
    embed_d1s = [seq_embed(seq_masking(ip)) for ip in inputs1_d1s] 
    embed_d1d = [seq_embed(seq_masking(ip)) for ip in inputs1_d1d]
    embed_d2s = [seq_embed(seq_masking(ip)) for ip in inputs1_d2s] 
    embed_d2d = [seq_embed(seq_masking(ip)) for ip in inputs1_d2d]

    inputs1_ds = [difference(x1, x2, mode= mode) for x1, x2 in zip(embed_d1s, embed_d2s)]
    inputs1_dd = [difference(x1, x2, mode= mode) for x1, x2 in zip(embed_d1d, embed_d2d)]

    # build up model
    # model two LSTM1
    seq_lstm1 = seq_lstm(size)
    # model two LSTM2
    seq_lstm2 = seq_lstm(size)
    # similarity nn
    seq_sim = seq_similarity(size)

    # input to lstm
    lstm1_ds = [seq_lstm1(traj_input) for traj_input in inputs1_ds]
    lstm1_dd = [seq_lstm2(traj_input) for traj_input in inputs1_dd]

    # get trip embeddings
    trip_emb = concatenate(lstm1_ds+lstm1_dd)

    # one day one driver has one profiel features
    if with_profile:
        # inputs 2: profile feature
        inputs2_d1 = Input((11,)) 
        inputs2_d2 = Input((11,)) 
        inputs2 = difference(inputs2_d1, inputs2_d2, mode= mode)
        # model profile features representation nn
        seq_pro = seq_profile()
        # get profile embeddings
        pro_emb = seq_pro(inputs2)

        # concatenate xyt(v) and profile 
        cat = concatenate([trip_emb]+[pro_emb])
    else:
        cat = trip_emb
        
    # merge input and output
    inputs_tmp = inputs1_d1s+inputs1_d1d+inputs1_d2s+inputs1_d2d
    if with_profile:
        inputs_tmp_2 = [inputs2_d1, inputs2_d2]
        
    # similarity nn for learning xyt and profile together
    prediction = seq_sim(cat)
    
    # training process
    siamese_net = Model(inputs=[inputs_tmp,inputs_tmp_2], outputs=prediction)

    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
    return siamese_net


