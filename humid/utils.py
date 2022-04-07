#!/usr/bin/env python
# coding: utf-8

import pickle
import random
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def load_data(path,filename):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(path+filename, 'rb')
    data = pickle.load(file)
    file.close() 
    return data

def expand(data):
    '''
    Expand dimension for data
    traj.shape == (num_states,num_features) ==> traj.shape == (1,num_states,num_features)
    '''
    data = np.expand_dims(data, axis=0)
    return data 

def get_days(ispos = True, num_days = 5,new_days = False):
    '''
    Return two days for positive or negative pair
    Args:
    ----
    ispos: boolean
        True: positive 
        False: negative
    num_days: int
        1 to 9
    new_days: boolean
        True: new days
        False: old days
    Return:
    -------
    
    '''
    # print(" -- ", ispos, num_days, new_days)
    if ispos:
        if not new_days:
            pos_rnd_days = random.sample(range(num_days),2) # randomly pick up 2 days from the first num_days days. 
            return pos_rnd_days
        else:
            if num_days == 9:
                pos_rnd_days = [8,9]
                return pos_rnd_days
            else:
                pos_rnd_days = random.sample(range(num_days,10),2) # randomly pick up 2 days from the rest days (new).
                return pos_rnd_days
    else:
        neg_rnd_days = []
        if not new_days:
            for _ in range(2):
                neg_rnd_days.extend(random.sample(range(num_days),1)) #  randomly pick up 1 day twice from first num_days days. 
            return neg_rnd_days
        else:
            for _ in range(2):
                neg_rnd_days.extend(random.sample(range(num_days,10),1)) #  randomly pick up 1 day twice from the rest days(new). 
            return neg_rnd_days

def get_trajs(data, rnd_plt, rnd_day, input_type, num_days):
    # print(" --- ", len(data), rnd_plt, rnd_day, input_type, num_days)
    '''
    Args:
    -----
    data: list of trajs (xyt or xytv)
    rnd_plt: str
    rnd_day: int
    input_type: 'seek' or 'serve'
    Return:
    -------
    trajs: 5 traj in the list
    '''
    # find trajs of the plate and the day and the type
    trajs = data[rnd_plt][rnd_day][input_type]
    # randomly select 5 trajectories for each day
    trajs = random.sample(trajs,num_days)
    return trajs

def merge_data(t,d1_profile,d2_profile):
    """
    merge all data, final step for preparing pair
    Args:
    ------
    t: list. [] or xyt(v) list
    d1_profile: list
    d2_profile: list
    Return:
    -------
    inputs: merged data
    """
    t.append(d1_profile)
    t.append(d2_profile)
    inputs = np.array(t)
    inputs = [expand(data) for data in inputs]
    return inputs
    
def get_pairs_s_or_d(data, profile_data, plates, input_type, num_days = 5, new_days = False):
    '''
    Get input pairs and label, batch_size = 1
    For postive pair,
        randomly select one driver and two days
        randomly select 5 seeking or 5 driving
    For negtive pair,
        randomly select two drivers and two days
        randomly select 5 seeking or 5 driving
    
    seek: xyt+(v)+(profile)
    drive: xyt+(v)+(profile)
    
    Args:
    -----
    data:
        xyt+v data
    profile_data:
        11 dimension profile features, one vector for one day one driver
    plates:
        plates containing enough data
    input_type:
        'seek' or 'serve'
    num_days:
        number of days to train
    new_days:
        flag, if use new days to test or validation
    Return:
    -------
        randomly return an input and lable, either positive or negative pair and label
        (input,label)
    '''
    # 0.5 probability to return positive pair
    if random.random()<=0.5: 
        # postive pair
        pos_rnd_days = get_days(ispos = True, num_days = num_days, new_days = new_days)
        pos_rnd_plt = random.sample(plates,1) # randomly pick up one driver
        
        d1 = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], input_type, num_days)
        d2 = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], input_type, num_days)
        
        if len(profile_data)>0: # if using profile data, add to the pair
            d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
            d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
            t_pos = d1+d2
            inputs_pos = merge_data(t_pos,d1_profile_pos,d2_profile_pos)
            
        else: # otherwise only using xyt(v) trajs as pair
            t_pos = d1+d2
            inputs_pos = [expand(data) for data in np.array(t_pos)]
            
        return inputs_pos,[0]
    
    else:
        # negative pair
        neg_rnd_days = get_days(ispos = False, num_days = num_days, new_days = new_days)
        neg_rnd_plt = random.sample(plates,2) # randomly pick up two drivers
        
        nd1 = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], input_type, num_days)
        nd2 = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], input_type, num_days)
        
        if len(profile_data)>0:
            d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
            d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
            t_neg = nd1+nd2
            inputs_neg = merge_data(t_neg,d1_profile_neg,d2_profile_neg)
        else:
            t_neg = nd1+nd2
            inputs_neg = [expand(data) for data in np.array(t_neg)]
            
        return inputs_neg,[1]

def get_pairs_s_and_d(data, profile_data, plates, input_type='', num_days = 5, new_days = False):
    '''
    Get input pairs and label, batch_size = 1
    For postive pair,
        randomly select one driver and two days
        randomly select 5 seeking and 5 driving
    For negtive pair,
        randomly select two drivers and two days
        randomly select 5 seeking and 5 driving
    
    seek+drive: xyt
    seek+drive: xyt+v
    seek+drive: xyt+v+profile
    seek+drive: profile
    
    using plates and new_days to return validation and test dataset
    
    Args:
    -----
    data:
        xyt+(v) data
    profile_data:
        11 dimension profile features, one vector for one day one driver
    plates:
        plates containing enough data
    input_type:
        keep consistant with get_pairs_s_or_d, furture used in _acc
    num_days:
        number of days to train
    new_days:
        flag, if use new days to test or validation
    Return:
    -------
        randomly return an input and lable, either positive or negative pair and label
        (input,label)
    '''
    # 0.5 probability to return positive pair
    if random.random()<=0.5: 
        # postive pair
        pos_rnd_days = get_days(ispos = True, num_days = num_days, new_days = new_days)
        pos_rnd_plt = random.sample(plates,1) # randomly pick up one driver
        if len(data)>0:# if using trajs data
            d1s = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], 'seek', num_days)
            d1d = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[0], 'serve', num_days)
            d2s = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], 'seek', num_days)
            d2d = get_trajs(data, pos_rnd_plt[0], pos_rnd_days[1], 'serve', num_days)

            if len(profile_data)>0: # if using profile data, add to the pair
                d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
                d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
                t_pos = d1s + d1d + d2s + d2d
                inputs_pos = merge_data(t_pos,d1_profile_pos,d2_profile_pos)

            else: # otherwise only using xyt(v) trajs as pair
                t_pos = d1s + d1d + d2s + d2d
                inputs_pos = [expand(data) for data in np.array(t_pos)]
                
        else: # if only using profile data            
            d1_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[0]]
            d2_profile_pos = profile_data[pos_rnd_plt[0]][pos_rnd_days[1]]
            inputs_pos = merge_data([],d1_profile_pos,d2_profile_pos)
            
        return inputs_pos,[0]
    
    else:
        # negative pair
        neg_rnd_days = get_days(ispos = False, num_days = num_days, new_days = new_days)
        neg_rnd_plt = random.sample(plates,2) # randomly pick up two drivers
        if len(data)>0:
            nd1s = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], 'seek', num_days)
            nd1d = get_trajs(data, neg_rnd_plt[0], neg_rnd_days[0], 'serve', num_days)
            nd2s = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], 'seek', num_days)
            nd2d = get_trajs(data, neg_rnd_plt[1], neg_rnd_days[1], 'serve', num_days)

            if len(profile_data)>0:
                d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
                d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
                t_neg = nd1s + nd1d + nd2s + nd2d
                inputs_neg = merge_data(t_neg,d1_profile_neg,d2_profile_neg)
            else:
                t_neg = nd1s + nd1d + nd2s + nd2d
                inputs_neg = [expand(data) for data in np.array(t_neg)]
        else:
            d1_profile_neg = profile_data[neg_rnd_plt[0]][neg_rnd_days[0]]
            d2_profile_neg = profile_data[neg_rnd_plt[1]][neg_rnd_days[1]]
            inputs_neg = merge_data([],d1_profile_neg,d2_profile_neg)
            
        return inputs_neg,[1]

def save_model(net, model_path, tag):
    # serialize model to JSON
    model_json = net.to_json()
    with open(model_path+"model_{0}.json".format(tag), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net.save_weights(model_path+"model_{0}.h5".format(tag))
    print("Saved model to disk")

def get_batch(raw_trajs, profile_data, train_plates, input_type, num_days,  batch_size = 8):
    # generate raw batch data      
    pair_data,label_list = [],[]
    pair_profile_data = []
    for item in range(batch_size):
        pair,label = get_pairs_s_and_d(raw_trajs, profile_data, train_plates, input_type, num_days)
        pair_data.append(pair[:num_days*4])
        label_list.extend(label)
        pair_profile_data.append(pair[num_days*4:])        
    return pair_data, label_list, pair_profile_data

def get_max_time_step(pair_data):
    max_len = 0
    for each in pair_data:
        for element in each:
            max_len = max(max_len, element.shape[1])
    return max_len

def pad_zeros_all(pair, maxlen, with_profile = False):
    # pad batch data with the maximum timestep length in the batch
    
    def np_pad(arr, maxlen):
        return np.pad(
            arr, 
            [(0, 0), (maxlen-arr.shape[1] if maxlen-arr.shape[1] > 0 else 0, 0), (0, 0)], 
            mode='constant', 
            constant_values=0
        )
    pair = [np_pad(arr, maxlen) for arr in pair]
    return pair

def process_batch_data(raw_trajs, profile_data, train_plates, input_type, num_days, batch_size):
    # pre-process batch data to feed to the model
    
    pair_data, label_list, pair_profile_data = get_batch(raw_trajs, profile_data, train_plates, input_type,num_days,  batch_size)
    max_time_step = get_max_time_step(pair_data)
    with_profile = True
    for data_num in range(len(pair_data)):
        pair_data[data_num] = pad_zeros_all(pair_data[data_num], max_time_step, with_profile)
    for data_num in range(len(pair_data)):
        pair_data[data_num] = np.stack( pair_data[data_num] )
    for data_num in range(len(pair_profile_data)):
        pair_profile_data[data_num] = np.stack( pair_profile_data[data_num] )        

    batch_data = np.stack( pair_data , axis = -3)
    batch_profile_data = np.stack( pair_profile_data , axis = -3)
    batch_data = np.squeeze(batch_data, axis=1)
    batch_profile_data = np.squeeze(batch_profile_data, axis=2)
    batch_data = list(batch_data)
    batch_profile_data = list(batch_profile_data)
    return batch_data, batch_profile_data, label_list, pair_data, label_list, pair_profile_data, max_time_step

def process_train_accuracy_check_batch(pair_data, label_list, pair_profile_data, max_time_step):
    # aggregate training data everytime model is trained on 1000 data points
    
    with_profile = True
    for data_num in range(len(pair_data)):
        pair_data[data_num] = pad_zeros_all(pair_data[data_num], max_time_step, with_profile)
    for data_num in range(len(pair_data)):
        pair_data[data_num] = np.stack( pair_data[data_num] )
    for data_num in range(len(pair_profile_data)):
        pair_profile_data[data_num] = np.stack( pair_profile_data[data_num] )        

    batch_data = np.stack( pair_data , axis = -3)
    batch_profile_data = np.stack( pair_profile_data , axis = -3)
    batch_data = np.squeeze(batch_data, axis=1)
    batch_profile_data = np.squeeze(batch_profile_data, axis=2)
    batch_data = list(batch_data)
    batch_profile_data = list(batch_profile_data)
    ## batch_data = batch_data.swapaxes(0, 2)            
    return batch_data, batch_profile_data, label_list
    
def calculate_metrics(y_hat, y):
    y_hat = y_hat.flatten()
    y_hat[y_hat<0.5] = 0
    y_hat[y_hat>=0.5] = 1
    accuracy = accuracy_score(list(y_hat), y)
    p = round(precision_score(y, y_hat, average='weighted'), 2)
    r = round(recall_score(y, y_hat, average='weighted'), 2)
    f = round(f1_score(y, y_hat, average='weighted'), 2)        
    return accuracy, p, r, f
