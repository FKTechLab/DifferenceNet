#!/usr/bin/env python
# coding: utf-8

from utils import *
from models import *
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import sys

def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Difference Net')
    
    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='abs', 
                        help='difference function to the model (abs/simple/square)')    
    parser.add_argument('--model_size', type=str, default='small', 
                        help='model architecture size (big/medium/small)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='training batch size (default: 8)')
    parser.add_argument('--iteration', type=int, default=200000, 
                        help='number of iterations (default: 200000)')
    
    # model hyper-parameters
    parser.add_argument('--input_type', type=str, default='all', 
                        help='input type, seek, serve or all')
    parser.add_argument('--num_train_plates', type=int, default=500,
                        help='number of training plates (default: 500)')
    parser.add_argument('--num_days', type=int, default=5,
                        help='number of training days (default: 5)')    
    
    # saving and loading directoreis
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--log_step', type=int , default=125)
    parser.add_argument('--checkpoint_every', type=int , default=5000)    
    return parser

def main(opts):
    """
    Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    create_dir(opts.log_path)
    create_dir(opts.model_path)
    model_path = opts.model_path

    # prepare logging file
    tag = str(opts.num_train_plates)+'plates_'+'days'+str(opts.num_days)+\
    '_inputs_'+str(opts.input_type)
    print(tag)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=opts.log_path+tag+'.log',
                        filemode='a')

    # load data
    all_plates = load_data(opts.data_path,'plates.pkl')
    all_plates.remove('d1329')
    train_plates = all_plates[:opts.num_train_plates]
    test_plates = all_plates[2000:] # 197 plates which are unseen

    if opts.input_type == 'all':
        input_type = ''
    else:
        input_type = opts.input_type
    
    with_speed = True
    with_profile = True

    iteration = opts.iteration
    model_size = opts.model_size
    mode = opts.mode
    num_days = opts.num_days
    batch_size = opts.batch_size

    raw_trajs = load_data(opts.data_path,'trajs_with_speed500.pkl')
    profile_data = load_data(opts.data_path,'profile_features500.pkl')

    siamese_net = build_model(with_speed,with_profile, num_days, mode, size=model_size)
    print(siamese_net.summary())

    # start training 
    loss_500 = []
    train_acc_500 = []
    val_acc_500 = []
    test_acc_500 = []
    
    # save best model
    current_train_acc = 0.7
    current_val_acc = 0.7
    current_test_acc = 0.7    
    
    t0 = datetime.now()
    t1 = datetime.now()

    # prepare evaluation dataset
    pairs,labels = [],[]
    val_pairs,val_labels = [],[]
    test_pairs,test_labels = [],[] 

    val_batch_data, val_batch_profile_data, val_label_list, _, _, _, _ = process_batch_data(raw_trajs, profile_data, train_plates, input_type, num_days, batch_size=1000)
    test_batch_data, test_batch_profile_data, test_label_list, _, _, _, _ = process_batch_data(raw_trajs, profile_data, test_plates, input_type, num_days, batch_size=1000)    

    max_time_step = 0
    pair_data_all = []
    label_list_all = []
    pair_profile_data_all = []
    time_start = time.time()

    for ite in range(iteration):

        batch_data, batch_profile_data, label_list, pair_data, label_list, pair_profile_data, time_step = process_batch_data(raw_trajs, profile_data, train_plates, input_type, num_days, batch_size=batch_size)
    
        max_time_step = max(max_time_step, time_step)
        pair_data_all.extend(pair_data)
        label_list_all.extend(label_list)
        pair_profile_data_all.extend(pair_profile_data)   

        loss = siamese_net.train_on_batch([batch_data, batch_profile_data], np.array(label_list, dtype=int))
        y_hat = siamese_net.predict([batch_data, batch_profile_data])                
        
        # save log
        if ite % opts.log_step == 0 and ite != 0:

            batch_data_train_acc, batch_profile_data_train_acc, label_list_train_acc = process_train_accuracy_check_batch(pair_data_all,label_list_all,pair_profile_data_all,max_time_step)
            y_hat = siamese_net.predict([batch_data_train_acc, batch_profile_data_train_acc])    
            train_acc,trp,trr,trf = calculate_metrics(y_hat, label_list_train_acc)    # old plates new days
            
            pair_data_all = []
            label_list_all = []
            pair_profile_data_all = []
                
            y_hat = siamese_net.predict([val_batch_data, val_batch_profile_data])    
            val_acc,vp,vr,vf = calculate_metrics(y_hat, val_label_list)    # old plates new days        
            y_hat = siamese_net.predict([test_batch_data, test_batch_profile_data])    
            test_acc,tep,ter,tef = calculate_metrics(y_hat, test_label_list)    # test with new plates and new days.
                    
            loss_500.append(loss)
            train_acc_500.append((train_acc,trp,trr,trf,))
            val_acc_500.append((val_acc,vp,vr,vf,))
            test_acc_500.append((test_acc,tep,ter,tef,))            
            print(
                '******iteration: '+str(ite)+'; loss: '+str(loss)+ \
                '; train acc: '+str(train_acc)+'; validation acc: '+ \
                str(val_acc)+'; test acc: '+str(test_acc)+ \
                '; test prf: '+str((tep,ter,tef,))
            )

            if train_acc > current_train_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_train')
                current_train_acc = train_acc
                print('best train model updated: ' + str(train_acc))
            if val_acc > current_val_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_val')
                current_val_acc = val_acc
                print('best validation model updated: ' + str(val_acc))
            if test_acc > current_test_acc:
                save_model(siamese_net, model_path, tag = tag+'_best_test')
                current_test_acc = test_acc
                print('best test model updated: ' + str(test_acc))
        if ite % opts.checkpoint_every == 0:
            save_model(siamese_net, model_path, tag = tag + '_mid_'+str(ite))
            pickle.dump(loss_500, open(model_path+'loss_{0}_{1}.pkl'.format(tag,str(ite)),'wb'))
            pickle.dump(train_acc_500, open(model_path+'train_acc_{0}_{1}.pkl'.format(tag,str(ite)),'wb'))
            pickle.dump(val_acc_500, open(model_path+'val_acc_{0}_{1}.pkl'.format(tag,str(ite)),'wb'))
            pickle.dump(test_acc_500, open(model_path+'test_acc_{0}_{1}.pkl'.format(tag,str(ite)),'wb'))            
    print('total running time: '+ str(datetime.now()-t0))
    save_model(siamese_net, model_path, tag = tag + '_iter'+str(iteration))
    pickle.dump(loss_500, open(model_path+'loss_{0}.pkl'.format(tag),'wb'))
    pickle.dump(train_acc_500, open(model_path+'train_acc_{0}.pkl'.format(tag),'wb'))
    pickle.dump(val_acc_500, open(model_path+'val_acc_{0}.pkl'.format(tag),'wb'))
    pickle.dump(test_acc_500, open(model_path+'test_acc_{0}.pkl'.format(tag),'wb'))

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    main(opts)
