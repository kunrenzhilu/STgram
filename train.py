seed = 1
import numpy as np
import random
np.random.seed(seed)
random.seed(seed)
import os 
import csv
import sys
import pickle
import time
import h5py
from collections import defaultdict
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy.ma as ma
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.1

from dataloader import load_data, DataLoader, DataLoader_time
from parser import get_parser
from evaluation import Evaluator
from utils import norm, normalize, is_normalized_matrix, extract_data, save_args, load_args, \
    save_embeddings, load_embeddings, DataStruct, save_model_tf, save_best_tf, load_model_tf
from logger import Logger
from model import init_params, crossentropy, choose_emb, choose_geo_loss, STSkipgram
from multiprocess_tools import multiprocess_compute_distance

def extract(list, indices):
    if type(list) is int:
        return list
    if type(indices) is int:
        return list[indices]
    else:
        return np.array(list)[indices]

def get_forward_dataset(trajs, forecast_step=1):
    all_data = list()
    for seq in trajs:
        if len(seq) < forecast_step + 1:
            continue
        else:
            for i in range(len(seq) - 1):
                end = i+forecast_step if (i+forecast_step) < len(seq)\
                    else len(seq)-1
                for j in range(i,end):
#                     all_data.append([seq[i],seq[j]])
                    all_data.append((seq[i], seq[j+1]))
    return np.array(all_data)

def get_backward_dataset(trajs, backward_step=1):
    all_data = list()
    for seq in trajs:
        if len(seq) < backward_step + 1:
            continue
        else:
            for i in reversed(range(len(seq))):
                head = i - backward_step if (i-backward_step) > 0 else 0
                for j in range(head, i):
#                     all_data.append([seq[i],seq[j]])
                    all_data.append((seq[i], seq[j]))
    return np.array(all_data)

def get_train_data(data, mode='both', ws=2):
    # ws stands for window size
    assert mode in ['forward', 'backward', 'both']
    res = []
    if (mode == 'foward') or (mode == 'both'):
        res.append(get_forward_dataset(data, ws))
    if (mode == 'backward') or (mode == 'both'):
        res.append(get_backward_dataset(data, ws))
    log = 'Mode:{}, '.format(mode) + ' '.join(['size:{}'.format(d.shape) for d in res])
    res = np.concatenate(res)
    log += ' total size:{}'.format(res.shape)
    print(log)
    return res

class BestCriteria:
    def __init__(self, metrics):
        self.best_score = 0
        self.metrics = metrics
    def should_save(self, result, ):
        tmp = list()
        for m in self.metrics:
            tmp.append(result[m])
        score = np.mean(tmp)
        if score > self.best_score:
            self.best_score = score
            return True
        return False

def update(losses, sk, geo, t):
    assert type(losses) is dict, 'losses is expected to be dict'
    losses['geo'].append(geo)
    losses['skipgram'].append(sk)
    losses['time'].append(t)
    return losses

def compute_weight_decay(t1, t2, temp):
    return np.exp(-1*((t1-t2)/60*temp)**2)

def evaluate(emb, evaluator):
    result = evaluator.evaluate(emb)
    evaluator.update_history(res_dict=result)
    evaluator.save_history()
    return result

def train(graph, sess, model, args, evaluator_emb, evaluator_weight, logger, dataloader, dataloader_time):
    save_args(args)
    losses = {'geo':[], 'skipgram':[], 'time':[]}
    n_batch = 0
    n_epoch = 0
    tick0 = time.time()
    
    best_criteria = BestCriteria(['{}_f1_{}'.format(mode, k) for mode in ['sub', 'root'] for k in args.Ks])
    with graph.as_default():
        saver = tf.train.Saver(model.all_params)
        if args.resume:
            sess = load_model_tf(saver, args, sess)
            evaluator.load_history(args)
        else:
            logger.renew_log_file()
            sess.run(tf.global_variables_initializer())
        logger.log('\nStart training')
        
        while dataloader.get_epoch() < args.num_epoch:
            if args.normalize_weight:
                _ = sess.run([model.normalize_geo_emb_op, model.normalize_sem_emb_op,
                              model.normalize_geo_wht_op, model.normalize_sem_wht_op])

            epoch_tick = time.time()
            emb, weight = sess.run([model.sem_emb, model.sem_wht])
            result_emb = evaluate(emb, evaluator_emb)
            result_weight = evaluate(weight, evaluator_weight)
            result = result_emb if args.main_emb == 'emb' else result_weight
            save_model_tf(saver, sess, args)
            if best_criteria.should_save(result):
                tmp = dict(result)
                tmp['epoch'] = n_epoch
                tmp['batch'] = n_batch
                save_best_tf(saver, sess, args, {'args':vars(args), 'result':tmp})
            #-- Optimization steps 
            while n_epoch >= dataloader.get_epoch():
                center, context = next(dataloader.dg)
                sk_loss, _, geo_loss, _ = sess.run([model.weighted_skipgram_loss, model.train_skipgram, model.geo_loss, model.train_geo],
                          {model.center_loc:center.ids, 
                           model.label_loc:context.ids.reshape(-1,1),
                           model.weight_decay: compute_weight_decay(center.timestmp, context.timestmp, args.time_temp),
                           model.coor_center:center.coors, 
                           model.coor_label:context.coors})
                
                loc, time_label = next(dataloader_time.dg)
                t_loss, _ = sess.run([model.time_loss, model.train_t],
                         {model.center_loc:loc, model.label_t:time_label})
                
                losses = update(losses, sk=sk_loss, geo=geo_loss, t=t_loss)
                
                if n_batch % 100 == 0:
                    losses = {k:np.mean(v) for k, v in losses.items()}
                    evaluator_emb.update_history(losses=losses)
                    evaluator_weight.update_history(losses=losses)
                    logstr = '[{}] LOSS '.format(n_batch) + "".join(['{} : {:.6f} '.format(k, v) for k, v in losses.items()])
                    losses = {'geo':[], 'skipgram':[], 'time':[]}
                    logger.log(logstr)
                    
                n_batch += 1
            #-----------------------
            n_epoch += 1
            logstr = '#'*50+'\n'
            logstr += 'Ecpoh {}, used time: {}, eval: {}'.format(n_epoch, time.time()-epoch_tick, result)
            logger.log(logstr)
    logger.log('FINISH, USED TIME:{}'.format(time.time()-tick0))
    return sess

if __name__=='__main__':
    tick = time.time()
    args = get_parser(sys.argv[1:])
#    args = get_parser(['--CITY', 'NYK', '--LOG_DIR', 'log', 
#                   '--WITH_TIME', '--normalize_weight'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    data, dicts = load_data(os.path.join(args.ROOT, 'data','{}_INTV_processed_voc5_len2_setting_WITH_GPS_WITH_TIME_WITH_USERID.pk'.format(args.CITY) ))
    args.vocabulary_size = dicts.vocabulary_size
    data, idx = extract_data(data, args) #put all data_extraction here
    train_data = get_train_data(data)
    
    dataloader = DataLoader(train_data, args)
    dataloader_time = DataLoader_time(data, args, idx)
    evaluator_emb = Evaluator(args, dicts, mode='emb')
    evaluator_weight = Evaluator(args, dicts, mode='weight')
    logger = Logger(os.path.join(args.LOG_DIR, 'log_txt'))

    graph = tf.Graph()
    with graph.as_default():
        model = STSkipgram(args)
        sess = tf.Session(graph=graph, config=config)
    state = train(graph, sess, model, args, evaluator_emb, evaluator_weight, logger, dataloader, dataloader_time)    
    sess.close()
    print('Done, used time: {}'.format(time.time()-tick))
