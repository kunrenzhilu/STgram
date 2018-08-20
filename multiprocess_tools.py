import time
import multiprocessing as mp
import os
import numpy as np
import numpy.ma as ma
from parser import get_parser
from dataloader import load_data, DataLoader, DataLoader_time
from parser import get_parser
from errata import correct_errata
from collections import defaultdict
from utils import norm, normalize, extract_data, save_args, load_args, \
    save_embeddings, load_embeddings, DataStruct, save_model_tf

def map_compute_distance(args):
    if len(args) == 4:
        dist_matrix, dicts, mode, ctgy_chkin_dict = args
        queue = None
    else:   
        dist_matrix, queue, dicts, mode, ctgy_chkin_dict = args
    inner_dist = list()
    outer_dist = list()
    for i, dist in enumerate(dist_matrix):
        if mode == 'sub':
            same_ctgy_chkins = ctgy_chkin_dict[dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]]]
        else:
            same_ctgy_chkins = ctgy_chkin_dict[dicts.ctgy_mapping['subctgy_ctgy_dict'][
                    correct_errata(dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]])  ]]
        ids = [dicts.dictionary[key] for key in same_ctgy_chkins if key in dicts.dictionary.keys()]
        same_ctgy_mask = np.ones(dist_matrix.shape[1]).astype(int)
        same_ctgy_mask[ids] = 0
        diff_ctgy_mask = np.logical_not(same_ctgy_mask)
        same_dists = ma.masked_array(dist, mask=same_ctgy_mask)
        diff_dists = ma.masked_array(dist, mask=diff_ctgy_mask)
        inner_dist.append(np.mean(same_dists))
        outer_dist.append(np.mean(diff_dists))
    if not queue is None:
        queue.put((inner_dist, outer_dist))
    else: return inner_dist, outer_dist

def reduce_compute_distance(dist_list):
    inner_dist, outer_dist = [], []
    for inner, outer in dist_list:
        inner_dist.extend(inner)
        outer_dist.extend(outer)
    return inner_dist, outer_dist

def multiprocess_compute_distance(nProcess, dist_matrix, dicts, mode):
    assert mode in ['root', 'sub'], 'mode incorrect'
    if mode == 'root':
        ctgy_chkin_dict = dicts.rootctgy_chkin_dict
    else:
        ctgy_chkin_dict = dicts.ctgy_chkin_dict
    try:
        tick = time.time()
        pool = mp.Pool(processes=nProcess)
        chunks = np.array_split(dist_matrix, nProcess)
        results = pool.map(map_compute_distance, zip(chunks, [dicts]*nProcess, [mode]*nProcess, [ctgy_chkin_dict]*nProcess ))
        inner_dist, outer_dist = reduce_compute_distance(results)
        score = -(np.mean(inner_dist) - np.mean(outer_dist))
        pool.close()
        pool.join()
        print('Job Done, used time {}'.format(time.time()-tick))
        return score
    except:
        pool.close()
        pool.join()
        raise
'''
def multiprocessing_dist():
    chunks = np.split(dist_matrix, nProcess)
    queue = mp.Queue(5)
#     workers = {i:Process(target=map_compute_distance, 
#          args=(chunks[i], queue, dicts, mode, ctgy_chkin_dict)) for i in range(nProcess)}
    workers = {i:Process(target=f, args=(queue,)) for i in range(nProcess)}
    try:
        print('starting..')
        for w in workers.values():
            w.start()
        print('all_started')
        results = [queue.get() for _ in range(nProcess)]
        print('got results')
        score = reduce_compute_distance(results)   
        for w in workers.values():
            w.join(4)
            w.terminate()
        return score
    except:
        print('interupted')
        for w in workers.values():
            w.join(4)
            w.terminate()
'''

if __name__ == '__main__':
    args = get_parser(['--CITY', 'NYC', '--LOG_DIR', 'log_test', '--normalize_weight', '--WITH_TIME', '--WITH_GPS', '--WITH_TIMESTAMP', '--geo_reg_type', 'l2'])
    nProcess = 4
    origin_data, dicts = load_data(os.path.join(args.ROOT, 'data','{}_INTV_processed_voc5_len2_setting_WITH_GPS_WITH_TIME_WITH_USERID.pk'.format(args.CITY) ))
    dist_matrix = np.random.uniform(size=(5452, 5452))
    mode = 'root'
    if mode == 'root':
        ctgy_chkin_dict = dicts.rootctgy_chkin_dict
    else:
        ctgy_chkin_dict = dicts.ctgy_chkin_dict
    print(type(dicts.ctgy_mapping))
    ctgy_modes = ['sub', 'root']
    args.n_process = 4
    history = defaultdict(list)
    for cmode in ctgy_modes:
        history['{}_{}'.format(mode, cmode)].append(multiprocess_compute_distance(args, dist_matrix, dicts, cmode))