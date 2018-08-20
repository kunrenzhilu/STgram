seed = 1
import numpy as np
import random
np.random.seed(seed)
random.seed(seed)
import os
import h5py
import pickle
from collections import defaultdict
from errata import correct_errata
from utils import DataStruct

ROOT = '/home/haibin2/data/checkins/data'
CITY = 'NYC'

class Dictionary(object):
    def __init__(self,):
        pass

def load_category_mapping(path=None):
    if path is None:
        path = os.path.join('data', 'category_mappings.pk')
    with open(path, 'rb') as f:
        both_dict = pickle.load(f)
    return both_dict

def get_rootctgy_chkin(ctgy_chkin_dict, subctgy_ctgy_dict):
    newdict = defaultdict(list)
    for key in ctgy_chkin_dict:
        newdict[subctgy_ctgy_dict[correct_errata(key)]].extend(ctgy_chkin_dict[key])
    return newdict
    
def load_data(data_path=None):
    if data_path == None:
        data_path = os.path.join(ROOT, '{}_DAY_processed_data_voc5_len2_wo_time.pk'.format(CITY))
    print('loading data from {}'.format(data_path))
    
    dicts = Dictionary()
    with open(data_path, 'rb') as f:
        data_from_dump = pickle.load(f)
    data = data_from_dump['data']
    dicts.dictionary = data_from_dump['dictionary']
    dicts.reverse_dictionary = data_from_dump['reverse_dictionary']
    dicts.ctgy_len_arrays = data_from_dump['ctgy_len_arrays']
    dicts.vocabulary_size = data_from_dump['vocabulary_size']
    dicts.chkin_ctgy_dict = data_from_dump['chkin_ctgy_dict']
    dicts.ctgy_chkin_dict = data_from_dump['ctgy_chkin_dict']
    dicts.chkin_len_arrays = data_from_dump['chkin_len_arrays']
    dicts.ctgy_mapping = load_category_mapping(os.path.join(ROOT, 'category_mappings.pk'))
    dicts.rootctgy_chkin_dict = get_rootctgy_chkin(dicts.ctgy_chkin_dict, dicts.ctgy_mapping['subctgy_ctgy_dict'])
    
    #keep the dictionary's key set fixed
    for k, v in dicts.__dict__.items():
        if type(v) == defaultdict:
            v.default_factory = None
    return data, dicts

def group_data_by_id(data):
    res = defaultdict(list)
    for seq in data:
        res[seq[0][1]].append(seq)
    res.default_factory = None
    sorted_key = sorted(res.keys(), key=lambda x: len(res[x]), reverse=True)
    return res, sorted_key

class DataLoader:
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.dg = self.data_generator(data, self.batch_size)
        self.reset()
        
    def __split__(self, data):
        assert data.shape[-1] == 5
        ds = DataStruct()
        ds.ids = data[:,0].astype(int)
        ds.coors = data[:,1:3].astype(float)
        ds.label_t = data[:,3].astype(int)
        ds.timestmp = data[:,4].astype(float)
        return ds
    
    def reset(self,):
        self.n_epoch = -1
        self.n_iter = 0
        
    def get_epoch(self,):
        return self.n_epoch
    
    def get_iter(self):
        return self.n_iter
    
    def data_generator(self, data, batch_size):
        while True:
            np.random.shuffle(data)
            self.n_epoch += 1
            for i in range(0, len(data)-batch_size+1,batch_size):
                self.n_iter += 1
                chunk = data[i:i+batch_size]
                center, context = np.split(chunk, 2, axis=1)
                center = np.squeeze(center)
                context = np.squeeze(context)
                center = self.__split__(center)
                context = self.__split__(context)
                yield center, context

class DataLoader_time:
    def __init__(self, data, args, idx):
        self.batch_size = args.batch_size
        self.data = np.concatenate(data)[:,[idx.LOC,idx.TIME]]
        self.dg = self.data_generator(self.data, self.batch_size)
        self.reset()
        
    def reset(self,):
        self.n_epoch = -1
        self.n_iter = 0
        
    def get_epoch(self,):
        return self.n_epoch
    
    def get_iter(self):
        return self.n_iter
    
    def data_generator(self, data, batch_size):
        while True:
            np.random.shuffle(data)
            self.n_epoch += 1
            for i in range(0, len(data)-batch_size+1,batch_size):
                self.n_iter += 1
                chunk = data[i:i+batch_size]
                loc, time_label = np.hsplit(chunk, 2)
                yield np.squeeze(loc), np.squeeze(time_label)