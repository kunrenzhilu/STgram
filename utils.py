import numpy as np
import datetime
import utils
import os
import json
import h5py
import matplotlib.pyplot as plt

normalize = lambda x: x/np.max((norm(x), np.ones(shape=(x.shape[0], 1))*1e-10))
norm = lambda x: np.sqrt(np.sum(np.square(x), axis=-1,keepdims=True))
l1norm = lambda x: np.sum(np.abs(x), axis=-1, keepdims=True)
l1normalize = lambda x: x/np.max((l1norm(x), 1e-10))

class DataStruct:
    def __init__(self):
        pass

def is_normalized_matrix(mat, epsilon=1e-6):
    indices = np.random.choice(mat.shape[0], size=10)
    return abs(np.max([norm(mat[i]) for i in indices]) - 1) < 1e-5

def replace_datetime_by_time_label(seq, pattern, args):
    assert pattern in ['day-6', 'day-24', 'weekly', 'week-end', 'hand']
    def digitize_datetime(dt, pattern):
        if pattern == 'day-6':
            bins = np.linspace(0,24,6+1)
            return np.digitize(list(map(lambda x:x.hour, dt)), bins)-1
        elif pattern == 'day-24':
            bins = np.linspace(0,24,24+1)
            return np.digitize(list(map(lambda x:x.hour, dt)), bins)-1
        elif pattern == 'weekly': #[0,6]
            bins = np.linspace(0,6,6+1)
            return np.digitize(list(map(lambda x:x.weekday(), dt)), bins)-1
        elif pattern == 'week-end':
            return np.array(list(map(lambda x:int(x.weekday()>4), dt)))
        elif pattern == 'hand': #[ 2.,  6., 10., 14., 18., 22., 26.])
            bins = np.linspace(0,24,6+1) + 2
            return np.digitize(list(map(lambda x:x.hour, dt)), bins) % 6
        else: return None
        
    def to_timestamp(dt):
        return np.array(list(map(lambda x:x.timestamp(), dt))).reshape(-1,1)
        
    tmp = np.array(seq)
    if args.WITH_TIMESTAMP:
        return np.concatenate([tmp[:,:-1],digitize_datetime(tmp[:,-1], pattern).reshape(-1,1), to_timestamp(tmp[:,-1])], axis=1)
    else:
        return np.concatenate([tmp[:,:-1],digitize_datetime(tmp[:,-1], pattern).reshape(-1,1)], axis=1)

def get_indices(args):
    count = 0
    indices = [0]
    idx = DataStruct()
    idx.LOC = len(indices)-1
    if args.WITH_USERID:
        indices.append(1)
        idx.USERID = len(indices)-1
    if args.WITH_GPS:
        indices.extend([2,3])
        idx.GEO = [len(indices)-2,len(indices)-1]
    if args.WITH_TIME:
        indices.append(4)
        idx.TIME = len(indices)-1
    if args.WITH_TIMESTAMP:
        indices.append(5)
        idx.TIMESTAMP = len(indices)-1
    print('indices setting : '+ ' '.join(list(map(lambda x:x[0], list(filter(lambda x: x[1] is True, args.__dict__.items()))))))
    return indices, idx

def extract_data(data, args):
    print('args.pattern: {}'.format(args.pattern))
    indices, idx = get_indices(args)
    print('indices: {}'.format(indices))
    
    res = []
    for seq in data:
        if args.WITH_TIME:
            res.append(replace_datetime_by_time_label(np.array(seq), args.pattern, args)[:,indices])
        else:
            res.append(np.array(seq)[:,indices])
    return res, idx

def save_args(args):
    with open(os.path.join(args.LOG_DIR, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print('Saved args to {}'.format(f.name))
    
def load_args(path):
    with open(path, 'r') as f:
        args = json.load(f)
    print(args)
    return args

def save_best_tf(saver, sess, args, state_dict, path=None):
    save_path = os.path.join(args.LOG_DIR, 'best', 'model.ckpt') if path is None else path
    saver.save(sess, save_path)
    state_path = os.path.join(save_path[:save_path.find('model.ckpt')], 'state.json')
    with open(state_path, 'w') as f:
        json.dump(state_dict, f, sort_keys=True, indent=4)
    print('Saved BEST model to {}'.format(save_path))

def save_model_tf(saver, sess, args, path=None):
    save_path = os.path.join(args.LOG_DIR, 'saved', 'model.ckpt') if path is None else path
    saver.save(sess, save_path)
    print('Saved model to {}'.format(save_path))
    
def load_model_tf(saver, args, sess, path=None):
    load_path = os.path.join(args.LOG_DIR, 'saved', 'model.ckpt') if path is None else path
    assert os.path.isfile(load_path+'.meta'), '{} is empty'.format(load_path)
    saver.restore(sess, load_path)
    return sess

def save_embeddings(path, mat):
    with h5py.File(path, 'w') as f:
        f.create_dataset('embeddings', data=mat, dtype=np.float32)
    print('Saved data to {}'.format(path))
    
def load_embeddings(path):
    with h5py.File(path, 'r') as f:
        data = f.get('embeddings').value
    print('Reading dadta from {}'.format(path))
    return data