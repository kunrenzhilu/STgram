import numpy as np
from errata import correct_errata
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
import numpy.ma as ma
import sys
import time
import os
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle
from utils import norm, normalize, is_normalized_matrix
from multiprocess_tools import multiprocess_compute_distance

def get_labels(dicts):
    subctgys = sorted(dicts.ctgy_chkin_dict.keys())
    subctgy_dictionary = {k:i for i, k in enumerate(subctgys)}
    sublabels = [subctgy_dictionary[dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]]] for i in range(dicts.vocabulary_size-1)]
    rootctgys = sorted(dicts.rootctgy_chkin_dict.keys())
    rootctgy_dictionary = {k:i for i, k in enumerate(rootctgys)}
    rootlabels = [rootctgy_dictionary[dicts.ctgy_mapping['subctgy_ctgy_dict'][correct_errata(dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]])]] for i in range(dicts.vocabulary_size-1)]
    return {'sub':sublabels, 'root':rootlabels}

def f1_score(p, r):
    return 2*p*r/(p+r)

def get_same_ctgy_chkins(i, dicts, mode):
    if mode == 'root':
        ctgy_chkin_dict = dicts.rootctgy_chkin_dict
        same_ctgy_chkins = ctgy_chkin_dict[dicts.ctgy_mapping['subctgy_ctgy_dict'][
            correct_errata(dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]])  ]]
    else:
        ctgy_chkin_dict = dicts.ctgy_chkin_dict
        same_ctgy_chkins = ctgy_chkin_dict[dicts.chkin_ctgy_dict[dicts.reverse_dictionary[i]]]
    return same_ctgy_chkins

def compute_max_relevent_len(valid_ids, dicts):  
    relevent_lens_dict = {'sub':[], 'root':[]}
    for i in valid_ids:
        for mode in ['sub', 'root']:
            relevent_lens_dict[mode].append(len(get_same_ctgy_chkins(i, dicts, mode)))
    return {k:np.array(v) for k, v in relevent_lens_dict.items()}

def get_distance_score(dist_matrix, dicts, mode):
    #if not cap by 1000, then the average speed is 11s for 'root' mode and 3s for 'sub' mode for the NYK dataset
    assert mode in ['root', 'sub'], 'mode incorrect'
    # The following assertations are must, since otherwise the rootctgy_chkin_dict would got changed       
    inner_dist = list()
    outer_dist = list()
    for i, dist in enumerate(dist_matrix):
        same_ctgy_chkins = get_same_ctgy_chkins(i, dicts, mode)
        ids = [dicts.dictionary[key] for key in same_ctgy_chkins if key in dicts.dictionary.keys()]
        same_ctgy_mask = np.ones(dist_matrix.shape[1]).astype(int)
        same_ctgy_mask[ids] = 0
        diff_ctgy_mask = np.logical_not(same_ctgy_mask)
        same_dists = ma.masked_array(dist, mask=same_ctgy_mask)
        diff_dists = ma.masked_array(dist, mask=diff_ctgy_mask)
        inner_dist.append(np.mean(same_dists))
        outer_dist.append(np.mean(diff_dists))
        #if i > 1000: break
    #The smaller the inner value the better, the larger the score the better
    score = -(np.mean(inner_dist) - np.mean(outer_dist))
    return score.round(6)

def get_scores(top_matches, dicts, valid_ids, mode='score', Ks=[1,5,10], relevent_lens_dict=None):
    assert mode in ['score', 'matrix']
    K=max(Ks)
    ctgy_mapping = dicts.ctgy_mapping
    #----------------------------------#
    def compute_score(mat, mode, maxlens=None, max_recalls=None):
        if mode == 'accuracy':
            # Coase Grain, hit = min(sum(mat[i], axis=1), 1)
            return np.mean(np.sum(mat, axis=1) > 0)
        if mode == 'precision':
            # Finer Grain, multiple hits are distinguished hits.
            return np.mean(np.mean(mat, axis=1))
        if mode == 'recall':
            recalls = np.sum(mat, axis=1)/maxlens
            relative_recalls = recalls
            return np.mean(relative_recalls)
        else: raise ValueError('Mode {} is not supported'.format(mode))
    def id2subctgy(dicts, idx):
        return dicts.chkin_ctgy_dict[dicts.reverse_dictionary[idx]]
    def id2rootctgy(dicts, ctgy_mapping, idx):
        return ctgy_mapping['subctgy_ctgy_dict'][correct_errata(dicts.chkin_ctgy_dict[dicts.reverse_dictionary[idx]])]
    #-----------------------------------#
    
    sub_match_mat = []
    root_match_mat = []
    for i in range(top_matches.shape[0]):
        sub_match_mat.append(
            [id2subctgy(dicts, int(top_matches[i,j])) == id2subctgy(dicts, int(valid_ids[i])) for j in range(K)]
        )
        root_match_mat.append(
            [id2rootctgy(dicts, ctgy_mapping, int(top_matches[i,j])) == 
                          id2rootctgy(dicts, ctgy_mapping, int(valid_ids[i])) for j in range(K)]
        )
    sub_match_mat = np.array(sub_match_mat)
    root_match_mat = np.array(root_match_mat)
    match_mat_dict = {'sub':sub_match_mat,'root':root_match_mat}
    
    if mode == 'matrix':
            return root_match_mat, sub_match_mat
    else:
        results = dict()
#         for k in [1,5,10]:
        for k in Ks:
            for ctgy in ['root', 'sub']:
                for mode in ['accuracy', 'precision', 'recall']:
                    score = compute_score(match_mat_dict[ctgy][:,:k], mode, relevent_lens_dict[ctgy], k/relevent_lens_dict[ctgy])
                    results['{}_{}_{}'.format(ctgy, mode, k)] = score.round(6)
                results['{}_{}_{}'.format(ctgy, 'f1', k)] = f1_score(p=results['{}_{}_{}'.format(ctgy, 'precision', k)], r=results['{}_{}_{}'.format(ctgy, 'recall', k)])
    return results

class Evaluator(object):
    def __init__(self, args, dicts, mode, valid_ids=None, tflogger=None):
        self.log_dir = args.LOG_DIR
        self.mode = 'emb' if mode is None else mode
        self.dicts = dicts
        self.valid_ids = range(args.vocabulary_size-1) if valid_ids is None else valid_ids
        self.history = defaultdict(list)
        self.tflogger = tflogger
        self.nProcess = args.n_processes
        self.relevent_lens_dict = compute_max_relevent_len(self.valid_ids, self.dicts)
        self.label_dict = get_labels(self.dicts) 
        self.K = max(args.Ks)
        self.Ks = args.Ks
        
    def reset(self,):
        self.history = defaultdict(list)
    
    def evaluate(self, embed):
        assert embed.shape[0] == self.dicts.vocabulary_size, 'embedding shape {}, vocab size {}'.format(embed.shape[0], self.dicts.vocabulary_size)
        embed = embed[self.valid_ids]
        tick = time.time()
        ctgy_modes = ['root', 'sub']
        if not is_normalized_matrix(embed):
            embed = normalize(embed)
        distance_mat = 1-np.matmul(embed, embed.T)
        
        # evaluate distance:    
        print('eval distance')
        distance_mat = 1-np.matmul(embed, embed.T) #get ride of the 'UNK'
        for cmode in ctgy_modes:
            self.history['distance_{}'.format(cmode)].append(
                multiprocess_compute_distance(self.nProcess, distance_mat, self.dicts, cmode))
            
        #evaluate silhouette_score and calinski_harabaz_score
        for cmode in ctgy_modes:
            self.history['silhouette_{}'.format(cmode)].append(silhouette_score(distance_mat, self.label_dict[cmode], metric='cosine'))
            self.history['harabaz_{}'.format(cmode)].append(calinski_harabaz_score(embed, self.label_dict[cmode]))
            
        # evaluate translation:
        mode = 'score'
        valid_emb = embed[self.valid_ids]
        scores = valid_emb.dot(embed.T)
        top_matches = (-scores).argsort()[:,1:self.K+1]
        res_dict = get_scores(top_matches, self.dicts,
                             self.valid_ids, mode, Ks=self.Ks, relevent_lens_dict=self.relevent_lens_dict)
        res_dict['t'] = time.time()-tick
        return res_dict
            
    def update_history(self, losses=None, res_dict=None):
        if not losses is None:
            for k, v in losses.items():
                self.history['loss_{}'.format(k)].append(v)
        if not res_dict is None:
            for k, v in res_dict.items():
                self.history[k].append(v)
                if not self.tflogger is None:
                    self.tflogger.scalar_summary(tag=k, value=v, step=len(self.history[k]))
        
    def save_history(self):
        save_path = os.path.join(self.log_dir, '{}_history.pk'.format(self.mode))
        with open(save_path, 'wb') as f:
            pickle.dump(self.history, f)
        print('saved history to {}'.format(save_path))
    
    def load_history(self, args):
        path = os.path.join(args.LOG_DIR, '{}_history.pk'.format(self.mode))
        print('load history from {}'.format(path))
        with open(path, 'rb') as f:
            self.history = pickle.load(f)
            
    @staticmethod
    def load(path):
        print('load history from {}'.format(path))
        with open(path, 'rb') as f:
            return pickle.load(f)
