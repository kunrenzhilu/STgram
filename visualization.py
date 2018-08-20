from utils import l1normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
from errata import correct_errata

def get_ctgy_histogram_dict(traj_list, dicts):
    timebins = list(range(24))
    substats = {k:np.zeros_like(timebins) for k in dicts.ctgy_chkin_dict}
    rootstats = {k:np.zeros_like(timebins) for k in dicts.ctgy_mapping['ctgy_subctgy_dict']}

    for seq in traj_list:
        for p in seq:
            poi = p[0]
            if dicts.reverse_dictionary[poi] == 'UNK':
                continue
            sctgy = dicts.chkin_ctgy_dict[dicts.reverse_dictionary[poi]]
            rctgy = dicts.ctgy_mapping['subctgy_ctgy_dict'][correct_errata(sctgy)]
            visit_time = p[-1].hour
            substats[sctgy][visit_time] += 1
            rootstats[rctgy][visit_time] += 1
    return substats, rootstats

def normalize_dict(dicts):
    for k, v in dicts.items():
        dicts[k] = l1normalize(v)
    return dicts

def integrate_dicts(dict, keyword_list):
    res_dict = {kw: np.zeros(24) for kw in keyword_list}
    for k, v in dict.items():
        for kw in keyword_list:
            if kw in k:
                res_dict[kw] += dict[k]
    return res_dict

def plot_multiple(shape, history, figsize=None, suptitle=None):
    indices = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize)
    if len(axs.shape) == 1: axs = axs[None,:]
        
    modes = ['sub', 'root']
    ks = [1, 5, 10]
    items = ['accuracy', 'recall', 'precision', 'f1', 'distance_sub', 'distance_root']
    for i, item in enumerate(items):
        idx = indices[i]
        if 'distance' in item:
            axs[idx].plot(history[item])
        else:    
            legends = []
            for mode in modes:
                for k in ks:
                    key = '{}_{}_{}'.format(mode,item,k)
                    axs[idx].plot(history[key])
                    legends.append(key)
            axs[idx].legend(legends, loc='upper right')
        axs[idx].set_title(item)
    if not suptitle is None: 
        fig.suptitle(suptitle)
    plt.show()
    return fig

def plot_history(history, mode='precision', keys=None, title=None):
    plt.rcParams['figure.figsize'] = (10, 6)
    assert mode in ['precision', 'accuracy', 'both']
    x = range(len(list(history.values())[0]))
    if keys is None: keys = history.keys()
    def plot(keys, mode, history, plt):
        lgds = list()
        for k in keys:
            if k != 'n_epoch' and mode in k:
                plt.plot(x, history[k])
                lgds.append(k)
        plt.legend(lgds, loc='upper right')
#         plt.show()
    if not mode == 'both':
        plot(keys, mode, history, plt)
        if not title is None:
            plt.title(title)
    else: 
        fig, axs = plt.subplots(2,1)
        plot(keys, 'precision', history, axs[0])
        plot(keys, 'accuracy', history, axs[1])
        if not title is None:
            fig.suptitle(title)
        
def plot_distance(both_hist, title=None):
    plt.rcParams['figure.figsize'] = (10, 6)
    assert len(both_hist) == 2
    keys = ['distance_sub', 'distance_root']
    lgds = list()
    for mode in ['emb', 'weight']:
        for k in keys:
            x = range(len(both_hist[mode][k]))
            plt.plot(x, both_hist[mode][k])
            lgds.append('{}_{}'.format(mode, k))
    plt.legend(lgds, loc='upper right')
    if not title is None:
        plt.title(title)
    plt.show()
    
