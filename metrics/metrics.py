import numpy as np
import time
import torch 
from sklearn.metrics import roc_auc_score

def auc_mp_fast(y_true, y_pred, num_class=None, multi_label=False):
    '''
    the goal of this fuc is to compute the one.vs.all (one.vs.rest) AUC scores for multi-class AUC optimization
    
    '''
    if y_true.max() == 1 and y_pred.shape[1] == 2:
        y_pred = y_pred[:,1]

    if multi_label:
        res = dict()
        for i in range(y_pred.shape[1]):
            res[str(i)] = roc_auc_score(y_true=y_true[:,i], y_score=y_pred[:,i])
        auc_mean = sum([res[i] for i in res.keys()]) / len(res)
        res['mean'] = auc_mean
        return res

    return roc_auc_score(y_true = y_true, y_score = y_pred, multi_class = 'ovr')

def test():

    print("======> run test!!!") 
    np.random.seed(19270817)
    y_pred = np.random.rand(100, 5000)
    y_pred /= (y_pred.sum(1)).reshape((-1, 1))
    y_true = np.random.rand(100, 5000)
    y_true = (y_true == np.max(y_true, 1).reshape((-1, 1))).astype(np.int32)

    print(y_pred.shape)
    print(y_true.shape)
    
    t0 = time.time()
    auc_std = 0 #auc_m(y_true, y_pred)
    t1 = time.time()
    auc_fast = auc_mp_fast(y_true, y_pred)
    t2 = time.time()

    print(auc_std, auc_fast)
    print('fast: %.4f,  naive: %.4f'%(t2 - t1, t1 - t0))

if __name__ == '__main__':
    test()
