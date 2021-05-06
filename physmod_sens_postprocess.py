import numpy as np
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from physmod_gosumd import PhysmodGosumd
import pickle

threshold = 20

def postprocess_sensitivity():
    with open('batch_config.json', 'r') as batch_file:
        batch_config = json.loads(batch_file.read())
    
    indices = []
    
    for tid in batch_config['task_ids']:
        for sid in batch_config['subject_ids']:
            sens_path = '%s/%s_%s/%s_%s_varsens.txt' % (batch_config['respath'], tid, sid, tid, sid)
            if not os.path.isfile(sens_path):
                print('Missing: %s_%s' % (tid, sid))
                continue
            df = pd.read_csv(sens_path, delimiter=',')
            df.drop(columns='Unnamed: 80', inplace=True)
        
            df.sort_values(by=7, axis=1, inplace=True)
            df = df.iloc[:, ::-1]
            
            indices.append(df.iloc[7, :threshold].index)
            
    indices = np.concatenate(indices)
    
    ind_values, ind_count = np.unique(indices, return_counts=True)
    return ind_values, ind_count
  
#'constantcontinuous'
def generate_initial_opti_files(params, factors, tid=None, sid=None):
    with open('batch_config.json', 'r') as batch_file:
        batch_config = json.loads(batch_file.read())
    
    #for tid in batch_config['task_ids']:
    #    for sid in batch_config['subject_ids']:
    model_file = '%s/%s_%s/%s_%s_model.p' % (batch_config['respath'], tid, sid, tid, sid)
    
    if not os.path.isfile(model_file):
        print(model_file)
        print('Missing: %s_%s' % (tid, sid))
        return
    with open(model_file, 'rb') as model:
        model = pickle.load(model)
    par_list = []
    constraints = []
    for par in model.param.params.keys():
        if par in params and not par in model.param._const_keys:
            p = model.param.params[par]
            lim_per = factors[params.index(par)] * np.abs(p)
            lower_lim = max(0.0, p - lim_per)
            upper_lim = p + lim_per
            print(par, lower_lim, upper_lim)
            par_list.append('%s uniformcontinuous %f %f' % (par, lower_lim, upper_lim))
            constraints.append(par)
        else:
            par_list.append('%s constantcontinuous %f' % (par, model.param.params[par]))
            
    param_file = '%s/%s_%s/%s_%s_gosumd_initial_opti.txt' % (batch_config['respath'], tid, sid, tid, sid) 
    constraints_file = '%s/%s_%s/%s_%s_gosumd_opti_constraints.txt' % (batch_config['respath'], tid, sid, tid, sid) 
    with open(constraints_file, 'w') as f:
        f.write('OBJECTIVE_HERE\n')
        for c in constraints:
            f.write('%s > 0\n' % (c))

    with open('%s/%s_%s/%s_%s_gosumd_initial_opti.txt' % (batch_config['respath'], tid, sid, tid, sid), 'w') as opti_init_file:
        opti_init_file.write('\n'.join(par_list))
        
    gosumd = PhysmodGosumd(model, '%s/%s_%s' % (batch_config['respath'], tid, sid))
    gosumd.write_optimization_file(param_file, 'physmod_gosumd_run.py')
    
if __name__ == '__main__':
    vals, cnts = postprocess_sensitivity()
    cnt_idx = np.argsort(cnts)[::-1]
    cnts = cnts[cnt_idx]
    vals = vals[cnt_idx]
    
    thresh = np.percentile(cnts, 64)
    opti_params = vals[cnts > thresh]
    print(opti_params)
    print(len(opti_params))
    
    generate_initial_opti_files(opti_params)
    
    