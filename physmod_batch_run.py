import numpy as np
import multiprocessing as mp
import subprocess
import sys
import json
import os

src_path = sys.argv[1]

with open('batch_config.json', 'r') as batch_file:
    batch_config = json.load(batch_file)

def sensitivity_worker(x):
    tid, sid = x
    print(tid, sid)
    respath = '%s\\%s_%s' % (batch_config['respath'], tid, sid)
    respath = respath.replace('/', '\\')
    os.makedirs(respath, exist_ok=True)
    subprocess.call('python %s\\physmod_initial_run.py %s %s' % (src_path, tid, sid), cwd=respath)
    
    os.chdir(respath)
    
    gosumd_path = batch_config['gosumd_exe']
    gosumd_path = gosumd_path.replace('/', '\\')
    subprocess.call('%s -sf %s\\%s_%s_physsens.txt' % (batch_config['gosumd_exe'], respath, tid, sid), cwd=respath)
    
    subprocess.call('python %s\\physmod_sens_postprocess.py' % (src_path))

def optimization_worker(x):
    tid, sid = x
    print(tid, sid)
    respath = '%s\\%s_%s' % (batch_config['respath'], tid, sid)
    respath = respath.replace('/', '\\')
    os.makedirs(respath, exist_ok=True)
    subprocess.call('python %s\\physmod_initial_run.py %s %s' % (src_path, tid, sid), cwd=respath)
    
    os.chdir(respath)
    
    gosumd_path = batch_config['gosumd_exe']
    gosumd_path = gosumd_path.replace('/', '\\')
    subprocess.call('%s -sf %s\\%s_%s_physopti.txt' % (batch_config['gosumd_exe'], respath, tid, sid), cwd=respath)

def optimization_worker_single(x):
    tid, sid = x

    with open('batch_config.json', 'r') as batch_file:
        batch_config = json.load(batch_file)

    print(tid, sid)
    respath = '%s\\%s_%s' % (batch_config['respath'], tid, sid)
    respath = respath.replace('/', '\\')

    os.chdir(respath)
    
    gosumd_path = batch_config['gosumd_exe']
    gosumd_path = gosumd_path.replace('/', '\\')
    print(gosumd_path, respath)
    subprocess.call('%s -sf %s\\%s_%s_physopti.txt' % (batch_config['gosumd_exe'], respath, tid, sid), cwd=respath)

if __name__ == '__main__': 
    sids = batch_config['subject_ids']
    tids = batch_config['task_ids']
    
    runargs = [ (task, subject) for task in tids for subject in sids ]
    with mp.Pool(batch_config['num_cores']) as p:
        print(p.map(sensitivity_worker, runargs))
        p.close()
        p.join()
        
    with mp.Pool(batch_config['num_cores']) as p:
        print(p.map(optimization_worker, runargs))
        p.close()
        p.join()
        
        
    