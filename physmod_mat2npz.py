# physmod_mat2npz.py - v2.0 - Aleksandr Andrejčuk
# An utility script for picking up the subject data and translating it to .npz
# format consumable by Python physiology code.
# v2.0  - reading subject data from text files defined by task and subject ID
#       - creates a dictionary for each subject-task combo

import numpy as np
import pandas as pd
import os
import sys
import warnings


data_path = 'd:/_projects/aimdyn-physiology_data/all_data/timeseries_data/'
init_file = 'd:/_projects/aimdyn-physiology_data/all_data/init_21S.txt'
hw_file = 'd:/_projects/aimdyn-physiology_data/all_data/CGE Measurements.xlsx'
output_path = 'd:/_projects/aimdyn-physiology_data/npz'
subject_ids_file = '%s/subject_ids.txt' % (data_path)

hw = pd.read_excel(hw_file)
hw['uid'] = hw['ID'].map(str) + hw['Session'].map(str)
hw.set_index('uid', inplace=True)

def file_exists(fname):
    ret = os.path.isfile(fname)
    if not ret:
        print('File does not exist:')
        print(fname)
        return False
    else:
        return True


def parse_data_type(fname):
    if 'RRI' in fname:
        return 'HP'
    elif 'Respiration Volume' in fname:
        return 'TR'
    elif 'DAP' in fname:
        return 'DAP'
    elif 'SAP' in fname:
        return 'SAP'
    elif 'Blood Pressure' in fname:
        return 'BP'


def process_respiration(TR):
    TR = TR[::50]
    return TR


init_state = np.loadtxt(init_file)
# Parse all files in the data_path directory (expects to have only data files i.e. no dirs)
data_file_list = os.listdir(data_path)
ALL_IDS = []
for data_file in data_file_list:
    root_id = data_file[:19]
    ALL_IDS.append(root_id)
ALL_IDS = set(ALL_IDS)

# Setting up the root dictionary containing all subject ids
root_dict = {id: {'age': 0, 'weight': 0, 'height': 0, 'sex': 'F'} for id in ALL_IDS}

for i, data_file in enumerate(data_file_list):
    subject_id = data_file[:21]
    task_id = subject_id[-2:]
    root_id = subject_id[:-2]
    age = subject_id[7:9]
    uid = subject_id[:6] + subject_id[6]

    if uid in hw.index:
        height = hw.loc[uid, 'Height (cm)']
        weight = hw.loc[uid, 'Weight (kg)']
    else:
        height = 170
        weight = 70
        warnings.warn('%s uid is not in height-weight file, setting default' % (uid))

    root_dict[root_id]['age'] = age
    root_dict[root_id]['height'] = height
    root_dict[root_id]['weight'] = weight

    if task_id not in root_dict[root_id].keys():
        root_dict[root_id][task_id] = {}

    data_type = parse_data_type(data_file)
    data = np.loadtxt('%s/%s' % (data_path, data_file))
    if data_type == 'TR':
        data = process_respiration(data)
    root_dict[root_id][task_id][data_type] = data.T
    print(i, len(data_file_list), subject_id, task_id, data_type)

used_task_ids = ['B1', '4P', '6P']
for root_key in root_dict.keys():
    weight = root_dict[root_key]['weight']
    height = root_dict[root_key]['height']
    age = root_dict[root_key]['age']
    sex = root_dict[root_key]['sex']
    print(height, weight)
    
    root_dt = {}
    # Compute dts
    for tid in used_task_ids:
        if tid not in root_dict[root_key].keys():
            continue
        TR = root_dict[root_key][tid]['TR']
        TR[0] = TR[0] - TR[0,0]
        HP = root_dict[root_key][tid]['HP']
        HP[0] = HP[0] - HP[0,0]
        
        root_dt['dt_%s' % (tid)] = HP[0,0] - TR[0,0] 
    
    for tid in used_task_ids:
        if tid not in root_dict[root_key].keys():
            continue
        
        # Normalizing time vector for each variable
        HP = root_dict[root_key][tid]['HP']
        HP[0] = HP[0] - HP[0,0]
        BP = root_dict[root_key][tid]['BP']
        BP[0] = BP[0] - BP[0,0]
        DAP = root_dict[root_key][tid]['DAP']
        DAP[0] = DAP[0] - DAP[0,0]
        SAP = root_dict[root_key][tid]['SAP']
        SAP[0] = SAP[0] - SAP[0,0]
        TR = root_dict[root_key][tid]['TR']
        TR[0] = TR[0] - TR[0,0]
        
        np.savez_compressed('%s/%s/%s' % (output_path, tid, root_key),
        subject_id=root_key,
        weight=weight,
        height=height,
        sex=sex,
        age=age,
        state0=init_state,
        HP=HP,
        BP=BP,
        DAP=DAP,
        SAP=SAP,
        TR=TR,
        dt=root_dt)
