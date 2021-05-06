import numpy as np
import json
import os


class PhysmodConfig:
    """
    Container for external parameters necessary for running the physiology
    model. Loads and parses .json config file, loads the subject config data
    from the file pointed to by the config.
    """

    def __init__(self, config_file, tid, sid):
        
        self.task_id = tid
        self.subject_id = sid
        
        with open(config_file) as f:
            data = json.load(f)

        for key in data.keys():
            self.__setattr__(key, data[key])

        self.__dict__['subject_path'] = '%s/%s' % (data['data_path'], tid)

        subject_data = np.load('%s/%s.npz' % (self.subject_path, sid), allow_pickle=True)

        for key in subject_data.keys():
            self.__dict__[key] = subject_data[key]

        # Some type adjustments since npz always saves np.arrays
        self.__dict__['weight'] = self.__dict__['weight'].tolist()
        self.__dict__['height'] = self.__dict__['height'].tolist()
        self.__dict__['age'] = float(self.__dict__['age'].tolist())

        self.__dict__['BP_mean'] = np.mean(self.BP[1])
        self.__dict__['BP_small'] = self.BP[:, ::60]

        init_file = '%s/%s_%s_initial_state.npz' % (self.respath, self.task_id, self.subject_id)
        init_file_exists = os.path.isfile(init_file)
        if init_file_exists:
            print('Loading initial state...')
            init_state = np.load(init_file)
            self.__dict__['state0'] = init_state['state0']
            self.__dict__['init_time'] = float(init_state['init_time'])
        else:
            self.__dict__['init_time'] = -1.0
            
        self.respath = '%s/%s_%s' % (self.respath, self.task_id, self.subject_id)
        
        os.makedirs(self.respath, exist_ok=True)

    def save_init_state(self, state, init_time):
        init_file = '%s/%s_%s_initial_state.npz' % (self.respath, self.task_id, self.subject_id)
        init_file_exists = os.path.isfile(init_file)
        if not init_file_exists:
            np.savez_compressed(init_file, state0=state, init_time=init_time)
