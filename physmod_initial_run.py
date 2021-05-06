"""
Main physiolgy model run script. Sets up the configuration and model objects
and runs the computaion.

Root Matlab reference code: Optimization_6P.m
"""

import numpy as np
from physmod_config import PhysmodConfig
from physmod_model import Physmod
from physmod_gosumd import PhysmodGosumd
import time
import os
import matplotlib.pyplot as plt
import sys


def initial_run(task_id, subject_id):
    start_time = time.time()
    config = PhysmodConfig('%s/config.json' % (os.path.dirname(os.path.abspath(__file__))), task_id, subject_id)

    model = Physmod(config)
    model.init_param()
    model.init_state()
    model.setup_pthor_cfg()
    model.remove_predicting_cfg()
    model.prerun_cfg()
    end_time = time.time()
    print('Config time: %.2f s' % (end_time-start_time))

    solve_res = model.run()
    end_time = time.time()
    print('Run time: %.2f s' % (end_time-start_time))

    model.postrun()

    gosumd = PhysmodGosumd(model)
    gosumd_initial_file = gosumd.write_paramater_file()
    gosumd.write_sensitivity_file(gosumd_initial_file, '%s/physmod_gosumd_run.py' % (os.path.dirname(os.path.abspath(__file__))))
    gosumd.write_optimization_file(gosumd_initial_file, '%s/physmod_gosumd_run.py' % (os.path.dirname(os.path.abspath(__file__))))
    end_time = time.time()
    print('Post time: %.2f s' % (end_time-start_time))

    end_time = time.time()
    print('Elapsed time: %.2f s' % (end_time-start_time))

    print('Saving model...')
    model.save_model()

if __name__ == '__main__':
    task_id = sys.argv[1]
    subject_id = sys.argv[2]
    initial_run(task_id, subject_id)
