"""
Main physiolgy model run script. Sets up the configuration and model objects
and runs the computaion.

Root Matlab reference code: Optimization_6P.m
"""
import sys
import logging

import numpy as np
from physmod_config import PhysmodConfig
from physmod_model import Physmod
from physmod_gosumd import PhysmodGosumd
import time
import os
import sys
import re


# python -i in -o out
input_file = sys.argv[2]
output_file = sys.argv[4]

match = re.search(r'/(\w{2})_(\w+)/physmod_gosumd_run', input_file)
addargs = [match.group(1), match.group(2)] 
#addargs = sys.argv[5].split(',')
#addargs = ['4P', '2650011441608878210']

tid, sid = addargs
start_time = time.time()
config = PhysmodConfig('%s/config.json' % (os.path.dirname(os.path.abspath(__file__))), tid, sid)

respath = config.respath


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='%s/log.txt' % respath,
                    filemode='a+')
logging.debug('working dir...%s', os.path.dirname(os.getcwd()))
logging.debug(addargs)
logging.debug(input_file)
logging.debug(output_file)
logging.debug('Input file exists: %s', os.path.isfile(input_file))
with open(input_file, 'r') as f:
    logging.debug(f.read())
os.chdir(config.respath)

model = Physmod(config)
logging.debug('Reading params...')
model.read_params(input_file)
logging.debug('Initializing state...')
model.init_state()
logging.debug('Setting Pthor...')
model.setup_pthor_cfg()
logging.debug('Predicting config...')
model.remove_predicting_cfg()
logging.debug('Prerun...')
model.prerun_cfg()
end_time = time.time()

start_calc = time.time()
logging.debug('Starting solver...')
solve_res = model.run()
end_calc = time.time()
end_time = time.time()

logging.debug('Postrun...')
model.postrun()
logging.debug('Frequency-amplitude...')
model.frequency_amplitude()
logging.debug('Calculating cost function...')
model.cost_function()

model.write_output_file(output_file)

logging.debug('Task: %s, FSID: %s, Total Time: %.2f, Calc. Time: %.2f, Input: %s' % (config.task_id, config.subject_id, end_time-start_time, end_calc-start_calc, input_file))
