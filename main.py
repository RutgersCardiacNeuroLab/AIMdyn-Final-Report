from physmod_initial_run import initial_run
from physmod_sens_postprocess import generate_initial_opti_files
from physmod_batch_run import optimization_worker_single
import sys

task_id = sys.argv[1]
subject_id = sys.argv[2]

initial_run(task_id, subject_id)
initial_run(task_id, subject_id)

params = ['Vblood', 'VLn', 'Csp', 'P0lv', 'Elvmax', 'Elvmin', 'Repmax', 'Repmin', 'GaTv', 'GaTs', 'Kelv', 'GpTv']
factors = [0.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3]

generate_initial_opti_files(params, factors, task_id, subject_id)
optimization_worker_single((task_id, subject_id))
