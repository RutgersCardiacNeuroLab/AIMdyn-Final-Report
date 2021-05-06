import numpy as np
import warnings
import sys
import os

class PhysmodGosumd:
    """
    Helper class for interfacing the Python physiology model implmenetation
    with GoSUMD v3.0
    """

    def __init__(self, model, wdir = None):
        
        if wdir is None:
            self.wdir = model.config.respath
        else:
            self.wdir = wdir
        self.model = model

    def write_gosumd_input(self, it):
        gosumd_input = '%s/in%i.txt' % (self.wdir, it)
        f = open(gosumd_input, 'w')
        # Create GOSUMD initial parameters file for sensitivity analysis
        for i, key in enumerate(self.model.param.param_names):
            p = self.model.param.params[key]
            f.write('%f\n' % (p))
        f.close()

    def write_gosumd_output(self, it):
        gosumd_output = '%s/out%i.txt' % (self.wdir, it)
        f = open(gosumd_output, 'w')
        # Create GOSUMD initial parameters file for sensitivity analysis
        f.write('%f' % (self.model.cost_res))
        f.close()

    def write_sensitivity_file(self, gosumd_initial, model_run_script):
        gosumd_script = '%s/%s_%s_physsens.txt' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        project_name = '%s-%s-physio-SA' % (self.model.config.task_id, self.model.config.subject_id)
        reduced_project_name = '%s-%s-physio-SA-reduced' % (self.model.config.task_id, self.model.config.subject_id)
        sample_type = 'dsample'
        sample_size = 300
        sensitivity_options = [1000, 0.005, 0.01, 0.01, 1]

        sens_file = open(gosumd_script, 'w')
        sens_file.write('set_project_name = %s\n' % (project_name))
        sens_file.write('set_project_path = %s\n' % (self.wdir))
        sens_file.write('set_project_type = modelanalysis\n')
        sens_file.write('import_inputs = %s\n' % (gosumd_initial))
        sens_file.write('set_resample_type = %s\n' % (sample_type))
        sens_file.write('set_resample_size = %i\n' % (sample_size))
        sens_file.write('resample_inputs\n')
        sens_file.write('set_core_size = 2\n')
        sens_file.write('set_python_path = %s\n' % (os.path.dirname(os.path.abspath(sys.executable))))
        sens_file.write('set_model_evaluator = pythonshell %s\n' % (model_run_script))
        sens_file.write('exe_add_args = %s,%s\n' % (self.model.config.task_id, self.model.config.subject_id))
        sens_file.write('evaluate_outputs\n')
        sens_file.write('set_mads_max_evaluation = 300\n')
        sens_file.write('save\n')
        sens_file.write('learn_model\n')
        sens_file.write('set_sensitivity_options = %s\n' % (' '.join([str(opt) for opt in sensitivity_options])))
        sens_file.write('compute_sensitivities\n')
        sens_file.write('export_variance_sensitivity = %s/%s_%s_varsens.txt\n' % (self.model.config.respath, self.model.config.task_id, self.model.config.subject_id))
        sens_file.write('save\n')
        sens_file.write('exit')
        sens_file.close()

    def write_optimization_file(self, opti_params_file, model_run_script):
        gosumd_script = '%s/%s_%s_physopti.txt' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        project_name = '%s-%s-physio-OA' % (self.model.config.task_id, self.model.config.subject_id)
        
        opti_hist_path = '%s/%s_%s_physopti_history' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        opti_const_path = '%s/%s_%s_gosumd_opti_constraints.txt' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        output_names_filepath = '%s/%s_%s_output_names.txt' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        output_names_file = open(output_names_filepath, 'w')
        if self.model.config.task_id == '6P' or self.model.config.task_id == '4P':
            out_names = ['HPL', 'HPH', 'HP', 'SAP', 'DAP', 'SAPT', 'DAPT', 'log_cost']
            objective_function = '1.5*HPL+0.5*HPH+2*HP+1.5*SAP+1.5*DAP+0.15*SAPT+0.15*DAPT'
        else:
            out_names = ['HPV', 'HPH', 'HP', 'SAP', 'DAP', 'SAPT', 'DAPT', 'log_cost']
            objective_function = 'HPV+0.5*HPH+2*HP+1.5*SAP+1.5*DAP+0.15*SAPT+0.15*DAPT'
            
        for i, n in enumerate(out_names):
            output_names_file.write('%i %s\n' % (i+1, n))
        output_names_file.close()

        optimization_method = 'mads'
        mads_max_evaluations = 250
        mads_init_mesh_size = 0.2
        mads_min_poll_size = 0.00001

        gosumd_script_file = open(gosumd_script, 'w')
        gosumd_script_file.write('set_project_name = %s\n' % (project_name))
        gosumd_script_file.write('set_project_path = %s\n' % (self.wdir))
        gosumd_script_file.write('set_project_type = modeloptimization\n')
        gosumd_script_file.write('import_inputs = %s\n' % (opti_params_file))
        gosumd_script_file.write('set_core_size = 1\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('set_python_path = %s\python.exe\n' % (os.path.dirname(os.path.abspath(sys.executable))))
        gosumd_script_file.write('set_model_evaluator = pythonshell %s\n' % (os.path.abspath(model_run_script)))
        gosumd_script_file.write('exe_add_args = %s,%s\n' % (self.model.config.task_id, self.model.config.subject_id))
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('rename_outputs = %s\n' % (output_names_filepath))
        gosumd_script_file.write('set_optimization_method = %s\n' % (optimization_method))
        gosumd_script_file.write('import_optimization_constraints = %s\n' % (opti_const_path))
        gosumd_script_file.write('set_objective = %s\n' % (objective_function))
        gosumd_script_file.write('set_mads_max_evaluation = %i\n' % (mads_max_evaluations))
        gosumd_script_file.write('set_mads_init_mesh_size = %f\n' % (mads_init_mesh_size))
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('minimize\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('export_optimization_history = %s_1.txt\n' % (opti_hist_path))
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('minimize\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('export_optimization_history = %s_2.txt\n' % (opti_hist_path))
        gosumd_script_file.write('minimize\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('export_optimization_history = %s_3.txt\n' % (opti_hist_path))
        gosumd_script_file.write('minimize\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('export_optimization_history = %s_4.txt\n' % (opti_hist_path))
        gosumd_script_file.write('minimize\n')
        gosumd_script_file.write('save\n')
        gosumd_script_file.write('export_optimization_history = %s_5.txt\n' % (opti_hist_path))
        gosumd_script_file.close()

    def write_optimization_parameters(self, lock_params):
        pass

    def write_paramater_file(self):
        gosumd_initial = '%s/%s_%s_gosumd_initial.txt' % (self.wdir, self.model.config.task_id, self.model.config.subject_id)
        f = open(gosumd_initial, 'w')

        # Create GOSUMD initial parameters file for sensitivity analysis
        for i, key in enumerate(self.model.param.param_names):
            p = self.model.param.params[key]
            lim_per = 0.1 * np.abs(p)
            lower_lim = p - lim_per
            upper_lim = p + lim_per
            if lower_lim >= upper_lim:
                warnings.warn('Lower limit greater or equal to upper limit in parameter: %s, %f  %f' % (key, lower_lim, upper_lim))
            f.write('%s\tuniformcontinuous\t%f\t%f\n' % (key, lower_lim, upper_lim))
        f.close()
        return gosumd_initial
