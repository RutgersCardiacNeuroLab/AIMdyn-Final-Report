import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy as sp


class PhysmodVis:
    """
    Module for visualizing physiology models.
    """

    def __init__(self):
        self._model = None

    def import_model(self, model_file):
        """
        Imports calculated model from a pickled file.
        """
        with open(model_file, 'rb') as model:
            self._model = pickle.load(model)
            
        self._model.frequency_amplitude()
            
    def plot_psa(self, exp=True, mask=None):
        plt.figure()
        plt.plot(self._model.state.tout, self._model.state.get_vector('Psa'))
        
        if exp:
            fbp = sp.interpolate.interp1d(self._model.config.BP[0][::60], self._model.config.BP[1][::60], kind='cubic')
            expbp = fbp(self._model.state.tout)
            plt.plot(self._model.state.tout, expbp, '--')
            if mask is not None:
                plt.xlim(mask[0], mask[1])
        plt.show()
    
    def plot_sap(self, exp=True, mask=None):
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        
        tt0 = self._model.config.t_begin
        Fs = 100
        
        time = dap_t
        sap_ind = sp.signal.argrelextrema(psa, np.greater)
        sap_t = self._model.state.tout[sap_ind] + self._model.config.t_begin
        t0 = max(tt0, np.ceil(time[0]*Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        sap_model = psa[sap_ind]
        fyy = sp.interpolate.interp1d(sap_t, sap_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)

        dt = self._model.config.dt.item()['dt_%s' % (self._model.config.task_id)]
        ttt1 = self._model.config.SAP[0] + dt
        sap_spline = sp.interpolate.interp1d(ttt1, self._model.config.SAP[1], kind='cubic')
        sap_exp = sap_spline(tt)
        
        plt.figure()
        plt.plot(self._model.state.tout+self._model.config.t_begin, psa)
        plt.plot(tt, yy)
        plt.plot(tt, sap_exp, '--')
        plt.show()
    
    def plot_dap(self, exp=True, mask=None):
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        
        tt0 = self._model.config.t_begin
        Fs = 100
        
        time = dap_t
        t0 = max(tt0, np.ceil(time[0]+Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        dap_model = psa[dap_ind]
        fyy = sp.interpolate.interp1d(time, dap_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)

        dt = self._model.config.dt.item()['dt_%s' % (self._model.config.task_id)]
        ttt1 = self._model.config.DAP[0] + dt
        dap_spline = sp.interpolate.interp1d(ttt1, self._model.config.DAP[1], kind='cubic')
        dap_exp = dap_spline(tt)
        
        plt.figure()
        plt.plot(self._model.state.tout+self._model.config.t_begin, psa)
        plt.plot(tt, yy)
        plt.plot(tt, dap_exp, '--')
        plt.show()
    
    def plot_hp(self, exp=True):
        Fs = 100.0
        dw = 0.25
        
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        hp_model = 1000 * np.diff(dap_t)
        
        time = dap_t[:-1]
        t0 = max(self._model.config.t_begin, np.ceil(time[0]+Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        fyy = sp.interpolate.interp1d(time, hp_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)

        hp_spline = sp.interpolate.interp1d(self._model.config.HP[0], self._model.config.HP[1], kind='cubic')
        hp_exp = hp_spline(tt)
        
        plt.figure()
        plt.plot(tt, yy)
        if exp:
            plt.plot(tt, hp_exp, '--')
        plt.show()
        
    def plot_state_variable(self, var):
        plt.figure()
        plt.plot(self._model.state.tout+self._model.config.t_begin, self._model.state.get_vector(var))
        plt.show()
        
    def plot_hp_psd(self):        
        Fs = 100.0
        dw = 0.25
        
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        hp_model = 1000 * np.diff(dap_t)
        
        time = dap_t[:-1]
        t0 = max(self._model.config.t_begin, np.ceil(time[0]+Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        fyy = sp.interpolate.interp1d(time, hp_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)
        hp_model_fq, hp_model_amp = self._model.psd_welch(yy, Fs, dw)
        
        plt.figure()
        plt.plot(hp_model_fq, hp_model_amp)
        plt.plot(self._model.config.hp_exp_fq, self._model.config.hp_exp_amp, '--')
        plt.show()
        
    def plot_sap_psd(self):
        Fs = 100.0
        dw = 0.25
        
        psa = self._model.state.get_vector('Psa')
        sap_ind = sp.signal.argrelextrema(psa, np.greater)
        sap_t = self._model.state.tout[sap_ind] + self._model.config.t_begin
        
        t0 = max(self._model.config.t_begin, np.ceil(sap_t[0]+Fs)/Fs)
        tt = np.arange(t0, sap_t[-1], 1/Fs)
        fyy = sp.interpolate.interp1d(sap_t, psa[sap_ind], kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)
        sap_model_fq, sap_model_amp = self._model.psd_welch(yy, Fs, dw)
        
        plt.figure()
        plt.plot(sap_model_fq, sap_model_amp)
        plt.plot(self._model.config.sap_exp_fq, self._model.config.sap_exp_amp, '--')
        plt.show()
        
    def plot_dap_psd(self):
        Fs = 100.0
        dw = 0.25
        
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        
        t0 = max(self._model.config.t_begin, np.ceil(dap_t[0]+Fs)/Fs)
        tt = np.arange(t0, dap_t[-1], 1/Fs)
        fyy = sp.interpolate.interp1d(dap_t, psa[dap_ind], kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)
        dap_model_fq, dap_model_amp = self._model.psd_welch(yy, Fs, dw)
        
        plt.figure()
        plt.plot(dap_model_fq, dap_model_amp)
        plt.plot(self._model.config.dap_exp_fq, self._model.config.dap_exp_amp, '--')
        plt.show()
        
    def plot_hp_attractor(self):
        Fs = 100
        dw = 0.25
        
        psa = self._model.state.get_vector('Psa')
        dap_ind = sp.signal.argrelextrema(psa, np.less)
        dap_t = self._model.state.tout[dap_ind] + self._model.config.t_begin
        hp_model = 1000 * np.diff(dap_t)
        
        state_x = hp_model[:-1]
        state_y = hp_model[1:]
        
        mstate_x = self._model.config.HP[1][:-1]
        mstate_y = self._model.config.HP[1][1:]
        
        plt.figure()
        plt.plot(state_x, state_y)
        plt.plot(mstate_x, mstate_y, '--')
        plt.show()
        
