import numpy as np
from scipy import interpolate
import scipy.signal as spsig
from scipy import integrate
import pickle
from physmod_params import PhysmodParams
from physmod_state import PhysmodState
#from physmod_dde import PhysmodDde
from physmod_equations import PhysmodEquations


class Physmod:
    """
    Model class for physiology calculations. Synthesizes other objects to
    perform compuations and contains additional functions.
    """

    def __init__(self, cfg):
        self.config = cfg
        self.out = None

    def init_state(self, state=None):
        if state:
            self.state = state
        else:
            self.state = PhysmodState()
            self.state.init_state(self.config.state0)

    def init_param(self, param=None):
        if param:
            self.param = param
            return

        self.param = PhysmodParams()
        self.param.calculate_vblood(self.config)
        self.param.initialize_constants()
        self.param.initialize_dependants(self.config)

    def run(self):
        #eqs = PhysmodDde(self.config, self.param, self.state, (self.PthorT, self.PthorDat), (self.PabdT, self.PabdDat), self.config.BP_small)
        #eqs.solve()

        eqs = PhysmodEquations(self.config, self.param, self.state, (self.PthorT, self.PthorDat), (self.PabdT, self.PabdDat), self.config.BP_small)
        t_span = (0.0, self.config.iteration_period+self.config.timestep)
        t_eval = np.arange(0.0+self.config.timestep, self.config.iteration_period, self.config.timestep)
        
        print('Initializing solver...')
        res = integrate.solve_ivp(eqs.step, t_span, self.state.get_state(0), t_eval=t_eval, method='RK45')

        # Appending initial state to the condition
        self.state.tout = np.append(0.0, res.t)
        self.state._phys_state = np.c_[res.y, self.state.get_state(0)]

        self.state.calculate_t(eqs, self.param)
        res = []
        return res

    def write_output_file(self, out_filepath):
        out_file = open(out_filepath, 'w')
        for out in self.out:
            out_file.write('%f\n' % (out))
        out_file.close()
    
    def read_params(self, in_filepath):
        self.param = PhysmodParams()
        self.param.read_from_file(in_filepath)
    
    def postrun(self):
        """
        Calculating new initial state based on the first run of the model
        Matlab reference code: GoSUM_Model2_firstrun_6P, after the call to the model executable
        """
        size2 = int(np.round(0.5*len(self.state.tout)))

        psa = self.state.get_vector('Psa')
        psa = psa[size2:]
        dap_ind = spsig.argrelextrema(psa, np.less)
        dap = psa[dap_ind]

        diff_1 = np.abs(dap - self.min_exp)
        ind_1_min = np.argmin(diff_1)
        ind_sel = ind_1_min + size2

        self.config.save_init_state(self.state.get_state(ind_sel), self.state.tout[ind_sel])

    def prerun_cfg(self):
        self.prerun(self.config.BP)

    def prerun(self, BP):
        """
        Performs data setup before running the model.

        Matlab code reference: GoSUM_Model2_firstrun_6P.m
        """
        BP = BP[:, ::60]
        Pthor1, Pthor2 = self.PthorT[0], self.PthorT[-1]
        Pabd1, Pabd2 = self.PabdT[0], self.PabdT[-1]
        t_begin1 = max(Pthor1, Pabd1)
        if self.excl_x_end <= 7:
            t_begin = max(self.excl_x_end+5, t_begin1)
        else:
            t_begin = max(5, t_begin1)
        t_finish1 = min(Pthor2, Pabd2)
        t_finish = min(BP[0, -1], t_finish1)

        ind_1_arr = np.argwhere(BP[0] < t_begin)
        ind_2_arr = np.argwhere(BP[0] < t_begin + 1)
        ind_1 = ind_1_arr[-1][0] + 1
        ind_2 = ind_2_arr[-1][0]
        ind_1_min = np.argmin(BP[1, ind_1:(ind_2 + 1)])
        self.min_exp = BP[1, ind_1:(ind_2 + 1)][ind_1_min]
        ind_begin = ind_1 + ind_1_min
        t_begin_new = BP[0, ind_begin]
        t_begin = t_begin_new
        iteration_period = t_finish - t_begin

        a = np.argwhere(BP[0] >= t_begin)[0][0] - 1
        a = max(0, a)

        intrp_psa = interpolate.CubicSpline(BP[0, a:(a+2)], BP[1, a:(a+2)])
        Psa0 = intrp_psa(t_begin)

        self.state.set('Psa', Psa0)
        self.config.t_begin = t_begin
        self.config.t_end = t_finish
        self.config.iteration_period = iteration_period

    def setup_pthor_cfg(self):
        self.setup_pthor(self.config.TR)

    def setup_pthor(self, TR):
        if self.config.task_id == '6P':
            tmin = 3.0
            tmax = 15.0
            cutoff = 0.03
        else:
            tmin = 2.0
            tmax = 10.0
            cutoff = 0.1
        ef = 0.0

        PthorT, PthorTDat = self.get_pthor(TR[0], TR[1], tmin, tmax, cutoff)

        self.RtDat = PthorT[::3]
        self.RPDat = np.diff(self.RtDat)

        TiTr0 = 0.4
        TeTr0 = 0.35

        Pth1 = -9
        Pth2 = -4
        Pabd1 = -2.5
        Pabd2 = 0

        Pth3 = Pth1 - ef * (Pth2 - Pth1) / 2
        Pth4 = Pth2 + ef * (Pth2 - Pth1) / 2
        Pabd3 = Pabd1 - ef * (Pabd2 - Pabd1) / 2
        Pabd4 = Pabd2 + ef * (Pabd2 - Pabd1) / 2

        self.PthorT = np.zeros(1+3*np.size(self.RPDat))
        self.PthorDat = np.zeros_like(self.PthorT)
        self.PabdT = np.zeros(1+4*np.size(self.RPDat))
        self.PabdDat = np.zeros_like(self.PabdT)

        self.PthorT[0] = self.RtDat[0]
        self.PthorDat[0] = Pth2
        self.PabdT[0] = self.RtDat[0]
        self.PabdDat[0] = Pabd2

        for i in np.arange(len(self.RPDat)):
            Tresp = self.RPDat[i]
            Ti = TiTr0*Tresp
            Te = TeTr0*Tresp

            sindex = (i)*3
            self.PthorT[sindex+1] = self.RtDat[i]+Ti
            self.PthorDat[sindex+1] = Pth1
            self.PthorT[sindex+2] = self.RtDat[i]+Ti+Te
            self.PthorDat[sindex+2] = Pth2
            self.PthorT[sindex+3] = self.RtDat[i]+Tresp
            self.PthorDat[sindex+3] = Pth2

            sindex = i*4
            self.PabdT[sindex+1] = self.RtDat[i]+Ti/2.0
            self.PabdDat[sindex+1] = Pabd1
            self.PabdT[sindex+2] = self.RtDat[i]+Ti
            self.PabdDat[sindex+2] = Pabd1
            self.PabdT[sindex+3] = self.RtDat[i]+Ti+Te
            self.PabdDat[sindex+3] = Pabd2
            self.PabdT[sindex+4] = self.RtDat[i]+Tresp
            self.PabdDat[sindex+4] = Pabd2

    def remove_predicting_cfg(self):
        self.remove_predicting(self.config.BP)

    def remove_predicting(self, BP):
        """
        Blood pressure calibration

        Reference Matlab code: Removing_Predicting_6P.m
        """
        Psa_x = BP[0][::60]
        Psa_y = BP[1][::60]

#        excl_x_begin_1 = np.zeros_like(Psa_x)
        excl_x_begin_1 = []
        excl_x_end_1 = []
        excl_i_begin = []
        excl_i_end = []

        st, rt, i = 5, -1, 0

        while i < np.size(Psa_y)-2:
            if np.abs(Psa_y[i+1]-Psa_y[i]) < 0.5:
                rt += 1
                if rt == st:
                    Psa_y[i+1-st] = 0.0
                    excl_x_begin_1.append(Psa_x[i-st])
                    excl_i_begin.append(i-st)
                    excl_x_end_1.append(excl_x_begin_1[-1]+2.0)
                    if excl_x_end_1[-1] < Psa_x[-1]:
                        excl_i_end.append(np.argwhere(Psa_x >= excl_x_end_1[-1])[0][0])
                        removed = np.arange(excl_i_begin[-1], excl_i_end[-1]+1, dtype=int)
                        Psa_x = np.delete(Psa_x, removed)
                        Psa_y = np.delete(Psa_y, removed)
                    else:
                        removed = np.arange(excl_i_begin[-1], np.size(Psa_x)+1, dtype=int)
                        Psa_x = np.delete(Psa_x, removed)
                        Psa_y = np.delete(Psa_y, removed)
            else:
                rt = -1
            i += 1

        self.excl_x_begin = excl_x_begin_1[0]
        self.excl_x_end = excl_x_end_1[0]

    def get_pthor(self, t, s, Tpmin, Tpmax, cutoff):
        """
        Calculates Pthor for specific model configuration (subject and task).

        Matlab code reference: GetPthor6P.m
        """
        Fs = 100.0

        smin = 2.0 * np.median(s[s < 0.0])
        smax = 2.0 * np.median(s[s > 0.0])
        s = np.maximum(s, smin)
        s = np.minimum(s, smax)

        tim1, sigHF, sigLF = self.remove_lf_signal(t, s, Fs, cutoff, 1, 1.0)
        s1 = sigHF - np.mean(sigHF)

        start_period2 = []
        for k in np.arange(np.size(s1) - 1):
            if (s1[k] > 0) and (s1[k+1] <= 0):
                start_period2.append(tim1[k])
        start_period2 = np.array(start_period2)
        start_index = []
        start_period1 = []
        for k in np.arange(np.size(s1) - 1):
            if (s1[k] <= 0.0) and (s1[k+1] > 0.0):
                check = np.count_nonzero(np.logical_and(start_period2 >= tim1[k]-Tpmin/2.0, start_period2 <= tim1[k]+Tpmin/2.0))
                if check <= 1:
                    start_period1.append(tim1[k])
                    start_index.append(k)
                else:
                    mask = np.argwhere(np.logical_and(start_period2 >= tim1[k]-Tpmin/2.0, start_period2 < tim1[k]))
                    start_period2 = np.delete(start_period2, mask)

        RPDat = np.diff(start_period1)
        dat_mask = np.concatenate(np.argwhere(RPDat > Tpmin))
        RPDat = RPDat[dat_mask]
        start_period1 = np.array(start_period1)[dat_mask]
        start_period1 = np.append( start_period1, start_period1[-1] + RPDat[-1] )

        start_period1_temp = []
        for i in np.arange(np.size(start_period1)-1):
            if RPDat[i] > Tpmax:
                start_period1_temp.append( (start_period1[i] + start_period1[i+1])/2.0  )

        start_period1_temp = np.array(start_period1_temp)
        start_period1 = np.sort( np.append(start_period1, start_period1_temp) )

        RPDat = np.diff(start_period1)
        RtDat = start_period1
        RtDat = np.append( RtDat, RtDat[-1] + RPDat[-1] )

        PthorT = np.zeros([1+3*np.size(RPDat)])
        PthorDat = np.zeros_like(PthorT)
        PthorT[0] = RtDat[0]
        PthorDat[0] = -4

        for i in np.arange(np.size(RPDat)):
            Tresp = RPDat[i]
            Ti = 0.4*Tresp
            Te = 0.35*Tresp

            sindex = i*3

            PthorT[sindex+1] = RtDat[i]+Ti
            PthorDat[sindex+1] = -9
            PthorT[sindex+2] = RtDat[i]+Ti+Te
            PthorDat[sindex+2] = -4
            PthorT[sindex+3] = RtDat[i]+Tresp
            PthorDat[sindex+3] = -4

        tt = np.arange(PthorT[0], PthorT[-1]+1.0/Fs, 1.0/Fs)

        yy = np.zeros_like(tt)
        for j in np.arange(np.size(tt)):
            allk = np.argwhere(PthorT > tt[j])
            if np.shape(allk)[0] == 0:
                k = np.size(PthorT)-1
            else:
                k = allk[0][0]
            yy[j] = PthorDat[k-1] + (tt[j]-PthorT[k-1])/(PthorT[k] - PthorT[k-1]) * (PthorDat[k] - PthorDat[k-1])

        fq, amp = self.psd_welch(yy, Fs, 0.2)
        fq1, amp1 = self.psd_welch(sigHF, Fs, 0.2)

        fact = np.sum(amp1)/np.sum(amp)

        approx = -np.sqrt(fact)*(yy+6.5)

        ks = int(np.round((tt[0] - tim1[0])*Fs))
        Ne = ks + int(np.round(Tpmin*Fs))

        err = np.zeros(Ne+1)
        itarray = np.arange(-1, Ne)
        for nshift in itarray:
            ix1 = np.max([0, ks-(nshift+1)])
            ix2 = np.size(approx)+ks-(nshift+1)
            iy1 = np.max([0, (nshift+1)-ks])
            err[nshift+1] = np.sqrt(np.mean((s1[ix1:ix2] - approx[iy1:])**2))

        mi = np.argmin(err)
        nshift0 = mi - 1
        PthorT = PthorT - (nshift0+1)/Fs

        return PthorT, PthorDat

    def remove_lf_signal(self, tim, sig, Fs, cutoff, order, tadjust):
        """
        Removes low frequency signal based on passed parameters.

        Matlab code reference: RemoveLFsignal.m
        """
        t0 = np.ceil(tim[0]*Fs)/Fs
        tt = np.arange(t0, tim[-1], 1.0/Fs)
        yfun = interpolate.interp1d(tim, sig, kind='cubic')
        y = yfun(tt)

        z, p, k = spsig.butter(order, cutoff*2.0/Fs, 'highpass', output='zpk')
        sos = spsig.zpk2sos(z, p, k)

        zi = spsig.sosfilt_zi(sos)
        yy, zf = spsig.sosfilt(sos, y, zi=zi)
        yy, zf = spsig.sosfilt(sos, y, zi=zf)
        yy = yy - np.mean(yy) + np.mean(y)

        Ne = int(np.round(tadjust*Fs))
        err = np.zeros(Ne+1)
        end = np.size(yy)
        itarray = np.arange(-1, Ne)
        for nshift in itarray:
            yarr = y[nshift+1:] - yy[:end-(nshift+1)]

            fq, amp = self.psd_welch(yarr, Fs, 0.2)
            k1 = np.argwhere(fq >= 0.1)[0][0]
            k2 = np.argwhere(fq > 0.5)[0][0] - 1
            err[nshift+1] = np.mean(amp[k1:k2+1])

        mi = np.argmin(err)
        nshift0 = mi - 1

        tim1 = tt[nshift0+1:]
        sigHF = yy[:end-(nshift0+1)]
        sigLF = y[nshift0+1:] - sigHF

        return tim1, sigHF, sigLF

    def psd_welch(self, y, Fs, window):
        N = np.size(y)
        PW2 = 2**(N-1).bit_length()
        f, PSD = spsig.welch(y - np.mean(y), Fs, window='hann', nperseg=np.floor(window*N),nfft=PW2)
        return f, PSD

    def frequency_amplitude(self):
        """
        Prepares experimental data for cost function calculation
        Matlab code reference: Frequency_Amplitude_6P.m
        """
        Fs = 100.0
        dw = 0.25

        tt = np.arange(self.config.t_begin, self.config.t_end, 1/Fs)

        dt = self.config.dt.item()['dt_%s' % (self.config.task_id)]

        # Setup experimental HP
        ttt1 = self.config.HP[0] + dt
        hp_exp_t = tt
        f_hp_exp = interpolate.interp1d(ttt1, self.config.HP[1], kind='cubic')
        hp_exp = f_hp_exp(hp_exp_t)
        hp_exp_mean = np.mean(hp_exp)
        hp_exp_fq, hp_exp_amp = self.psd_welch(hp_exp, Fs, dw)

        # Setup experimental SAP
        ttt1 = self.config.SAP[0] + dt
        sap_exp_t = tt
        f_sap_exp = interpolate.interp1d(ttt1, self.config.SAP[1], kind='cubic')
        sap_exp = f_sap_exp(sap_exp_t)
        sap_exp_mean = np.mean(sap_exp)
        sap_exp_fq, sap_exp_amp = self.psd_welch(sap_exp, Fs, dw)

        # Setup experimental dap
        ttt1 = self.config.DAP[0] + dt
        dap_exp_t = tt
        f_dap_exp = interpolate.interp1d(ttt1, self.config.DAP[1], kind='cubic')
        dap_exp = f_dap_exp(dap_exp_t)
        dap_exp_mean = np.mean(dap_exp)
        dap_exp_fq, dap_exp_amp = self.psd_welch(dap_exp, Fs, dw)

        self.config.hp_exp_t = hp_exp_t
        self.config.hp_exp = hp_exp
        self.config.hp_exp_mean = hp_exp_mean
        self.config.hp_exp_fq = hp_exp_fq
        self.config.hp_exp_amp = hp_exp_amp

        self.config.sap_exp_t = sap_exp_t
        self.config.sap_exp = sap_exp
        self.config.sap_exp_mean = sap_exp_mean
        self.config.sap_exp_fq = sap_exp_fq
        self.config.sap_exp_amp = sap_exp_amp

        self.config.dap_exp_t = dap_exp_t
        self.config.dap_exp = dap_exp
        self.config.dap_exp_mean = dap_exp_mean
        self.config.dap_exp_fq = dap_exp_fq
        self.config.dap_exp_amp = dap_exp_amp

    def cost_function(self):
        """
        Calculate cost function variables and the cost itself
        Matlab code reference: GoSUMD_Model2_combined_6P.m
        """
        psa = self.state.get_vector('Psa')
        dap_ind = spsig.argrelextrema(psa, np.less)
        dap_t = self.state.tout[dap_ind] + self.config.t_begin

        tt0 = self.config.t_begin
        Fs = 100
        dw = 0.25

        time = dap_t[:-1]
        hp_model = 1000 * np.diff(dap_t)
        t0 = max(tt0, np.ceil(time[0]*Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)

        fyy = interpolate.interp1d(time, hp_model, kind='cubic')
        yy = fyy(tt)
        PLd, PHd, PVd, PLHVd = self.comparison_indices(time, hp_model, self.config.hp_exp_t, self.config.hp_exp)

        opti_out = np.zeros(8)
        opti_out[0] = PVd
        opti_out[1] = PLd
        opti_out[2] = PHd

        dt = self.config.dt.item()['dt_%s' % (self.config.task_id)]

        ttt1 = self.config.HP[0] + dt
        hp_spline = interpolate.interp1d(ttt1, self.config.HP[1], kind='cubic')
        hp_exp = hp_spline(tt)
        opti_out[3] = np.mean(np.abs(yy - hp_exp)) / np.mean(hp_exp)

        time = dap_t
        sap_ind = spsig.argrelextrema(psa, np.greater)
        sap_t = self.state.tout[sap_ind] + self.config.t_begin
        t0 = max(tt0, np.ceil(time[0]*Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        sap_model = psa[sap_ind]
        fyy = interpolate.interp1d(sap_t, sap_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)

        ttt1 = self.config.SAP[0] + dt
        sap_spline = interpolate.interp1d(ttt1, self.config.SAP[1], kind='cubic')
        sap_exp = sap_spline(tt)
        opti_out[4] = np.mean(np.abs(yy-sap_exp)) / np.mean(sap_exp)

        time = dap_t
        t0 = max(tt0, np.ceil(time[0]+Fs)/Fs)
        tt = np.arange(t0, time[-1], 1/Fs)
        dap_model = psa[dap_ind]
        fyy = interpolate.interp1d(time, dap_model, kind='cubic', fill_value='extrapolate')
        yy = fyy(tt)

        ttt1 = self.config.DAP[0] + dt
        dap_spline = interpolate.interp1d(ttt1, self.config.DAP[1], kind='cubic')
        dap_exp = dap_spline(tt)
        opti_out[5] = np.mean(np.abs(yy-dap_exp)) / np.mean(dap_exp)

        dpd_sap = self.dpd_indices(sap_t, sap_model, self.config.sap_exp_t, self.config.sap_exp)
        dpd_dap = self.dpd_indices(dap_t, dap_model, self.config.dap_exp_t, self.config.dap_exp)

        print(dpd_sap)
        print(dpd_dap)

        opti_out[5] = dpd_sap
        opti_out[6] = dpd_dap

        if self.config.task_id == '6P':
            coefs = np.array([0.0, 1.5, 0.5, 2.0, 1.5, 1.5, 0.15, 0.15])
        else:
            coefs = np.array([1.0, 0.0, 1.5, 2.0, 1.0, 1.0, 0.15, 0.15])

        cost = np.sum(opti_out[:-1] * coefs[coefs > 0.0])
        opti_out[7] = np.log(cost)
        
        self.out = opti_out
        
        return opti_out

    def comparison_indices(self, t1, y1, t2, y2):
        """
        Matlab reference script: Comparison_indices.m
        """
        Fmax = np.array([0.5])  # to array for compatibility

        P1, f1 = self.NonuniformFFTamplitudeHann(t1, y1, Fmax, 8, 0, 0)
        P2, f2 = self.NonuniformFFTamplitudeHann(t2, y2, Fmax, 8, 0, 0)
        F0 = 0.002
        F1 = 0.05
        F2 = 0.15
        F3 = 0.15
        i01 = int(np.argwhere(f1[0] > F0)[0])
        i02 = np.argwhere(f2[0] > F0)[0]
        i11 = int(np.argwhere(f1[0] > F1)[0])
        i12 = np.argwhere(f2[0] > F1)[0]
        i21 = int(np.argwhere(f1[0] > F2)[0])
        i22 = np.argwhere(f2[0] > F2)[0]
        i31 = int(np.argwhere(f1[0] > F3)[0])
        i32 = np.argwhere(f2[0] > F3)[0]
        PSD1 = P1[0]**2
        PSD2 = P2[0]**2

        fspline = interpolate.interp1d(f2[0], PSD2, kind='cubic', fill_value='extrapolate')
        PSD2_spline = fspline(f1[0])

        PLd = np.sum(np.abs(PSD2_spline[i11:(i21-1)]-PSD1[i11:(i21-1)])) / np.sum(PSD2_spline[i11:(i21-1)])
        PHd = np.sum(np.abs(PSD2_spline[i31:]-PSD1[i31:])) / np.sum(PSD2_spline[i31:])
        PVd = np.sum(np.abs(PSD2_spline[i01:(i11-1)]-PSD1[i01:(i11-1)])) / np.sum(PSD2_spline[i01:(i11-1)])

        PLHVd = np.sum(np.abs(PSD2_spline-PSD1))/np.sum(PSD2_spline)

        return PLd, PHd, PVd, PLHVd

    def NonuniformFFTamplitudeHann(self, x, y, f, iofac, kwind, ovlap):
        """
        Matlab reference code: NonuniformFFTamplitudeHann.m
        """
        if kwind < 1:
            PrHann, fr = self.NonuniformFFTamplitude(x, y, f, iofac)
            return PrHann, fr

        T0 = np.min(x)
        Ttot = np.max(x)-T0

        Twind = Ttot/(ovlap+kwind*(1-ovlap))
        Toverlap = ovlap*Twind

        t00 = T0
        KMU = 0
        firstP = True

        for i in np.arange(0, kwind+1):
            index_mask = np.logical_and(x >= t00, x < t00 + Twind)
            t = x[index_mask]
            yw = spsig.detrend(y[index_mask], 0)

            if np.size(np.unique(yw)) > 1:
                if not i == 1:
                    t = np.vstack((t00, t))
                    yw = np.vstack((0, yw))
                t = t - t[0]
                if t[-1] != Twind:
                    t = np.vstack((t, Twind))
                    yw = np.vstack(yw, 0)

                window = 0.5 * (1-np.cos(2*np.pi*t/Twind))
                yw = window * yw
                KMU = KMU + (integrate.trapz(window, t) / Twind)**2

                if firstP:
                    Pr, fr = self.NonuniformFFTamplitude(t, yw, f, np.round(iofac*Ttot/Twind))
                    PrHann = np.zeros(np.shape(Pr))
                    for j in np.arange(0, len(Pr)+1):
                        PrHann[j] = Pr[j]**2
                    firstP = False
                else:
                    Pr, _ = self.NonuniformFFTamplitude(t, yw, f, np.round(iofac*Ttot/Twind))
                    for j in np.arange(0, len(PrHann)+1):
                        if np.shape(PrHann[j])[0] != np.shape(Pr[j])[0]:
                            print('Warning!')
                        else:
                            PrHann[j] = PrHann[j] + Pr[j]**2
            t00 = t00 + (Twind - Toverlap)

        if firstP:
            fr = np.vstack((2/(Twind*np.round(iofac*Ttot/Twind)), f))
            PrHann = np.vstack((np.nan, np.nan))
        else:
            for j in np.arange(0, len(PrHann)+1):
                PrHann[j] = np.sqrt(PrHann[j]/KMU)

        return PrHann, fr

    def NonuniformFFTamplitude(self, x, y, f, iofac=4):
        """
        Matlab reference code: NonuniformFFTamplitudeHann.m, NonuniformFFTamplitude function
        """
        nfreq = 64
        macc = 3  # originally 4, adjusted for Python indices

        n = np.size(y)
        ave = np.mean(y)
        vr = np.var(y)
        xmin = np.min(x)
        xmax = np.max(x)
        xdif = xmax - xmin
        df = 1.0/(xdif * iofac)
        f = np.sort(f[f > df])
        f = np.append(df, f)
        fc = n / (2 * xdif)

        hifac = f / fc
        nout = np.round(0.5 * iofac * hifac * n)
        nout = np.array([int(i) for i in nout])
        noutmax = nout[-1]
        nfreqt = 2 * noutmax * macc

        if nfreq < nfreqt:
            nfreq = 2**int(nfreqt-1).bit_length()

        ndim = nfreq

        wk1 = np.zeros(ndim)
        wk2 = np.zeros_like(wk1)
        fac = ndim/(xdif*iofac)
        fndim = ndim
        ck = 1.0 + np.remainder((x-xmin)*fac, fndim)
        ckk = 1.0 + np.remainder(2.0*(ck-1.0), fndim)

        for j in np.arange(n):
            wk1 = self.spread(wk1, y[j]-ave, ndim, ck[j], macc)
            wk2 = self.spread(wk2, 1, ndim, ckk[j], macc)

        tmp1 = np.fft.fft(wk1[:nfreq+1])
        tmp2 = np.fft.fft(wk2[:nfreq+1])

        s1 = int(np.size(tmp1)/2+2)
        wk1 = tmp1[s1:][::-1]
    #    wk1 = np.vstack((np.real(wk1), np.imag(wk1)))
        wk1 = np.concatenate([[x, y] for x, y in zip(np.real(wk1), np.imag(wk1))])
        s2 = int(np.size(tmp2)/2+2)
        wk2 = tmp2[s2:][::-1]
        wk2 = np.concatenate([[x, y] for x, y in zip(np.real(wk2), np.imag(wk2))])

        k = np.arange(0, 2*noutmax-1+1, 2, dtype=int)
        kp1 = k + 1
        hypo = np.sqrt(wk2[k]**2 + wk2[kp1]**2)
        hc2wt = 0.5*wk2[k] / hypo
        hs2wt = 0.5*wk2[kp1] / hypo
        cwt = np.sqrt(0.5 + hc2wt)

        swt = np.abs(np.sqrt(0.5-hc2wt) * np.sign(hs2wt))
        den = 0.5 * n + hc2wt * wk2[k] + hs2wt * wk2[kp1]
        cterm = (cwt * wk1[k] + swt * wk1[kp1])**2 / den
        sterm = (cwt * wk1[kp1] - swt * wk1[k]**2) / (n-den)
        P = (cterm + sterm) / (2 * vr)
        P[P < 0] = np.min(P[P > 0])
        F = np.arange(0, noutmax + 1) * df

        fr = {}
        Pr = {}

        for j in np.arange(0, len(f)-1, dtype=int):
            fr[j] = F[nout[j]:nout[j+1]]
            Pr[j] = 2 * np.sqrt(vr * P[nout[j]:nout[j+1]]) / n

        return Pr, fr

    def spread(self, yy, y, n, x, m):
        nfac = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])

        ix = int(x)

        if (x == np.fix(x)):
            yy[ix] = yy[ix] + y
        else:
            ilo = int(min(max(np.round(x - 0.5 * m + 1), 1), n - m + 1))
            ihi = int(ilo + m - 1)
            nden = nfac[m]

            fac = 1
            for j in np.arange(ilo, ihi+1):
                fac = fac*(x-j)
            yy[ihi] = yy[ihi] + y*fac/(nden*(x-ihi))

            for j in np.arange(ihi-1, ilo-1, -1):
                nden = np.fix(nden/(j+1-ilo))*(j-ihi)
                yy[j] = yy[j] + y*fac/(nden*(x-j))

        return yy

    def dpd_indices(self, t1, y1, t2, y2):
        Fmax = 0.5
        N1 = np.size(t1)
        N2 = np.size(t2)
        T1 = np.max(t1) - np.min(t1)
        T2 = np.max(t2) - np.min(t2)
        dt = 0.5 * np.min([1/Fmax, 0.5*T1/(N1-1) + T2/(N2-1)])
        tt1 = np.min(t1) + np.arange(0, T1, dt)
        tt2 = np.min(t2) + np.arange(0, T2, dt)
        N = np.min([np.size(tt1), np.size(tt2)])
        ttt1 = tt1[:N+1]
        ttt2 = tt2[:N+1]
        T = np.max(ttt1) - np.min(ttt1)
        Lmax = int(np.floor(Fmax*T)+1)
        ff = np.arange(0, Lmax+1)/T
        df = 1.0/T
        dfend = Fmax-ff[-2]

        f1 = interpolate.interp1d(t1, y1, kind='cubic')
        yy1 = f1(ttt1)
        f2 = interpolate.interp1d(t2, y2, kind='cubic')
        yy2 = f2(ttt2)

        YY1 = np.fft.fft(yy1-np.mean(yy1))/N
        PSD1 = 2.0*np.abs(YY1[:Lmax+1])**2/N
        YY2 = np.fft.fft(yy2-np.mean(yy2))/N
        PSD2 = 2.0*np.abs(YY2[:Lmax+1])**2/N

        F1 = 0.002
        i1 = int(1 + np.floor(F1*T))

        dpd = np.sum(np.abs(PSD2[i1:] - PSD1[i1:])) / np.sum(PSD2[i1:])

        return dpd

    def save_model(self):
        pickle_path = '%s/%s_%s_model.p' % (self.config.respath, self.config.task_id, self.config.subject_id)
        pickled_model = pickle.dump(self, open(pickle_path, 'wb'))
