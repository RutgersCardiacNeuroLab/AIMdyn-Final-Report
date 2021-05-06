import numpy as np
from scipy import interpolate
from chspy import CubicHermiteSpline, Anchor

class PhysmodEquations:
    def __init__(self, cfg, param, state, pthor, pabd, psa):
        self.fpthor = interpolate.interp1d(pthor[0], pthor[1], bounds_error=False, fill_value=np.nan)
        self.fpabd = interpolate.interp1d(pabd[0], pabd[1], bounds_error=False, fill_value='extrapolate')

        self.fpsa = interpolate.interp1d(psa[0], psa[1], bounds_error=False, fill_value=np.nan)

        hist_filter = psa[0] < cfg.t_begin
        self.psa_spline = CubicHermiteSpline.from_data(psa[0][hist_filter], np.expand_dims(psa[1][hist_filter], axis=1))
        
        self.param = param
        self.state = state
        self.cfg = cfg

    def delayed_pthor(self, t):
        # t is the proper simulation time
        delays = np.append(t - self.param.get_delay_vector() + self.cfg.t_begin, t + self.cfg.t_begin)
        pthors = self.fpthor(delays)
        pthors[np.isnan(pthors)] = -4

        return pthors

    def interp_pabd(self, t):
        return self.fpabd(t+self.cfg.t_begin)

    def delayed_psa(self, t):
        # delays = self.param.get_delay_vector()
        # intrp_t = t + self.cfg.t_begin - delays
        # psas = self.fpsa(intrp_t)
        # psas[delays >= t] = self.state.get('Psa', 0)
        delays = self.param.get_delay_vector()
        intrp_t = t - delays + self.cfg.t_begin
        psas = self.psa_spline.get_state(intrp_t)
        psas[intrp_t <= self.cfg.t_begin] = self.state.get('Psa', 0)
        psas = np.concatenate(psas)
        return psas

    def update_psa(self, t, psa, dpsa):
        new_anchor = Anchor(t+self.cfg.t_begin, psa, dpsa)
        if t % self.cfg.timestep > 0.05*self.cfg.timestep or (np.isclose(new_anchor.time, self.psa_spline.times[-1]) or new_anchor.time < self.psa_spline.times[-1]):
            return
        self.psa_spline.append(new_anchor)

    def step(self, t, y):
        PsaDtv, PsaDts, PsaDElv, PsaDErv, PsaDVusv, PsaDVuev, PsaDRep, PsaDRsp = self.delayed_psa(t)
        pthors = self.delayed_pthor(t)
        VLDtv, VLDts, VLDElv, VLDErv, VLDVusv, VLDVuev, VLDRep, VLDRsp = 1900 - 100 * pthors[:8]
        pthort = pthors[-1]
        pabdt = self.interp_pabd(t)

        # unravel state
        Ppa, Fpa, Ppp, Ppv, Psa, Fsa, Psp, Pev, Pla, Pra, Vrv, Vlv,  xi, xTs, xTv, Emaxlv, Emaxrv, Rsp, Rep, Vusv, Vuev = y

        Ppa = Ppa - pthort
        Ppp = Ppp - pthort
        Ppv = Ppv - pthort
        Psp = Psp - pabdt
        Pla = Pla - pthort
        Pra = Pra - pthort

        tmpNoise = 0
        if self.param.params['A'] != 0:
            tmpNoise = 0

        Rep = Rep * (1 + self.param.params['A'] * tmpNoise)

        kT = (self.param.params['Tmax'] - self.param.params['Tmin'])/(4*self.param.params['St0'])
        #calculate T
        VT = self.param.params['GaTv'] * (PsaDtv - self.param.params['Psan']) - self.param.params['GpTv'] * (VLDtv - self.param.params['VLn']) - self.param.params['G1Tv'] # add delay
        ST = self.param.params['GaTs'] * (PsaDts - self.param.params['Psan']) + self.param.params['GpTs'] * (VLDts - self.param.params['VLn'])  # add delay
        dxTv = 1.0 / self.param.params['tauTv'] * (VT - xTv)  # state dxTv
        dxTs = 1.0 / self.param.params['tauTs'] * (ST - xTs)  # state dxTs
        xT = xTv + xTs
        T = (self.param.params['Tmin'] + self.param.params['Tmax'] * np.exp(xT / kT)) / (1 + np.exp(xT / kT))  # 0.833; modified using Eq.A14

        dxi = 1 / T  # state 1 added using Eq.21
        u = xi - np.floor(xi)

        TT = (self.param.params['Tsys0'] - self.param.params['Ksys'] / T) / T
        Pmaxlv = self.huai(u, TT) * Emaxlv * (Vlv - self.param.params['Vulv'])\
                + ( 1 - self.huai(u, TT)) * self.param.params['P0lv'] * (np.exp(self.param.params['Kelv'] * Vlv) - 1)  # Eq.18
        Pmaxrv = self.huai(u, TT) * Emaxrv * (Vrv - self.param.params['Vurv'])\
                + (1 - self.huai(u, TT)) * self.param.params['P0rv'] * (np.exp(self.param.params['Kerv'] * Vrv)-1)  # Eq.29

        # calculate Rlv using Eq.16
        Rlv = self.param.params['Krlv'] * Pmaxlv

        # Eq.5
        dPsa = (self.func(Pmaxlv, Psa, Rlv) - Fsa) / self.param.params['Csa']
        
        self.update_psa(t, Psa, dPsa)

        # how to add in the Delays
        xThetaElv = self.param.params['GaElv'] * (PsaDElv - self.param.params['Psan']) + self.param.params['GpElv'] * (VLDElv - self.param.params['VLn'])
        kThetaElv = (self.param.params['Elvmax'] - self.param.params['Elvmin']) / (4 * self.param.params['Selv0'])
        dEmaxlv = (1.0 / self.param.params['tauElv']) * (self.sigma(self.param.params['Elvmin'], self.param.params['Elvmax'], xThetaElv, kThetaElv) - Emaxlv)  # state dEmaxlv added using Eq.A6

        xThetaErv = self.param.params['GaErv'] * (PsaDErv - self.param.params['Psan']) + self.param.params['GpErv'] * (VLDErv - self.param.params['VLn'])
        kThetaErv = (self.param.params['Ervmax'] - self.param.params['Ervmin']) / (4 * self.param.params['Serv0'])
        dEmaxrv = (1 / self.param.params['tauErv']) * (self.sigma(self.param.params['Ervmin'], self.param.params['Ervmax'], xThetaErv, kThetaErv) - Emaxrv)  # state dEmaxrv added using Eq.A6

        xThetaRsp = self.param.params['GaRsp'] * (PsaDRsp - self.param.params['Psan']) + self.param.params['GpRsp'] * (VLDRsp - self.param.params['VLn']) + self.param.params['G2Rsp']
        kThetaRsp = (self.param.params['Rspmax'] - self.param.params['Rspmin']) / (4 * self.param.params['Srsp0'])
        dRsp = (1 / self.param.params['tauRsp']) * (self.sigma(self.param.params['Rspmin'], self.param.params['Rspmax'], xThetaRsp, kThetaRsp) - Rsp)  # state dRsp added using Eq.A6

        xThetaRep = self.param.params['GaRep'] * (PsaDRep - self.param.params['Psan']) + self.param.params['GpRep'] * (VLDRep - self.param.params['VLn']) + self.param.params['G3Rep']
        kThetaRep = (self.param.params['Repmax'] - self.param.params['Repmin']) / (4 * self.param.params['Srep0'])
        dRep = (1 / self.param.params['tauRep']) * (self.sigma(self.param.params['Repmin'], self.param.params['Repmax'], xThetaRep, kThetaRep) - Rep)  # state dRep added using Eq.A6

        xThetaVusv = self.param.params['GaVusv'] * (PsaDVusv - self.param.params['Psan']) + self.param.params['GpVusv'] * (VLDVusv - self.param.params['VLn'])
        kThetaVusv = (self.param.params['Vusvmax'] - self.param.params['Vusvmin']) / (4 * self.param.params['Svusv0'])
        dVusv = (1 / self.param.params['tauVusv']) * (self.sigma(self.param.params['Vusvmin'], self.param.params['Vusvmax'], xThetaVusv, kThetaVusv) - Vusv)  # state dVusv added using Eq.A6

        xThetaVuev = self.param.params['GaVuev'] * (PsaDVuev - self.param.params['Psan']) + self.param.params['GpVuev'] * (VLDVuev - self.param.params['VLn'])
        kThetaVuev = (self.param.params['Vuevmax'] - self.param.params['Vuevmin']) / (4 * self.param.params['Svuev0'])
        dVuev = (1 / self.param.params['tauVuev']) * (self.sigma(self.param.params['Vuevmin'], self.param.params['Vuevmax'], xThetaVuev, kThetaVuev) - Vuev) # state dVuev added using Eq.A6

        # calculate Rrv using Eq.27
        Rrv = self.param.params['Krrv'] * Pmaxrv
        # Eq.1
        dPpa = (self.func(Pmaxrv, Ppa, Rrv) - Fpa) / self.param.params['Cpa']

        # Eq.2
        dFpa = (Ppa - Ppp - self.param.params['Rpa'] * Fpa) / self.param.params['Lpa']

        # Eq.3
        dPpp = (Fpa - (Ppp - Ppv) / self.param.params['Rpp']) / self.param.params['Cpp']

        # Eq.4
        dPpv = ((Ppp - Ppv) / self.param.params['Rpp'] - (Ppv - Pla) / self.param.params['Rpv']) / self.param.params['Cpv']

        # Eq.6
        dFsa = (Psa - Psp - self.param.params['Rsa'] * Fsa) / self.param.params['Lsa']

        # calculate Vu using Eq.10
        Vu = self.param.params['Vusa'] + self.param.params['Vusp'] + self.param.params['Vuep'] + Vusv + Vuev + self.param.params['Vura'] + self.param.params['Vupa'] + self.param.params['Vupp'] + self.param.params['Vupv'] + self.param.params['Vula']

        # calculate Psv using Eq.9
        Psv = (self.param.params['Vblood'] - self.param.params['Csa'] * Psa - (self.param.params['Csp'] + self.param.params['Cep']) * Psp - self.param.params['Cev'] * Pev - self.param.params['Cra'] * Pra - Vrv - self.param.params['Cpa'] * Ppa - self.param.params['Cpp'] * Ppp - self.param.params['Cpv'] * Ppv - self.param.params['Cla'] * Pla - Vlv - Vu) / self.param.params['Csv'] - pabdt

        # Eq.7
        dPsp = (Fsa - (Psp - Psv) / Rsp - (Psp - Pev) / Rep) / (self.param.params['Csp'] + self.param.params['Cep'])

        # Eq.8
        dPev = ((Psp - Pev) / Rep - (Pev - Pra) / self.param.params['Rev'] - dVuev ) / self.param.params['Cev']

        # calculate Plv using Eq.17
        Plv = Pmaxlv - Rlv * self.func(Pmaxlv, Psa, Rlv)
        # Eq.12
        dPla = ((Ppv - Pla) / self.param.params['Rpv'] - self.func(Pla, Plv, self.param.params['Rla'])) / self.param.params['Cla']

        # calculate Prv using Eq.28
        Prv = Pmaxrv - Rrv * self.func(Pmaxrv, Ppa, Rrv)

        # Eq.23
        dPra = ((Psv - Pra) / self.param.params['Rsv'] + (Pev - Pra)/self.param.params['Rev'] - self.func(Pra, Prv, self.param.params['Rra'])) / self.param.params['Cra']

        # Eq.25
        dVrv = self.func(Pra, Prv, self.param.params['Rra']) - self.func(Pmaxrv, Ppa, Rrv)  # state 6 added using Eq.25

        # Eq.14
        dVlv = self.func(Pla, Plv, self.param.params['Rla']) - self.func(Pmaxlv, Psa, Rlv)  # state 7 added using Eq.14


        return np.array([dPpa, dFpa, dPpp, dPpv, dPsa, dFsa, dPsp, dPev, dPla, dPra, dVrv, dVlv, dxi, dxTs, dxTv, dEmaxlv, dEmaxrv, dRsp, dRep, dVusv, dVuev])

    def huai(self, u, TT):
        #  Eq. 19
        if u >= 0 and u <= TT:
            return np.sin(np.pi * u / TT)**2
        else:
            return 0.0

    #  function for calculating For(using Eq.26), Fol(using Eq.15), Fil(using Eq.13), Fir(using Eq.24)
    def func(self, x1, x2, x3):
        if x1 <= x2:
            return 0.0
        else:
            return (x1 - x2) / x3

    def sigma(self, min_val, max_val, x_theta, k_theta):
        return (min_val + max_val * np.exp(x_theta / k_theta)) / (1 + np.exp(x_theta / k_theta))
