from jitcdde import jitcdde_input, y, t, input
import numpy as np
from scipy import interpolate
import symengine as sp
from chspy import CubicHermiteSpline
from matplotlib import pyplot as plt

class PhysmodDde:
    def __init__(self, cfg, param, state, pthor, pabd, psa):
        self.fpthor = interpolate.interp1d(pthor[0], pthor[1], bounds_error=False, fill_value='extrapolate')
        self.fpabd = interpolate.interp1d(pabd[0], pabd[1], bounds_error=False, fill_value='extrapolate')
        
        min_pthor = np.min(pthor[0])
        min_pabd = np.min(pabd[0])
        max_pthor = np.max(pthor[0])
        max_pabd = np.max(pabd[0])

        dtpthor = np.median(np.diff(pthor[0]))
        dtpabd = np.median(np.diff(pabd[0]))
        intrp_min = min(min_pthor, min_pabd)
        intrp_max = max(max_pthor, max_pabd)
        print(min_pthor, max_pthor)
        print(min_pabd, max_pabd)
        print(intrp_min, intrp_max)

        self.intrp_time = np.arange(intrp_min, intrp_max, 0.25)
        self.Pthor = self.fpthor(self.intrp_time)
        self.Pabd = self.fpabd(self.intrp_time)
        print(len(self.Pthor))
        plt.figure()
        plt.plot(self.Pthor)
        plt.plot(self.Pabd)
        plt.show()

        self.param = param
        self.state = state
        self.cfg = cfg

    def interp_pabd(self, y, vart):
        return self.fpabd(vart)

    def interp_pthor(self, y, vart):
        return self.fpthor(vart)

    def solve(self):
        f = [   (1/self.param.params['Cpa'] * self.func(self.Pmaxrv(), y(0), self.Rrv()) - y(1)) / self.param.params['Cpa'], # 1 Ppa
                (y(0) - y(2) - self.param.params['Rpa'] * y(1)) / self.param.params['Lpa'], # 2 Fpa
                (y(1) - (y(2) - y(3)) / self.param.params['Rpp']) / self.param.params['Cpp'], # 3 Ppp
                ((y(2) - y(3)) / self.param.params['Rpp'] - (y(3) - y(8)) / self.param.params['Rpv']) / self.param.params['Cpv'], # 4 Ppv
                (self.func(self.Pmaxlv(), y(4), self.Rlv()) - y(5)) / self.param.params['Csa'], # 5 Psa
                (y(4) - y(6) - self.param.params['Rsa'] * y(5)) / self.param.params['Lsa'], # 6 Fsa
                (y(5) - (y(6) - self.Psv()) / y(17) - (y(6) - y(7)) / y(18)) / (self.param.params['Csp'] + self.param.params['Cep']), # 7 Psp
                ((y(6) - y(7)) / y(18) - (y(7) - y(9)) / self.param.params['Rev'] - self.dVuev() ) / self.param.params['Cev'], # 8 Pev
                ((y(3) - y(8)) / self.param.params['Rpv'] - self.func(y(8), self.Plv(), self.param.params['Rla'])) / self.param.params['Cla'], # 9 Pla
                ((self.Psv() - y(9)) / self.param.params['Rsv'] + (y(7) - y(9))/self.param.params['Rev'] - self.func(y(9), self.Prv(), self.param.params['Rra'])) / self.param.params['Cra'], # 10 Pra
                self.func(y(9), self.Prv(), self.param.params['Rra']) - self.func(self.Pmaxrv(), y(0), self.Rrv()), # 11 Vrv
                self.func(y(8), self.Plv(), self.param.params['Rla']) - self.func(self.Pmaxlv(), y(4), self.Rlv()), # 12 Vlv
                1 / self.T(self.xT(), self.kT()), # 13 xi
                1.0 / self.param.params['tauTs'] * (self.ST() - y(13)), # 14 xTs
                1.0 / self.param.params['tauTv'] * (self.VT() - y(14)) , # 15 xTv 
                self.dEmaxlv(), # 16 Emaxlv
                self.dEmaxrv(), # 17 Emaxrv
                self.dRsp(), # 18 Rsp
                self.dRep(), # 19 Rep
                self.dVusv(), # 20 Vusv
                self.dVuev() # 21 Vuev
            ]
        # f = [   (1/self.param.params['Cpa'] * self.func(y(1), y(0), y(1)) - y(1)) / self.param.params['Cpa'], # 1 Ppa 
        #         (y(0) - y(1) - self.param.params['Rpa'] * y(1)) / self.param.params['Lpa'] # 2 Fpa
        #     ]
        
        dt = 0.003
        t_eval = np.arange(0*dt, 1000*dt, dt)

        intrp_state = np.vstack((self.Pthor, self.Pabd)).T
        input_spline = CubicHermiteSpline.from_data(self.intrp_time, intrp_state)

        fig,axes = plt.subplots()
        input_spline.plot(axes)
        axes.set_title("input")
        plt.show()

        DDE = jitcdde_input(f, input=input_spline)
        DDE.check()
        DDE.compile_C(verbose=True)
        DDE.constant_past(np.zeros(len(f)))
        print(DDE.t)
        

    #  function for calculating For(using Eq.26), Fol(using Eq.15), Fil(using Eq.13), Fir(using Eq.24)
    def func(self, x1, x2, x3):
        return sp.Max(0.0, x1 - x2) / x3

    def Pmaxrv(self):
        xT = self.xT()
        kT = self.kT()
        T = self.T(xT, kT)
        u = self.u()
        TT = self.TT(T)
        return self.huai(u, TT) * y(16) * (y(10) - self.param.params['Vurv'])\
                + (1 - self.huai(u, TT)) * self.param.params['P0rv'] * (sp.functions.exp(self.param.params['Kerv'] * y(10))-1)
    
    def Pmaxlv(self):
        xT = self.xT()
        kT = self.kT()
        T = self.T(xT, kT)
        u = self.u()
        TT = self.TT(T)
        return self.huai(u, TT) * y(15) * (y(11) - self.param.params['Vulv'])\
                + ( 1 - self.huai(u, TT)) * self.param.params['P0lv'] * (sp.functions.exp(self.param.params['Kelv'] * y(11)) - 1)  # Eq.18

    def huai(self, u, TT):
        #  Eq. 19
        if u >= 0 and u <= TT:
            return sp.sin(sp.pi * u / TT)**2
        else:
            return 0.0 

    def TT(self, T):
        return (self.param.params['Tsys0'] - self.param.params['Ksys'] / T) / T

    def u(self):
        return y(12) - sp.floor(y(12))

    def T(self, xT, kT):
        return (self.param.params['Tmin'] + self.param.params['Tmax'] * sp.functions.exp(xT / kT)) / (1 + sp.functions.exp(xT / kT))  # 0.833; modified using Eq.A14

    def xT(self):
        return y(14) + y(13)

    def kT(self):
       return (self.param.params['Tmax'] - self.param.params['Tmin'])/(4*self.param.params['St0'])

    def Rrv(self):
        return self.param.params['Krrv'] * self.Pmaxrv()
    
    def Rlv(self):
        return self.param.params['Krlv'] * self.Pmaxlv()

    def Psv(self):
        return (self.param.params['Vblood'] - self.param.params['Csa'] * y(4) - (self.param.params['Csp'] + self.param.params['Cep']) * y(6) - self.param.params['Cev'] * y(7) - self.param.params['Cra'] * y(9) - y(10) - self.param.params['Cpa'] * y(0) - self.param.params['Cpp'] * y(2) - self.param.params['Cpv'] * y(3) - self.param.params['Cla'] * y(8) - y(11) - self.Vu()) / self.param.params['Csv'] - input(1, t)

    def Vu(self):
        return self.param.params['Vusa'] + self.param.params['Vusp'] + self.param.params['Vuep'] + y(19) + y(20) + self.param.params['Vura'] + self.param.params['Vupa'] + self.param.params['Vupp'] + self.param.params['Vupv'] + self.param.params['Vula']

    def dVuev(self):
        VLDVuev = 1900 - 100 * input(0, t - self.param.params['DVuev'])
        xThetaVuev = self.param.params['GaVuev'] * ( y(4, t-self.param.params['DVuev']) - self.param.params['Psan']) + self.param.params['GpVuev'] * (VLDVuev - self.param.params['VLn'])
        kThetaVuev = (self.param.params['Vuevmax'] - self.param.params['Vuevmin']) / (4 * self.param.params['Svuev0'])
        return (1 / self.param.params['tauVuev']) * (self.sigma(self.param.params['Vuevmin'], self.param.params['Vuevmax'], xThetaVuev, kThetaVuev) - y(20))

    def dEmaxlv(self):
        VLDElv = 1900 - 100 * input(0, t - self.param.params['DElv'])
        xThetaElv = self.param.params['GaElv'] * (y(4, t-self.param.params['DElv']) - self.param.params['Psan']) + self.param.params['GpElv'] * (VLDElv - self.param.params['VLn'])
        kThetaElv = (self.param.params['Elvmax'] - self.param.params['Elvmin']) / (4 * self.param.params['Selv0'])
        return (1.0 / self.param.params['tauElv']) * (self.sigma(self.param.params['Elvmin'], self.param.params['Elvmax'], xThetaElv, kThetaElv) - y(15))

    def dEmaxrv(self):
        VLDErv = 1900 - 100 * input(0, t - self.param.params['DErv'])
        xThetaErv = self.param.params['GaErv'] * (y(4, t-self.param.params['DErv']) - self.param.params['Psan']) + self.param.params['GpErv'] * (VLDErv - self.param.params['VLn'])
        kThetaErv = (self.param.params['Ervmax'] - self.param.params['Ervmin']) / (4 * self.param.params['Serv0'])
        return (1 / self.param.params['tauErv']) * (self.sigma(self.param.params['Ervmin'], self.param.params['Ervmax'], xThetaErv, kThetaErv) - y(16))  # state dEmaxrv added using Eq.A6

    def dRsp(self):
        VLDRsp = 1900 - 100 * input(0, t - self.param.params['DRsp'])
        xThetaRsp = self.param.params['GaRsp'] * (y(4,t-self.param.params['DRsp']) - self.param.params['Psan']) + self.param.params['GpRsp'] * (VLDRsp - self.param.params['VLn']) + self.param.params['G2Rsp']
        kThetaRsp = (self.param.params['Rspmax'] - self.param.params['Rspmin']) / (4 * self.param.params['Srsp0'])
        return (1 / self.param.params['tauRsp']) * (self.sigma(self.param.params['Rspmin'], self.param.params['Rspmax'], xThetaRsp, kThetaRsp) - y(17))  # state dRsp added using Eq.A6

    def dRep(self):
        VLDRep = 1900 - 100 * input(0, t - self.param.params['DRep'])
        xThetaRep = self.param.params['GaRep'] * (y(4, self.param.params['DRep']) - self.param.params['Psan']) + self.param.params['GpRep'] * (VLDRep - self.param.params['VLn']) + self.param.params['G3Rep']
        kThetaRep = (self.param.params['Repmax'] - self.param.params['Repmin']) / (4 * self.param.params['Srep0'])
        return (1 / self.param.params['tauRep']) * (self.sigma(self.param.params['Repmin'], self.param.params['Repmax'], xThetaRep, kThetaRep) - y(18))  # state dRep added using Eq.A6

    def dVusv(self):
        VLDVusv = 1900 - 100 * input(0, t - self.param.params['DVusv'])
        xThetaVusv = self.param.params['GaVusv'] * (y(4, t-self.param.params['DVusv']) - self.param.params['Psan']) + self.param.params['GpVusv'] * (VLDVusv - self.param.params['VLn'])
        kThetaVusv = (self.param.params['Vusvmax'] - self.param.params['Vusvmin']) / (4 * self.param.params['Svusv0'])
        return (1 / self.param.params['tauVusv']) * (self.sigma(self.param.params['Vusvmin'], self.param.params['Vusvmax'], xThetaVusv, kThetaVusv) - y(19))  # state dVusv added using Eq.A6

    def sigma(self, min_val, max_val, x_theta, k_theta):
        return (min_val + max_val * sp.functions.exp(x_theta / k_theta)) / (1 + sp.functions.exp(x_theta / k_theta))

    def Plv(self):
        return self.Pmaxlv() - self.Rlv() * self.func(self.Pmaxlv(), y(4), self.Rlv())

    def Prv(self):
        return self.Pmaxrv() - self.Rrv() * self.func(self.Pmaxrv(), y(0), self.Rrv())

    def ST(self):
        VLDts = 1900 - 100 * input(0, t - self.param.params['Dts'])
        return self.param.params['GaTs'] * (y(4, t-self.param.params['Dts']) - self.param.params['Psan']) + self.param.params['GpTs'] * (VLDts - self.param.params['VLn'])
    
    def VT(self):
        VLDtv = 1900 - 100 * input(0, t - self.param.params['Dtv'])
        return self.param.params['GaTv'] * (y(4, t-self.param.params['Dtv']) - self.param.params['Psan']) - self.param.params['GpTv'] * (VLDtv - self.param.params['VLn']) - self.param.params['G1Tv']