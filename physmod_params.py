import numpy as np
import logging

class PhysmodParams:
    """
    Container for physiology parameters which are either calculated from blood
    volume or are set constant. Constants are initialized automatically when
    this object is created.

    Matlab code reference: heartGeneral_21S_data_6P.m
    """

    def __init__(self):
        self.param_names = ['Cpa', 'Lpa', 'Cpp', 'Cpv', 'Csa', 'Lsa', 'Csp', 'Cep', 'Cev',
                        'Cla', 'Cra', 'Krrv', 'Rpa', 'Rpp', 'Rpv', 'Krlv', 'Rsa', 'Rra',
                        'Vusp', 'Vuep', 'Vura', 'Vupp', 'Vupv', 'Vula', 'Rev', 'Rsv', 'Tsys0',
                        'Ksys', 'Vurv', 'Vulv', 'P0lv', 'P0rv', 'Kelv', 'Kerv', 'Csv', 'Rla',
                        'Vblood', 'GaTv', 'GpTv', 'GaTs', 'tauTv', 'tauTs', 'Psan', 'VLn', 'Tmin',
                        'Tmax', 'GaElv', 'Elvmax', 'Elvmin', 'tauElv', 'GaErv', 'Ervmax', 'Ervmin', 'tauErv',
                        'GaRsp', 'GpRsp', 'Rspmax', 'Rspmin', 'tauRsp', 'GaRep', 'GpRep', 'Repmax', 'Repmin', 'tauRep',
                        'GaVusv', 'Vusvmax', 'Vusvmin', 'tauVusv', 'GaVuev', 'Vuevmax', 'Vuevmin', 'tauVuev',
                        'Dtv', 'Dts', 'DElv', 'DErv', 'DVusv', 'DVuev', 'DRep', 'DRsp']

        self.params = {pn: -1 for pn in self.param_names}
        self._const_keys = ['Vusa', 'Vupa', 'GpTs', 'GpElv', 'Selv0', 'GpErv',
                            'Serv0', 'Srsp0', 'Srep0', 'GpVusv', 'Svusv0', 'GpVuev',
                            'Svuev0', 'St0', 'A', 'G1Tv', 'G2Rsp', 'G3Rep']
        self._dep_keys = ['Cpa', 'Cpp', 'Cpv', 'Csa', 'Csp', 'Cep', 'Cev',
                          'Csv', 'Vusv', 'Vuep', 'Vupp', 'Vupv', 'Vusvmax', 'Vusvmin', 
                          'Vuevmax', 'Vuevmin', 'Rpa', 'Rpp', 'Rpv', 'Rsa', 'Rev', 'Rsv',
                          'Rspmax', 'Rspmin', 'Repmax', 'Repmin', 'Lpa', 'Lsa']
        self.initialize_constants()

    def isconst(self, param_key):
        if param_key in self._const_keys:
            return True
        elif param_key in self.param_names:
            return False
        else:
            raise Exception('Non existant parameter key.')

    def get_delay_dict(self):
        return {key: self.params[key] for key in self.params.keys() if key[0] == 'D'}

    def get_delay_vector(self):
        # ordered properly
        return np.array([self.params['Dtv'], self.params['Dts'], self.params['DElv'],
                        self.params['DErv'], self.params['DVusv'], self.params['DVuev'],
                        self.params['DRep'], self.params['DRsp']])

    def read_from_file(self, file_path):
        param_data = np.loadtxt(file_path)
        logging.debug(param_data)
        for i, n in enumerate(self.param_names):
            self.params[n] = param_data[i]
    
    def get_max_delay(self):
        max_delay = np.max(list(self.get_delay_dict().values()))
        print('Max. delay: %.2f' % (max_delay))
        return max_delay

    def calculate_vblood(self, cfg):
        if cfg.sex == 'F':
            Vblood = 1000*(0.3561*(cfg.height / 100.0)**3 + 0.03308*cfg.weight + 0.1833)
        else:
            Vblood = 1000*(0.3669*(cfg.height/100.0)**3 + 0.03219*cfg.weight + 0.6041)

        self.params['Vblood'] = Vblood

    def initialize_dependants(self, cfg):
        if cfg.weight < 50:
            vv = 0.8 * self.params['Vblood']/5300
        else:
            vv = self.params['Vblood']/5300

        self.params['Cpa'] = 0.76*vv
        self.params['Cpp'] = 5.80*vv
        self.params['Cpv'] = 25.37*vv
        self.params['Csa'] = 0.28*vv
        self.params['Csp'] = 2.05*vv
        self.params['Cep'] = 1.67*vv
        self.params['Cev'] = 50.0*vv
        self.params['Csv'] = 61.11*vv
        self.params['Vusp'] = 274.4*vv
        self.params['Vuep'] = 336.6*vv
        self.params['Vupp'] = 123*vv
        self.params['Vupv'] = 120*vv
        self.params['Vusvmax'] = 1371*vv
        self.params['Vusvmin'] = 871*vv
        self.params['Vuevmax'] = 1475*vv
        self.params['Vuevmin'] = 1275*vv
        self.params['Rpa'] = 0.023/vv
        self.params['Rpp'] = 0.0894/vv
        self.params['Rpv'] = 0.0056/vv
        self.params['Rsa'] = 0.06/vv
        self.params['Rev'] = 0.016/vv
        self.params['Rsv'] = 0.038/vv
        self.params['Rspmax'] = 4.5/vv
        self.params['Rspmin'] = 2.12/vv
        self.params['Repmax'] = 1.9/vv
        self.params['Repmin'] = 0.91/vv
        self.params['Lpa'] = 0.18e-3/vv
        self.params['Lsa'] = 0.22e-3/vv

        task_id = cfg.task_id

        if task_id == '6P':
            if cfg.sex == 'F':
                VLn = 1000 * (1.81 * cfg.height/100 + 0.016 * cfg.age - 2.00)
            else:
                VLn = 1000 * (1.31 * cfg.height/100 + 0.022 * cfg.age - 1.23)
        else:
            if cfg.sex == 'F':
                VLn = 1000 * (2.24 * cfg.height/100 + 0.001 * cfg.age - 1.00)
            else:
                VLn = 1000 * (2.34 * cfg.height/100 + 0.01 * cfg.age - 1.09)

        self.params['VLn'] = VLn

        self.params['Psan'] = cfg.BP_mean

        self.params['Tmin'] = 0.001 * np.min(cfg.HP[1])
        self.params['Tmax'] = 0.001 * np.max(cfg.HP[1])

    def initialize_constants(self):
        self.params['Cla'] = 19.23
        self.params['Cra'] = 31.25
        self.params['Krrv'] = 1.4e-3
        self.params['Krlv'] = 3.75e-4
        self.params['Rra'] = 2.5e-3
        self.params['Vura'] = 25
        self.params['Vula'] = 25
        self.params['Tsys0'] = 0.5
        self.params['Ksys'] = 0.075
        self.params['Vurv'] = 40.8
        self.params['Vulv'] = 16.77
        self.params['P0lv'] = 1.5
        self.params['P0rv'] = 1.5
        self.params['Kelv'] = 0.014
        self.params['Kerv'] = 0.011
        self.params['Rla'] = 2.5e-3
        self.params['GaTv'] = 0.028
        self.params['GpTv'] = 0.25e-3
        self.params['GaTs'] = 0.015
        self.params['tauTv'] = 0.8
        self.params['tauTs'] = 1.8
        self.params['Psan'] = 95
        self.params['VLn'] = 2300
        self.params['GaElv'] = 0.012
        self.params['Elvmax'] = 3.45
        self.params['Elvmin'] = 2.45
        self.params['tauElv'] = 1.5
        self.params['GaErv'] = 0.012
        self.params['Ervmax'] = 2.25
        self.params['Ervmin'] = 1.25
        self.params['tauErv'] = 1.5
        self.params['GaRsp'] = 0.1
        self.params['GpRsp'] = 0.33e-3
        self.params['tauRsp'] = 1.5
        self.params['GaRep'] = 0.1
        self.params['GpRep'] = 0.33e-3
        self.params['tauRep'] = 1.5
        self.params['GaVusv'] = 10.8
        self.params['tauVusv'] = 10
        self.params['GaVuev'] = 5.5
        self.params['tauVuev'] = 10
        self.params['Dtv'] = 0.5
        self.params['Dts'] = 3
        self.params['DElv'] = 2
        self.params['DErv'] = 2
        self.params['DVusv'] = 5
        self.params['DVuev'] = 5
        self.params['DRep'] = 3
        self.params['DRsp'] = 3

        self.params['G1Tv'] = 0
        self.params['G2Rsp'] = 0
        self.params['G3Rep'] = 0

        self.params['Vusa'] = 0
        self.params['Vupa'] = 0
        self.params['GpTs'] = 0
        self.params['GpElv'] = 0
        self.params['Selv0'] = -1
        self.params['GpErv'] = 0
        self.params['Serv0'] = -1
        self.params['Srsp0'] = -1
        self.params['Srep0'] = -1
        self.params['GpVusv'] = 0
        self.params['Svusv0'] = 1
        self.params['GpVuev'] = 0
        self.params['Svuev0'] = 1
        self.params['St0'] = 1
        self.params['A'] = 0
