import numpy as np


class PhysmodState:
    def __init__(self):
        self._vars = ['Ppa', 'Fpa', 'Ppp', 'Ppv', 'Psa', 'Fsa', 'Psp',
                      'Pev', 'Pla', 'Pra', 'Vrv', 'Vlv',  'xi', 'xTs',
                      'xTv', 'Emaxlv', 'Emaxrv', 'Rsp', 'Rep', 'Vusv', 'Vuev']
        self._phys_state = None
        self._current_state = None
        self.Tinst = []
        self.tout = []

    def init_state(self, state):
        if (state.size != len(self._vars)):
            raise ValueError("State vector variable count doesn't match the expected count!")
        self._phys_state = state.reshape((len(state), 1))
        self._phys_state[:, 0] = state

    def get_vector(self, var):
        var_id = self._vars.index(var)
        return self._phys_state[var_id]

    def get_state(self, i=-1):
        return np.array(self._phys_state[:, i])

    def get(self, var, i=-1):
        return self._phys_state[self._vars.index(var), i]

    def set(self, var, val, i=-1):
        self._phys_state[self._vars.index(var)][i] = val

    def varindex(self, var):
        return self._vars.index(var)

    def calculate_t(self, eqs, param):
        kT = (param.params['Tmax'] - param.params['Tmin']) / (4 * param.params['St0'])
        xT = self.get_vector('xTv') + self.get_vector('xTs')
        self.Tinst = (param.params['Tmin'] + param.params['Tmax'] * np.exp(xT / kT)) / (1 + np.exp(xT / kT))
