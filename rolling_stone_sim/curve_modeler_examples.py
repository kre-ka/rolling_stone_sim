from typing import Dict
import sympy as sym
from math import pi

from curve_modeler import *


def cos_sin_modeler(slider_params_override: None|Dict[str, Dict[str, float]]=None) -> CurveModeler:
    '''
    This is pretty general modeler, you can use slider_params_override parameter to make more specific cases of it (see circle_modeler())
    '''
    t = sym.Symbol('t', real=True)
    f_params = dict(zip(('Ax Ay wx wy'.split()), sym.symbols('A_x A_y w_x w_y', real=True)))
    x = f_params['Ax']*sym.cos(f_params['wx']*t)
    y = f_params['Ay']*sym.sin(f_params['wy']*t)
    slider_params = {'Ax': {'min': -3, 'max': 3, 'step': 0.1, 'value': 0.3},
                     'Ay': {'min': 0.1, 'max': 3, 'step': 0.1, 'value': 2},
                     'wx': {'min': -5, 'max': 5, 'step': 0.1, 'value': 3},
                     'wy': {'min': -5, 'max': 5, 'step': 0.1, 'value': 1},
                     't':  {'min':-pi, 'max':pi, 'step': pi*0.05, 'value': (-2.8, -pi+2.8)}}
    
    # this section handles overriding slider parameters (it isn't mandatory)
    f_param_subs = {}
    if slider_params_override:
        for slider_key, slider_value in slider_params_override.items():
            if slider_key in slider_params.keys():
                for param_key, param_value in slider_value.items():
                    if param_key == 'constant':
                        del slider_params[slider_key]
                        f_param_subs[slider_key] = param_value
                        continue
                    if param_key in slider_params[slider_key].keys():
                        slider_params[slider_key][param_key] = param_value
    # substitute variables for constants
    for key, value in f_param_subs.items():
            x = x.subs(f_params[key], value)
            y = y.subs(f_params[key], value)
            del f_params[key]
    
    # this one is mandatory
    return CurveModeler(t, f_params, x, y, slider_params)

def circle_modeler() -> CurveModeler:
    # you can override parameters of a more general cos_sin_modeler
    # you only need to use the ones you want to change
    # use key 'constant' with constant value you wish to substitute parameter with (this will remove the slider for that parameter)
    slider_params_override = {'Ax': {'value': 1},
                              'Ay': {'value': 1},
                              'wx': {'constant': 1},
                              'wy': {'constant': 1},
                              't':  {'min':(0.5+0.05)*pi, 'max':(2.5-0.05)*pi, 'value': ((0.5+0.05)*pi, (2.5-0.05)*pi)}}
    return cos_sin_modeler(slider_params_override)

def teardrop_modeler() -> CurveModeler:
    slider_params_override = {'wx': {'min': 1},
                              'wy': {'constant': 1},
                              't': {'min': -(1.5-0.05)*pi, 'max': (0.5-0.05)*pi}}
    return cos_sin_modeler(slider_params_override)


if __name__ == '__main__':
    circle_modeler()