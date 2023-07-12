import sympy as sym
from math import pi

from curve_modeler import *


def cos_sin_modeler() -> CurveModeler:
    t = sym.Symbol('t', real=True)
    f_params = dict(zip(('Ax Ay wx wy'.split()), sym.symbols('A_x A_y w_w w_y', real=True)))
    x = f_params['Ax']*sym.cos(f_params['wx']*t)
    y = f_params['Ay']*sym.sin(f_params['wy']*t)
    slider_params = {'Ax': {'min': -3, 'max': 3, 'step': 0.1, 'value': 0.3},
                     'Ay': {'min': -3, 'max': 3, 'step': 0.1, 'value': 2},
                     'wx': {'min': -5, 'max': 5, 'step': 0.1, 'value': 3},
                     'wy': {'min': -5, 'max': 5, 'step': 0.1, 'value': 1},
                     't':  {'min':-pi, 'max':pi, 'step': pi*0.05, 'value': (-2.8, -pi+2.8)}}
    return CurveModeler(t, f_params, x, y, slider_params)
