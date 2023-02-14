import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp


# base class, don't use it on its own
class _Slope:
    def __init__(self) -> None:
        # define symbols
        self._t = sym.Symbol('t', nonnegative=True)
        self._x = sym.Function('x')(self._t)
        self._g = sym.Symbol('g', real=True)
    
    def calc_equations(self):
        '''
        Calculates y(x) and dynamics equations to be used by numerical solver.

        Call this at the end of inherited constructor'''
        # make a numerical version of the y(x) equation
        self._y_f = sym.lambdify([self._x,self._slope_params], self._y)

        # lagrange dynamics
        # kinetic energy
        T = 0.5 * self._x.diff(self._t)**2 + self._y.diff(self._t)**2
        # potential energy
        V = self._g * self._y
        # lagrangian
        L = T - V
        # lagrange-euler eq: d(dL/dx')/dt = dL/dx
        LE = (L.diff(self._x.diff(self._t)).diff(self._t) - L.diff(self._x))

        # dynamics eq: x''
        # this is a 2nd order ODE
        dyn_eq = sym.solve(LE, sym.diff(self._x,self._t,self._t))[0].simplify()

        # turn 2nd order dyn_eq into system of 2 1st order ODEs with additional variable v
        v = sym.Function('v')(self._x)
        # x' = v
        # v' = x'' = dyn_eq
        dyn_eq = [v, dyn_eq.subs(self._x.diff(self._t),v)]
        # make the numerical version of the equation
        # t must go first, then state vector (x,v), then parameters
        # left side of the equation is derivative of state i.g.(x',v')
        self._dyn_eq_f = sym.lambdify([self._t,(self._x,v),self._slope_params,self._g], dyn_eq)
    
    # initial_conditions = (x(0), x'(0))
    # slope_params = (A, B, C)
    # g - gravity constant
    # returns (t, x, y) vectors
    def simulate(self, initial_conditions, slope_params, g):
        # start and stop times
        t_0 = 0
        t_f = t_0 + 10
        # evaluation times vector
        t = np.linspace(t_0, t_f, 100)
        # solve dynamics equation numerically with given parameters
        sol = solve_ivp(self._dyn_eq_f, (0, 10), initial_conditions, args=(slope_params, g), t_eval=t)
        print(sol)
        x = sol.y[0]
        y = self._y_f(x, slope_params)
        return t, x, y

class LinearSlope(_Slope):
    '''Utilizes function y = Ax + B'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_params = sym.symbols('A B', real=True)
        # define y(x)
        self._y = self._slope_params[0]*self._x + self._slope_params[1]

        self.calc_equations()
    
    def simulate(self, initial_conditions, slope_params, g):
        '''slope_params = (A, B)'''
        return super().simulate(initial_conditions, slope_params, g)

class QuadraticSlope(_Slope):
    '''Utilizes function y = Ax**2 + Bx + C'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_params = sym.symbols('A B C', real=True)
        # define y(x)
        self._y = self._slope_params[0]*self._x**2 + self._slope_params[1]*self._x + self._slope_params[2]

        self.calc_equations()

    def simulate(self, initial_conditions, slope_params, g):
        '''slope_params = (A, B, C)'''
        return super().simulate(initial_conditions, slope_params, g)