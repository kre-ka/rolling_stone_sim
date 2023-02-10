import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp


class LinearSlope:
    def __init__(self) -> None:
        # define symbols
        t = sym.Symbol('t', nonnegative=True)
        x = sym.Function('x')(t)
        A, B = sym.symbols('A B', real=True)
        g = sym.Symbol('g', real=True)

        # define y(x)
        y = A*x + B
        # make a numerical version of the equation
        self._y_f = sym.lambdify([x,A,B], y)

        # lagrange dynamics
        # kinetic energy
        T = 0.5 * x.diff(t)**2 + y.diff(t)**2
        # potential energy
        V = g * y
        # lagrangian
        L = T - V
        # lagrange-euler eq: d(dL/dx')/dt = dL/dx
        LE = (L.diff(x.diff(t)).diff(t) - L.diff(x)).simplify()

        # dynamics eq: x''
        # this is a 2nd order ODE
        dyn_eq = sym.solve(LE, sym.diff(x,t,t))[0]

        # turn 2nd order dyn_eq into system of 2 1st order ODEs with additional variable v
        v = sym.Function('v')(x)
        # x' = v
        # v' = x'' = dyn_eq
        dyn_eq = [v, dyn_eq]
        # make the numerical version of the equation
        # t must go first, then state vector (x,v), then parameters
        # left side of the equation is derivative of state i.g.(x',v')
        self._dyn_eq_f = sym.lambdify([t,(x,v),A,B,g], dyn_eq)
    
    # initial_conditions = (x(0), x'(0))
    # params = (A, B, g)
    # returns (t, x, y) vectors
    def simulate(self, initial_conditions, params):
        # start and stop times
        t_0 = 0
        t_f = t_0 + 10
        # evaluation times vector
        t = np.linspace(t_0, t_f, 100)
        # solve dynamics equation numerically with given parameters
        sol = solve_ivp(self._dyn_eq_f, (0, 10), initial_conditions, args=params, t_eval=t)
        print(sol)
        x = sol.y[0]
        y = self._y_f(x, params[0], params[1])
        return t, x, y
