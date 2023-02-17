import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        self._y_f = sym.lambdify([self._x,self._slope_coef], self._y)

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
        self._dyn_eq_f = sym.lambdify([self._t,(self._x,v),self._slope_coef,self._g], dyn_eq)
    
    def simulate(self, slope_coef, xf, g=9.81, t_max=30, t_res=10):
        '''
        slope_coef - coefficients of a slope polynomial, can vary between child classes

        xf - finish x [m] - simulation terminates on reaching x=xf

        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 10 [Hz]

        returns (t, x, y, tf) - movement vectors and finish time
        '''
        # stop on reaching x=xf
        def event_finish(t, x, slope_coef, g):
            return x[0] - xf
        event_finish.terminal = True
        # evaluation times vector
        t = np.linspace(0, t_max, int(t_max*t_res))
        initial_conditions = (0, 0)  # (x(0) [m], x'(0) [m/s])
        # solve dynamics equation numerically with given parameters
        sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, args=(slope_coef, g), t_eval=t, events=[event_finish])
        # print(sol)
        t = sol.t
        x = sol.y[0]
        y = self._y_f(x, slope_coef)
        tf = sol.t_events[0][0]
        return t, x, y, tf
    
    def _event_finish(t, y):
        return y

class LinearSlope(_Slope):
    '''Utilizes function y = Ax + B'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_coef = sym.symbols('A B', real=True)
        # define y(x)
        self._y = self._slope_coef[0]*self._x + self._slope_coef[1]

        self.calc_equations()
    
    def simulate(self, slope_params, g=9.81, t_max=30, t_res=10):
        '''

        slope_params = (y0, xf) - start and finish point coordinates i.e. (0, y0), (xf, 0) [m]

        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 10 [Hz]

        returns (t, x, y) vectors
        '''
        y0, xf = slope_params
        B = y0
        A = -B / xf
        return super().simulate((A, B), xf, g, t_max, t_res)


class QuadraticSlope(_Slope):
    '''Utilizes function y = Ax**2 + Bx + C'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_coef = sym.symbols('A B C', real=True)
        # define y(x)
        self._y = self._slope_coef[0]*self._x**2 + self._slope_coef[1]*self._x + self._slope_coef[2]

        self.calc_equations()

    def simulate(self, slope_params, g, t_max, t_res):
        '''
        initial_conditions = (x(0), x'(0)) ([m], [m/s])

        slope_params = (y0, xf, steepness) - start and finish point coordinates i.e. (0, y0), (xf, 0) [m],
                                             slope steepness parameter in range [0, 1)

        g - gravity constant [m/s**2]

        t_max - max simulation time [s]

        t_res - time resolution [Hz]

        returns (t, x, y) vectors
        '''
        y0, xf, s = slope_params
        C = y0
        B = - y0/(xf*(1-s))
        A = - s*B/xf
        return super().simulate((A, B, C), xf, g, t_max, t_res)


# may not work in real time if time resolution is too large
def plot_sim_results(t, x, y):
    # make axis limits a little bigger than necessary
    def expand_limits(lim, amount):
        additional_range = (lim[1] - lim[0]) * amount
        return (lim[0] - additional_range, lim[1] + additional_range)
    
    # fix axis limits so scale is the same on both axes
    def equalize_axis_scales(x_lim, y_lim):
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        if x_range > y_range:
            y_middle = (y_lim[0] + y_lim[1]) / 2
            y_lim = (y_middle - x_range/2, y_middle + x_range/2)
        elif y_range > x_range:
            x_middle = (x_lim[0] + x_lim[1]) / 2
            x_lim = (x_middle - y_range/2, x_middle + y_range/2)
        return x_lim, y_lim
    
    t_plt = []
    x_plt = []
    y_plt = []
    fig, axs = plt.subplots(1,3)

    # make static plot limits
    t_lim = expand_limits((t.min(), t.max()), 0.05)
    x_lim = expand_limits((x.min(), x.max()), 0.05)
    y_lim = expand_limits((y.min(), y.max()), 0.05)

    # x(t)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')
    axs[0].set_xlim(t_lim)
    axs[0].set_ylim(x_lim)
    line_xt, = axs[0].plot(t_plt, x_plt)

    # y(t)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('y')
    axs[1].set_xlim(t_lim)
    axs[1].set_ylim(y_lim)
    line_yt, = axs[1].plot(t_plt, y_plt)

    # y(x)
    x_lim, y_lim = equalize_axis_scales(x_lim, y_lim)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_xlim(x_lim)
    axs[2].set_ylim(y_lim)
    axs[2].axis("scaled")
    line_yx, = axs[2].plot(x, y)
    point_yx, = axs[2].plot(x_plt, y_plt, 'o')

    def animate(i):
        t_plt.append(t[i])
        x_plt.append(x[i])
        y_plt.append(y[i])

        line_xt.set_xdata(t_plt)
        line_xt.set_ydata(x_plt)

        line_yt.set_xdata(t_plt)
        line_yt.set_ydata(y_plt)

        point_yx.set_xdata(x[i])
        point_yx.set_ydata(y[i])

        fig.canvas.draw()
        fig.canvas.flush_events()

    ani = FuncAnimation(plt.gcf(), animate, frames=len(x), interval=int((t[1]-t[0])*1000), repeat=False)
    plt.show()