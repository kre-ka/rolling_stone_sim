import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets

# base class, don't use it on its own
class _Slope:
    def __init__(self) -> None:
        # define symbols
        self._t = sym.Symbol('t', nonnegative=True)
        self._x = sym.Function('x')(self._t)
        self._g = sym.Symbol('g', real=True)
    
    def calc_equations(self):
        '''
        Calculates equations to be used by numerical solver.

        Call this at the end of inherited constructor
        '''
        self.calc_dyn_equations()
        self.calc_y_equations()

    def calc_dyn_equations(self):
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
    
    def calc_y_equations(self):
        '''
        gives numerical version of y, y' and y'' equations
        
        self._y_f = y(x, slope_coef)
        self._y_full_f = (y, y', y'') = y((x, x', x''), slope_coef)
        '''
        self._y_f = sym.lambdify([self._x, self._slope_coef], self._y)
        # y = [y, y', y'']
        y = [self._y, self._y.diff(self._t), self._y.diff(self._t, self._t)]
        # x = [x, x', x'']
        x = sym.symbols('x x\' x\'\'', cls=sym.Function)
        x = [x_i(self._t) for x_i in x]
        # substitute Derivative(x,t) with x' and x'' in symbolic expressions to allow lambdify
        y = [y_i.subs(self._x.diff(self._t, self._t), x[2])\
                .subs(self._x.diff(self._t), x[1])\
                for y_i in y]
        self._y_full_f = sym.lambdify([(x[0], x[1], x[2]), self._slope_coef], y)

    def set_slope_coef(self, slope_coef, slope_length):
        '''
        expand this method in child class to support more user-friendly parameters
        
        slope_coef: tuple of slope polynomial coefficient values

        slope_length: x coordinate of finish point
        '''
        # 'n' stands for 'numerical' - used to distinguish from abstract coefficients in self._slope_coef 
        self._slope_coef_n = slope_coef
        self._xf = slope_length

    def simulate(self, g=9.81, t_max=30, t_res=10, terminate=True):
        '''
        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 10 [Hz]

        returns (t, (x, x', x''), (y, y', y''), tf) - movement vectors and finish time
        '''
        # stop on reaching x=xf
        def event_finish(t, x, slope_coef, g):
            return x[0] - self._xf
        event_finish.terminal = True
        # evaluation times vector
        t = np.linspace(0, t_max, int(t_max*t_res))
        initial_conditions = (0, 0)  # (x(0) [m], x'(0) [m/s])
        # solve dynamics equation numerically with given parameters
        if terminate:
            sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, method='Radau', args=(self._slope_coef_n, g), t_eval=t, events=[event_finish])
        else:
            sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, method='Radau', args=(self._slope_coef_n, g), t_eval=t)
        # print(sol)
        t = sol.t
        # this gives x and x'
        x = sol.y
        # this gives x''
        a = self._dyn_eq_f(t, x, self._slope_coef_n, g)[1]
        # concatenate a into x for array (x, x', x'')
        if not isinstance(a, np.ndarray):
            a = np.full(len(x[0]), a)
        x = np.row_stack((x,a))
        y = self._y_full_f(x, self._slope_coef_n)
        if terminate:
            tf = sol.t_events[0][0]
        else:
            tf = None
        return t, x, y, tf
    

class LinearSlope(_Slope):
    '''Utilizes function y = Ax + B'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_coef = sym.symbols('A B', real=True)
        # define y(x)
        self._y = self._slope_coef[0]*self._x + self._slope_coef[1]

        self.calc_equations()
    
    def set_slope_coef(self, height, length):
        '''
        Slope goes from point (0, height) to (length, 0)
        '''
        B = height
        A = -B / length
        super().set_slope_coef((A, B), length)
    
    def model_slope(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        ax.set_aspect("equal", adjustable="datalim")
        fig.canvas.header_visible = False
        
        lines = plt.plot(0,0)
        plt.axhline(color='black', ls='--', lw=1, alpha=0.5)
        plt.axvline(color='black', ls='--', lw=1, alpha=0.5)
        plt.ion()

        def plot(height=1.0, length=1.0):
            self.set_slope_coef(height, length)
            x = np.linspace(0, length, int(10 * length))
            y = self._y_f(x, self._slope_coef_n)
            lines[0].set_data(x, y)
            ax.relim()
            ax.autoscale_view()
        
        widgets.interact(plot, height=(0.1, 2.0, 0.1), length=(0.1, 2.0, 0.1))

    def simulate(self, g=9.81, t_max=30, t_res=10, terminate=True):
        '''
        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 10 [Hz]

        returns (t, (x, x', x''), (y, y', y''), tf) - movement vectors and finish time
        '''
        return super().simulate(g, t_max, t_res, terminate)


class QuadraticSlope(_Slope):
    '''Utilizes function y = Ax**2 + Bx + C'''

    def __init__(self) -> None:
        super().__init__()

        # define slope parameters
        self._slope_coef = sym.symbols('A B C', real=True)
        # define y(x)
        self._y = self._slope_coef[0]*self._x**2 + self._slope_coef[1]*self._x + self._slope_coef[2]

        self.calc_equations()

    def set_slope_coef(self, height, length, steepness):
        '''
        Slope goes from point (0, height) to (length, 0)

        steepness - float in range [0, 1), higher value - steeper slope (0 makes linear slope)
        '''
        C = height
        B = - height/(length*(1-steepness))
        A = - steepness*B/length
        super().set_slope_coef((A, B, C), length)
    
    def model_slope(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        ax.set_aspect("equal", adjustable="datalim")
        fig.canvas.header_visible = False
        
        lines = plt.plot(0,0)
        plt.axhline(color='black', ls='--', lw=1, alpha=0.5)
        plt.axvline(color='black', ls='--', lw=1, alpha=0.5)
        plt.ion()

        def plot(height=1.0, length=1.0, steepness=0.5):
            self.set_slope_coef(height, length, steepness)
            x = np.linspace(0, length, int(20 * length))
            y = self._y_f(x, self._slope_coef_n)
            lines[0].set_data(x, y)
            ax.relim()
            ax.autoscale_view()
        
        widgets.interact(plot, height=(0.1, 2.0, 0.1), length=(0.1, 2.0, 0.1), steepness=(0.0, 0.9, 0.01))

    def simulate(self, g=9.81, t_max=30, t_res=10, terminate=True):
        '''
        g - gravity constant [m/s**2]

        t_max - max simulation time [s]

        t_res - time resolution [Hz]

        returns (t, (x, x', x''), (y, y', y''), tf) - movement vectors and finish time
        '''
        return super().simulate(g, t_max, t_res, terminate)


# may not work in real time if time resolution is too large
def plot_sim_results(t, x, y):
    # make axis limits a little bigger than necessary
    def expand_limits(lim, amount):
        additional_range = (lim[1] - lim[0]) * amount
        if additional_range == 0:
            additional_range = 1 * amount
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
    
    x, x_dot, x_dot_dot = x
    y, y_dot, y_dot_dot = y

    v = (x_dot**2 + y_dot**2)**0.5
    a = (x_dot_dot**2 + y_dot_dot**2)**0.5

    t_plt = []
    x_plt = []
    x_dot_plt = []
    x_dot_dot_plt = []
    y_plt = []
    y_dot_plt = []
    y_dot_dot_plt = []
    v_plt = []
    a_plt = []

    fig, axs = plt.subplots(3,3)
    fig.set_size_inches(10,10)
    fig.canvas.header_visible = False

    # make static plot limits
    t_lim = expand_limits((t.min(), t.max()), 0.05)
    x_lim = expand_limits((x.min(), x.max()), 0.05)
    x_dot_lim = expand_limits((x_dot.min(), x_dot.max()), 0.05)
    x_dot_dot_lim = expand_limits((x_dot_dot.min(), x_dot_dot.max()), 0.05)
    y_lim = expand_limits((y.min(), y.max()), 0.05)
    y_dot_lim = expand_limits((y_dot.min(), y_dot.max()), 0.05)
    y_dot_dot_lim = expand_limits((y_dot_dot.min(), y_dot_dot.max()), 0.05)
    v_lim = expand_limits((v.min(), v.max()), 0.05)
    a_lim = expand_limits((a.min(), a.max()), 0.05)

    # x(t)
    axs[0][0].set_xlabel('t')
    axs[0][0].set_ylabel('x')
    axs[0][0].set_xlim(t_lim)
    axs[0][0].set_ylim(x_lim)
    line_xt, = axs[0][0].plot(t_plt, x_plt)

    # x'(t)
    axs[1][0].set_xlabel('t')
    axs[1][0].set_ylabel('x\'')
    axs[1][0].set_xlim(t_lim)
    axs[1][0].set_ylim(x_dot_lim)
    line_x_dot_t, = axs[1][0].plot(t_plt, x_dot_plt)

    # x''(t)
    axs[2][0].set_xlabel('t')
    axs[2][0].set_ylabel('x\'\'')
    axs[2][0].set_xlim(t_lim)
    axs[2][0].set_ylim(x_dot_dot_lim)
    line_x_dot_dot_t, = axs[2][0].plot(t_plt, x_dot_dot_plt)

    # y(t)
    axs[0][1].set_xlabel('t')
    axs[0][1].set_ylabel('y')
    axs[0][1].set_xlim(t_lim)
    axs[0][1].set_ylim(y_lim)
    line_yt, = axs[0][1].plot(t_plt, y_plt)

    # y'(t)
    axs[1][1].set_xlabel('t')
    axs[1][1].set_ylabel('y\'')
    axs[1][1].set_xlim(t_lim)
    axs[1][1].set_ylim(y_dot_lim)
    line_y_dot_t, = axs[1][1].plot(t_plt, y_dot_plt)

    # y''(t)
    axs[2][1].set_xlabel('t')
    axs[2][1].set_ylabel('y\'\'')
    axs[2][1].set_xlim(t_lim)
    axs[2][1].set_ylim(y_dot_dot_lim)
    line_y_dot_dot_t, = axs[2][1].plot(t_plt, x_dot_dot_plt)

    # y(x)
    x_lim, y_lim = equalize_axis_scales(x_lim, y_lim)
    axs[0][2].set_xlabel('x')
    axs[0][2].set_ylabel('y')
    axs[0][2].set_xlim(x_lim)
    axs[0][2].set_ylim(y_lim)
    axs[0][2].axis("scaled")
    line_yx, = axs[0][2].plot(x, y)
    point_yx, = axs[0][2].plot(x_plt, y_plt, 'o')

    # # v(x)
    # x_lim, v_lim = equalize_axis_scales(x_lim, v_lim)
    # axs[1][2].set_xlabel('x')
    # axs[1][2].set_ylabel('v')
    # axs[1][2].set_xlim(x_lim)
    # axs[1][2].set_ylim(v_lim)
    # axs[1][2].axis("scaled")
    # line_vx, = axs[1][2].plot(x, v)
    # point_vx, = axs[1][2].plot(x_plt, v_plt, 'o')

    # # a(x)
    # x_lim, a_lim = equalize_axis_scales(x_lim, a_lim)
    # axs[2][2].set_xlabel('x')
    # axs[2][2].set_ylabel('a')
    # axs[2][2].set_xlim(x_lim)
    # axs[2][2].set_ylim(a_lim)
    # axs[2][2].axis("scaled")
    # line_ax, = axs[2][2].plot(x, a)
    # point_ax, = axs[2][2].plot(x_plt, a_plt, 'o')

    # v(t)
    axs[1][2].set_xlabel('t')
    axs[1][2].set_ylabel('v')
    axs[1][2].set_xlim(t_lim)
    axs[1][2].set_ylim(v_lim)
    line_vx, = axs[1][2].plot(t_plt, v_plt)

    # a(t)
    axs[2][2].set_xlabel('t')
    axs[2][2].set_ylabel('a')
    axs[2][2].set_xlim(t_lim)
    axs[2][2].set_ylim(a_lim)
    line_ax, = axs[2][2].plot(t_plt, a_plt)

    plt.tight_layout()
    plt.ion()

    def animate(i):
        t_plt.append(t[i])
        x_plt.append(x[i])
        x_dot_plt.append(x_dot[i])
        x_dot_dot_plt.append(x_dot_dot[i])
        y_plt.append(y[i])
        y_dot_plt.append(y_dot[i])
        y_dot_dot_plt.append(y_dot_dot[i])
        v_plt.append(v[i])
        a_plt.append(a[i])

        line_xt.set_xdata(t_plt)
        line_xt.set_ydata(x_plt)

        line_x_dot_t.set_xdata(t_plt)
        line_x_dot_t.set_ydata(x_dot_plt)

        line_x_dot_dot_t.set_xdata(t_plt)
        line_x_dot_dot_t.set_ydata(x_dot_dot_plt)

        line_yt.set_xdata(t_plt)
        line_yt.set_ydata(y_plt)

        line_y_dot_t.set_xdata(t_plt)
        line_y_dot_t.set_ydata(y_dot_plt)

        line_y_dot_dot_t.set_xdata(t_plt)
        line_y_dot_dot_t.set_ydata(y_dot_dot_plt)

        point_yx.set_xdata(x[i])
        point_yx.set_ydata(y[i])

        # point_vx.set_xdata(x[i])
        # point_vx.set_ydata(v[i])

        # point_ax.set_xdata(x[i])
        # point_ax.set_ydata(a[i])

        line_vx.set_xdata(t_plt)
        line_vx.set_ydata(v_plt)

        line_ax.set_xdata(t_plt)
        line_ax.set_ydata(a_plt)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    return FuncAnimation(plt.gcf(), animate, frames=len(x), interval=int((t[1]-t[0])*1000), repeat=False)