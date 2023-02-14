from sim import LinearSlope, QuadraticSlope
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


if __name__ == '__main__':
    # create a simulator instance
    sim = LinearSlope()
    # set initial conditions
    # (x(0), x'(0))
    initial_conditions = (0,0)
    # set slope parameters
    # for descriptions see slope object and simulate method docstrings
    slope_params = (-0.5, 0)
    # simulate movement using given parameters
    t, x, y = sim.simulate(initial_conditions, slope_params, 9.81)

    # sim = QuadraticSlope()
    # initial_conditions = (0,0)
    # slope_params = (0.1, -2, 10)
    # t, x, y = sim.simulate(initial_conditions, slope_params, 9.81)

    # plot results
    # that's rather crude
    fig, axs = plt.subplots(1,3)
    t_plt = []
    x_plt = []
    y_plt = []
    def animate(i):
        t_plt.append(t[i])
        x_plt.append(x[i])
        y_plt.append(y[i])
        axs[0].clear()
        axs[0].plot(t_plt, x_plt)
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('x')
        axs[1].clear()
        axs[1].plot(t_plt, y_plt)
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('y')
        axs[2].clear()
        axs[2].plot(x,y)
        axs[2].plot(x[i],y[i],'o')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')

    ani = FuncAnimation(plt.gcf(), animate, frames=len(x), interval=100, repeat=False)
    plt.show()