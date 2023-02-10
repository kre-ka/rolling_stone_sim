from sim import LinearSlope
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # create a simulator instance
    # y = Ax + B
    sim = LinearSlope()

    # set initial conditions and parameters
    # x(0), x'(0)
    initial_conditions = (0, 0)
    # A, B, g
    params = (-0.5, 100, 9.81)
    
    # simulate
    t, x, y = sim.simulate(initial_conditions, params)

    # plot results
    fig, axs = plt.subplots(1,3)
    axs[0].plot(t,x)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')
    axs[1].plot(t,y)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('y')
    axs[2].plot(x,y)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    plt.show()    