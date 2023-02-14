from sim import LinearSlope, QuadraticSlope, plot_sim_results


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

    plot_sim_results(t, x, y)