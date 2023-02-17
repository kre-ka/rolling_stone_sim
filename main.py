from sim import LinearSlope, QuadraticSlope, plot_sim_results


if __name__ == '__main__':
    # # create a simulator instance
    # sim = LinearSlope()
    # # set slope parameters
    # # for descriptions see slope object and simulate method docstrings
    # slope_params = (5, 20)
    # # simulate movement using given parameters
    # t, x, y = sim.simulate(slope_params, 9.81, 30, 10)

    sim = QuadraticSlope()
    slope_params = (5, 10, 0.5)
    t, x, y, tf = sim.simulate(slope_params, 9.81, 30, 100)
    print(f"slope params: {slope_params}")
    print(f"finish time: {tf}")

    plot_sim_results(t, x, y)