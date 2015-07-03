__author__ = 'Stuart Gordon Reid'


from AssetSimulator import AssetSimulator
from Barebones import BarebonesOptimizer
import matplotlib.pyplot as plt
from matplotlib import cm
import Portfolio
import cProfile
import pandas
import numpy


def plot_paths(r, sim):
    """
    This method plots a number of asset paths generated using Geometric Brownian Motion
    """
    asset_prices = sim.asset_prices(n, returns=r)
    for p in asset_prices:
        plt.plot(p)
    plt.show()


def plot_results(results, labels=None):
    plt.ylabel("- Sharpe Ratio")
    plt.xlabel("Iterations")
    for i in range(len(results)):
        if labels is None:
            plt.plot(results[i])
        else:
            plt.plot(results[i], label=labels[i])
    plt.legend(loc="best")
    plt.show()


def two_dimensional_landscape(returns, corr_m, size):
    """
    This method plots the fitness landscape for each three methods in two dimensions (grid)
    """
    m_lagrange = numpy.zeros(shape=(size, size))
    m_repair = numpy.zeros(shape=(size, size))
    m_none = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = float(i/size), float(j/size)
            p = Portfolio.Portfolio(returns, corr_m, numpy.array([x, y]))
            m_lagrange[i][j] = p.lagrange_objective()
            m_none[i][j] = p.min_objective()
            m_repair[i][j] = p.repair_objective()
    plt.matshow(numpy.fliplr(m_none))
    plt.matshow(numpy.fliplr(m_repair))
    plt.matshow(numpy.fliplr(m_lagrange))
    plt.show()


def three_dimensional_landscape(returns, corr_m, size):
    """
    This method plots the fitness landscape for each three methods in three dimensions (surface plot)
    """
    step = float(1/size)
    x_axis = numpy.arange(step, 1 + step, step)
    y_axis = numpy.arange(step, 1 + step, step)
    z_axis_nochange = numpy.zeros(shape=(size, size))
    z_axis_repaired = numpy.zeros(shape=(size, size))
    z_axis_lagrange = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = x_axis[i], y_axis[j]
            p = Portfolio.Portfolio(returns, corr_m, numpy.array([x, y]))
            z_axis_lagrange[i][j] = p.lagrange_objective()
            z_axis_nochange[i][j] = p.min_objective()
            z_axis_repaired[i][j] = p.repair_objective()
    x_axis, y_axis = numpy.meshgrid(x_axis, y_axis)
    plot_surface(x_axis, y_axis, z_axis_nochange)
    plot_surface(x_axis, y_axis, z_axis_nochange)
    plot_surface(x_axis, y_axis, z_axis_lagrange)


def plot_surface(X, Y, Z):
    """
    This method actually plots and shows the three dimensional surface
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    fig.colorbar(surf)
    plt.show()


def runner(n, sigma, delta, mu, time, iterations=30):
    asset_simulator = AssetSimulator(delta, sigma, mu, time)

    Portfolio.memoizer = {}
    lagrange, repair, preserve, ss = [], [], [], 25

    for i in range(iterations):
        print("Iteration", i, "Simulating Returns ...")
        asset_returns = asset_simulator.assets_returns(n)
        corr = pandas.DataFrame(asset_returns).transpose().corr()
        # three_dimensional_landscape(asset_returns, corr, 100)

        print("Iteration", i, "Starting Lagrange Optimizer ...")
        lagrange_opt = BarebonesOptimizer(ss, asset_returns, corr)
        lagrange.append(lagrange_opt.optimize_lagrange(501))

        print("Iteration", i, "Starting Repair Method Optimizer ...")
        repair_opt = BarebonesOptimizer(ss, asset_returns, corr)
        repair.append(repair_opt.optimize_preserving(501))

        print("Iteration", i, "Starting Preserving Feasibility Optimizer ...")
        preserve_opt = BarebonesOptimizer(ss, asset_returns, corr)
        preserve.append(preserve_opt.optimize_repair(501))

    lagrange_data = pandas.DataFrame(lagrange).transpose()
    lagrange_data.to_csv("Results/Lagrange_" + str(n) + ".csv")

    repair_data = pandas.DataFrame(repair).transpose()
    repair_data.to_csv("Results/Repair_" + str(n) + ".csv")

    preserve_data = pandas.DataFrame(preserve).transpose()
    preserve_data.to_csv("Results/Preserve_" + str(n) + ".csv")

    return lagrange_data, repair_data, preserve_data

if __name__ == '__main__':
    l, r, p = runner(8, 0.125, float(1/252), 0.08, 500, iterations=1)
    plot_results([l[0], r[0], p[0]], ["lagrange", "repair", "preserving"])
