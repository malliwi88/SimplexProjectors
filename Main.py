__author__ = 'Stuart Gordon Reid'


from AssetSimulator import AssetSimulator
from Barebones import BarebonesOptimizer
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import Portfolio
import cProfile
import pandas
import numpy
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


def three_dimensional_landscape(returns, corr_m, size, c_e=1.0, c_b=1.0, m_e=1.0, m_b=1.0):
    """
    This method plots the fitness landscape for each three methods in three dimensions (surface plot)
    """
    step = float(1/size)
    x_axis = numpy.arange(-0.5, 1.5, 2*step)
    y_axis = numpy.arange(-0.5, 1.5, 2*step)
    z_axis_nochange = numpy.zeros(shape=(size, size))
    z_axis_repaired = numpy.zeros(shape=(size, size))
    z_axis_penaltym = numpy.zeros(shape=(size, size))
    z_axis_lagrange = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = x_axis[i], y_axis[j]
            p = Portfolio.Portfolio(returns, corr_m, numpy.array([x, y]))
            z_axis_nochange[i][j] = p.min_objective()
            z_axis_penaltym[i][j] = p.penalty_objective(c_e, c_b)
            z_axis_lagrange[i][j] = p.lagrange_objective(c_e, c_b, m_e, m_b)
            z_axis_repaired[i][j] = p.repair_objective()
    x_axis, y_axis = numpy.meshgrid(x_axis, y_axis)
    plot_surface(x_axis, y_axis, z_axis_nochange, "N")
    plot_surface(x_axis, y_axis, z_axis_repaired, "R")
    plot_surface(x_axis, y_axis, z_axis_penaltym, "P")
    plot_surface(x_axis, y_axis, z_axis_lagrange, "L")


def plot_surface(X, Y, Z, label):
    """
    This method actually plots and shows the three dimensional surface
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    '''
    for ii in range(0, 360, 90):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("Views/View II " + label + " " + str(ii) + ".png")
    '''
    plt.show()


def runner(n, sigma, delta, mu, time, iterations=30):
    asset_simulator = AssetSimulator(delta, sigma, mu, time)

    Portfolio.memoizer = {}
    none, lagrange, repair, preserve, ss = [], [], [], [], 25
    none_v, lagrange_v, repair_v, preserve_v = [], [], [], []

    for i in range(iterations):
        print("Iteration", i, "Simulating Returns ...")
        asset_returns = asset_simulator.assets_returns(n)
        corr = pandas.DataFrame(asset_returns).transpose().corr()
        # three_dimensional_landscape(asset_returns, corr, 100)

        print("Iteration", i, "Starting Optimizer ...")
        none_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation = none_opt.optimize_none(201)
        none_v.append(violation)
        none.append(result)

        print("Iteration", i, "Starting Lagrange Optimizer ...")
        lagrange_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation = lagrange_opt.optimize_lagrange(201)
        lagrange_v.append(violation)
        lagrange.append(result)

        print("Iteration", i, "Starting Repair Method Optimizer ...")
        repair_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation = repair_opt.optimize_repair(201)
        repair_v.append(violation)
        repair.append(result)

        print("Iteration", i, "Starting Preserving Feasibility Optimizer ...")
        preserve_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation = preserve_opt.optimize_preserving(201)
        preserve_v.append(violation)
        preserve.append(result)

    n_r = pandas.DataFrame(none)
    n_r.to_csv("Results/Results_None_" + str(n) + ".csv")

    n_v = pandas.DataFrame(none_v)
    n_v.to_csv("Results/Violations_None_" + str(n) + ".csv")

    l_r = pandas.DataFrame(lagrange)
    l_r.to_csv("Results/Results_Lagrange_" + str(n) + ".csv")

    l_v = pandas.DataFrame(lagrange_v)
    l_v.to_csv("Results/Violations_Lagrange_" + str(n) + ".csv")

    r_r = pandas.DataFrame(repair)
    r_r.to_csv("Results/Results_Repair_" + str(n) + ".csv")

    r_v = pandas.DataFrame(repair_v)
    r_v.to_csv("Results/Violations_Repair_" + str(n) + ".csv")

    p_r = pandas.DataFrame(preserve)
    p_r.to_csv("Results/Results_Preserve_" + str(n) + ".csv")

    p_v = pandas.DataFrame(preserve_v)
    p_v.to_csv("Results/Violations_Preserve_" + str(n) + ".csv")

    plot_results([n_r.mean(), l_r.mean(), r_r.mean(), p_r.mean()], ["none", "lagrange", "repair", "preserving"])
    plot_results([n_r.std(), l_r.std(), r_r.std(), p_r.std()], ["none", "lagrange", "repair", "preserving"])
    plot_results([n_v.std(), l_v.std(), p_v.std()], ["none", "lagrange", "preserving"])


def surface_plotter(n, sigma, delta, mu, time, c_e, c_b, m_e, m_b):
    asset_simulator = AssetSimulator(delta, sigma, mu, time)
    asset_returns = asset_simulator.assets_returns(n)
    corr = pandas.DataFrame(asset_returns).transpose().corr()
    three_dimensional_landscape(asset_returns, corr, 200, c_e, c_b, m_e, m_b)


if __name__ == '__main__':
    # runner(8, 0.125, float(1/252), 0.08, 250, iterations=5)
    surface_plotter(2, 0.125, float(1/252), 0.08, 250, 2.0, 2.0, 0.5, 0.5)
