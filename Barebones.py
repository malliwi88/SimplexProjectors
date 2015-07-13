__author__ = 'Stuart Gordon Reid'

"""
This file contains an implementation of the Barebones Particle Swarm Optimization algorithm for the Portfolio
Optimization problem. This file contains two constraint satisfaction techniques namely, repair by normalization and
lagrange multipliers method. These results are compared to the implementation of the Preserving Feasibility Barebones
Particle Swarm Optimization algorithm which uses a preserving feasibility constraint satisfaction technique.
"""

import numpy
import random
import pandas
import numpy.random as nrand
from Portfolio import Portfolio
from AssetSimulator import AssetSimulator


class BarebonesOptimizer:
    def __init__(self, swarm_size, asset_returns, corr):
        """
        Initialization method for a BarebonesOptimizer object
        :param swarm_size: the number of particles in the swarm
        :param asset_returns: a list of asset return sequences
        :param corr: correlation matrix of asset returns
        """
        self.corr, self.returns, self.swarm_size = corr, asset_returns, swarm_size
        # Get the number of assets in the portfolio
        self.n = len(asset_returns)

        # Matrices of current positions and personal bests for each particle
        self.pbest_portfolios = []
        self.xbest_portfolios = []
        for i in range(swarm_size):
            weights = nrand.dirichlet(numpy.ones(self.n), 1)[0]
            self.xbest_portfolios.append(Portfolio(self.returns, self.corr, weights))
            self.pbest_portfolios.append(Portfolio(self.returns, self.corr, weights))

    def best(self, objective, ce, cb, le, lb):
        """
        This step in the algorithm finds the best portfolio from the swarm
        :param objective: objective function to use
        :param ce: coefficient for equality constraint
        :param cb: coefficient for boundary constraint
        :param le: lagrangian multiplier for equality constraint
        :param lb: lagrangian multiplier for boundary constraint
        :return: index of global best, global best fitness
        """
        gbest_index, gbest_fitness = 0, float('+inf')

        for i in range(1, self.swarm_size):
            pbest_portfolio = self.pbest_portfolios[i]
            xbest_portfolio = self.xbest_portfolios[i]

            if objective == "repair":
                xbest_portfolio.repair()

            pbest_fitness = pbest_portfolio.get_fitness(objective, ce, cb, le, lb)
            xbest_fitness = xbest_portfolio.get_fitness(objective, ce, cb, le, lb)

            if xbest_fitness <= pbest_fitness:
                xweights = xbest_portfolio.get_weights()
                self.pbest_portfolios[i].update_weights(xweights)

            if xbest_fitness < gbest_fitness:
                gbest_index = i
                gbest_fitness = xbest_fitness

        return gbest_index, gbest_fitness

    def update(self, iterations, objective, ce, cb, le, lb, growth_rate=1.1):
        """
        This update equation can be used for every method except the feasibility preserving method
        :param iterations: number of iterations
        :param objective: objective function
        :param ce: coefficient for equality constraint
        :param cb: coefficient for boundary constraint
        :param le: lagrangian multiplier for equality constraint
        :param lb: lagrangian multiplier for boundary constraint
        :param growth_rate: growth rate of the coefficients for the constraints
        :return: history of gbest fitness, equality, and boundary violations
        """
        history = numpy.zeros(iterations)
        violation_e = numpy.zeros(iterations)
        violation_b = numpy.zeros(iterations)

        for i in range(iterations):
            best_index, best_fitness = self.best(objective, ce, cb, le, lb)
            best_weights = self.pbest_portfolios[best_index].get_weights()
            best_portfolio = self.pbest_portfolios[best_index]

            for j in range(self.swarm_size):
                if j != best_index:
                    if objective != "preserving":
                        pbest_weights = self.pbest_portfolios[j].get_weights()
                        loc = numpy.array((best_weights + pbest_weights) / 2.0)
                        sig = numpy.array(numpy.abs(best_weights - pbest_weights))
                        velocity = numpy.empty(len(best_weights))
                        for k in range(len(best_weights)):
                            velocity[k] = random.normalvariate(loc[k], sig[k])
                        self.xbest_portfolios[j].update_weights(velocity)

                    elif objective == "preserving":
                        pbest_weights = self.pbest_portfolios[j].get_weights()
                        loc = numpy.array((best_weights + pbest_weights)/2)
                        velocity = nrand.dirichlet(loc, 1)[0]
                        self.xbest_portfolios[j].update_weights(velocity)

            history[i] = best_portfolio.min_objective()
            violation_e[i] = best_portfolio.get_boundary_penalty()
            violation_b[i] = best_portfolio.get_equality_penalty()
            if objective == "penalty" or objective == "lagrange":
                ce *= growth_rate
                cb *= growth_rate
                if objective == "lagrange":
                    le -= ce * violation_e[i]
                    lb -= cb * violation_b[i]
        return history, violation_e, violation_b

    def optimize_none(self, iterations, ce, cb, le, lb):
        return self.update(iterations, "none", ce, cb, le, lb)

    def optimize_repair(self, iterations, ce, cb, le, lb):
        return self.update(iterations, "repair", ce, cb, le, lb)

    def optimize_penalty(self, iterations, ce, cb, le, lb):
        return self.update(iterations, "penalty", ce, cb, le, lb)

    def optimize_lagrange(self, iterations, ce, cb, le, lb):
        return self.update(iterations, "lagrange", ce, cb, le, lb)

    def optimize_preserving(self, iterations, ce, cb, le, lb):
        return self.update(iterations, "preserving", ce, cb, le, lb)


if __name__ == '__main__':
    n, delta, sigma, mu, time = 16, float(1/252), 0.125, 0.08, 500
    asset_simulator = AssetSimulator(delta, sigma, mu, time)
    asset_returns = asset_simulator.assets_returns(n)
    corr = pandas.DataFrame(asset_returns).transpose().corr()
    bpso = BarebonesOptimizer(25, asset_returns, corr)
