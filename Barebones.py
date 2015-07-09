__author__ = 'Stuart Gordon Reid'

"""
This file contains an implementation of the Barebones Particle Swarm Optimization algorithm for the Portfolio
Optimization problem. This file contains two constraint satisfaction techniques namely, repair by normalization and
lagrange multipliers method. These results are compared to the implementation of the Preserving Feasibility Barebones
Particle Swarm Optimization algorithm which uses a preserving feasibility constraint satisfaction technique.
"""

import numpy
import random
import numpy.random as nrand
from Portfolio import Portfolio


class BarebonesOptimizer:
    def __init__(self, swarm_size, asset_returns, corr):
        self.corr = corr
        self.n = len(asset_returns)
        self.returns = asset_returns
        self.swarm_size = swarm_size
        self.swarm = []
        for i in range(swarm_size):
            ones = numpy.ones(self.n)
            self.swarm.append(nrand.dirichlet(ones, 1)[0])

    def get_best(self, objective, ce, cb, le, lb):
        best_index, best_weights, best_fitness = 0, None, float('+inf')
        for j in range(self.swarm_size):
            portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
            fitness = portfolio.get_fitness(objective, ce, cb, le, lb)
            if objective == "repair":
                self.swarm[j] = portfolio.weights
            if fitness < best_fitness:
                best_weights, best_fitness, best_index = self.swarm[j], fitness, j
        return best_index, best_weights, best_fitness

    def optimize_none(self, iterations, ce, cb, le, lb):
        return self.general_update(iterations, "none", ce, cb, le, lb)

    def optimize_repair(self, iterations, ce, cb, le, lb):
        return self.general_update(iterations, "repair", ce, cb, le, lb)

    def optimize_penalty(self, iterations, ce, cb, le, lb):
        return self.general_update(iterations, "penalty", ce, cb, le, lb)

    def optimize_lagrange(self, iterations, ce, cb, le, lb):
        return self.general_update(iterations, "lagrange", ce, cb, le, lb)

    def optimize_preserving(self, iterations, ce, cb, le, lb):
        return self.preserving_update(iterations, "preserving", ce, cb, le, lb)

    def general_update(self, iterations, objective, ce, cb, le, lb, q=1.1):
        history = numpy.zeros(iterations)
        violation_e = numpy.zeros(iterations)
        violation_b = numpy.zeros(iterations)
        for i in range(iterations):
            best_index, best_weights, best_fitness = self.get_best(objective, ce, cb, le, lb)
            for j in range(self.swarm_size):
                if j != best_index:
                    loc = numpy.array((best_weights + self.swarm[j]) / 2.0)
                    sig = numpy.array(numpy.abs(best_weights - self.swarm[j]))
                    velocity = numpy.empty(len(best_weights))
                    for k in range(len(best_weights)):
                        velocity[k] = random.normalvariate(loc[k], sig[k])
                    self.swarm[j] = velocity
            best_portfolio = Portfolio(self.returns, self.corr, best_weights)
            history[i] = best_portfolio.min_objective()
            violation_e[i] = best_portfolio.get_boundary_penalty()
            violation_b[i] = best_portfolio.get_equality_penalty()
            if objective == "penalty" or objective == "lagrange":
                ce *= q
                cb *= q
        return history, violation_e, violation_b

    def preserving_update(self, iterations, objective, ce, cb, le, lb):
        history = numpy.zeros(iterations)
        violation_e = numpy.zeros(iterations)
        violation_b = numpy.zeros(iterations)
        for i in range(iterations):
            best_index, best_weights, best_fitness = self.get_best(objective, ce, cb, le, lb)
            for j in range(self.swarm_size):
                if j != best_index:
                    r = nrand.dirichlet(numpy.ones(self.n), 1)[0] - float(1/self.n)
                    velocity = best_weights - self.swarm[j] + r
                    self.swarm[j] += velocity
                    portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
                    portfolio.repair()
                    self.swarm[j] = portfolio.weights
            best_portfolio = Portfolio(self.returns, self.corr, best_weights)
            history[i] = best_portfolio.min_objective()
            violation_e[i] = best_portfolio.get_boundary_penalty()
            violation_b[i] = best_portfolio.get_equality_penalty()
        return history, violation_e, violation_b

