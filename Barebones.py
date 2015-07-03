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

    def get_best(self, objective):
        best_index, best_weights, best_fitness = 0, None, float('+inf')
        for j in range(self.swarm_size):
            portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
            fitness = portfolio.get_fitness(objective)
            if objective == "repair":
                self.swarm[j] = portfolio.weights
            if fitness < best_fitness:
                best_weights, best_fitness, best_index = self.swarm[j], fitness, j
        return best_index, best_weights, best_fitness

    def optimize_none(self, iterations):
        return self.general_update(iterations, "none")

    def optimize_repair(self, iterations):
        return self.general_update(iterations, "repair")

    def optimize_lagrange(self, iterations):
        return self.general_update(iterations, "lagrange")

    def optimize_preserving(self, iterations):
        return self.preserving_update(iterations)

    def general_update(self, iterations, objective):
        history = numpy.zeros(iterations)
        violation = numpy.zeros(iterations)
        for i in range(iterations):
            best_index, best_weights, best_fitness = self.get_best(objective)
            for j in range(self.swarm_size):
                if j != best_index:
                    loc = numpy.array((best_weights + self.swarm[j]) / 2.0)
                    sig = numpy.array(numpy.abs(best_weights - self.swarm[j]))
                    velocity = numpy.empty(len(best_weights))
                    for k in range(len(best_weights)):
                        velocity[k] = random.normalvariate(loc[k], sig[k])
                    self.swarm[j] = velocity
            history[i] = best_fitness
            violation[i] = self.constraint_violation()
        return history, violation

    def preserving_update(self, iterations, objective="preserving"):
        history = numpy.zeros(iterations)
        violation = numpy.zeros(iterations)
        for i in range(iterations):
            best_index, best_weights, best_fitness = self.get_best(objective)
            for j in range(self.swarm_size):
                if j != best_index:
                    r = nrand.dirichlet(numpy.ones(self.n), 1)[0] - float(1 / self.n)
                    velocity = (numpy.array(best_weights - self.swarm[j])/2) + r
                    self.swarm[j] += velocity
            history[i] = best_fitness
            violation[i] = self.constraint_violation()
        return history, violation

    def constraint_violation(self):
        violation = 0.0
        for i in range(self.swarm_size):
            violation += 1.0 - sum(self.swarm[i])
        return violation / self.swarm_size
