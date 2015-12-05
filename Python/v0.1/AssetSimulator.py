__author__ = 'Stuart Gordon Reid'


import math
import numpy
import pandas
import numpy.random as nrand


class AssetSimulator:
    def __init__(self, delta, sigma, mu, time):
        self.delta = delta
        self.sigma = sigma
        self.time = time
        self.mu = mu

    def wiener_process(self):
        sqrt_delta_sigma = math.sqrt(self.delta) * self.sigma
        return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=self.time)

    def geometric_brownian_motion(self):
        wiener_process = self.wiener_process()
        sigma_pow_mu_delta = (self.mu - 0.5 * math.pow(self.sigma, 2.0)) * self.delta
        gbm_process = wiener_process + sigma_pow_mu_delta
        return numpy.exp(gbm_process) - numpy.ones(len(gbm_process))

    def assets_returns(self, n):
        returns = numpy.ones(shape=(n, self.time))
        for i in range(n):
            returns[i] = self.geometric_brownian_motion()
        return pandas.DataFrame(returns)