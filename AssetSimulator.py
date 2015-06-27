__author__ = 'Stuart Gordon Reid'


import math
import numpy
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
        returns = []
        for i in range(n):
            returns.append(self.geometric_brownian_motion())
        return returns

    def asset_prices(self, n, returns=None):
        if returns is None:
            returns = self.assets_returns(n)
        prices = []
        for asset in returns:
            prices.append(self.returns_to_prices(asset))
        return prices

    def returns_to_prices(self, returns):
        prices = numpy.empty(len(returns) + 1)
        prices[0] = 100
        for i in range(len(returns)):
            prices[i + 1] = prices[i] * (1 + returns[i])
        return prices