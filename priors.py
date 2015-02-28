from __future__ import division, print_function

import math

from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy import nddata

import emcee

#used alot, so we slightly speed it up by pre-calcing
INF = np.inf
MINF = -INF
TWOPI = 2 * np.pi


class Prior(object):
    """
    An abstract class that is here mostly just to document how the priors will be called
    """
    @abstractmethod
    def __call__(self, value, allparamdct={}):
        raise NotImplementedError

    @abstractmethod
    def sample(self, n):
        """
        Return `n` samples from this prior distribution

        Might raise a ValueError for cases like tied parameters and such
        """
        raise NotImplementedError


class UniformPrior(Prior):
    """
    A uniform prior.  One or both of the sides might be tied to another
    parameter, in which case it should be passed in as a string
    """
    def __init__(self, lower, upper):
        if isinstance(lower, basestring):
            self.lower = None
            self.lower_var = lower
        else:
            self.lower = lower
            self.lower_var = None

        if isinstance(upper, basestring):
            self.upper = None
            self.upper_var = upper
        else:
            self.upper = upper
            self.upper_var = None


        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError('lower is bigger than upper!')

    def __call__(self, value, allparamdct={}):
        if self.lower_var is not None:
            self.lower = allparamdct[self.lower_var]
        if self.upper_var is not None:
            self.upper = allparamdct[self.upper_var]

        if self.lower is None or self.upper is None:
            if self.upper is not None and value > self.upper:
                return MINF
            elif self.lower is not None and value < self.lower:
                return MINF
            else:
                return 0.0

        elif self.lower < value < self.upper:
            return -np.log(self.upper - self.lower)

        else:
            return MINF

    def sample(self, n):
        if self.upper_var is not None or self.lower_var is not None:
            raise ValueError('cannot sample if the distribution is tied to other params')
        if self.upper is None or self.lower is None:
            raise ValueError('cannot sample from unbounded uniforms')
        return np.random.rand(n) * (self.upper - self.lower) + self.lower

    def __repr__(self):
        r = super(UniformPrior, self).__repr__()
        lower = self.lower if self.lower_var is None else self.lower_var
        upper = self.upper if self.upper_var is None else self.upper_var
        vrs = ': lower={0}, upper={1}'
        return r.split(' at ')[0]  + vrs.format(lower, upper) + '>'


class NormalPrior(Prior):
    """
    A standard 1D gaussian prior, possibly clipped
    """
    def __init__(self, mu, var=None, sig=None, lower=None, upper=None):
        self.mu = mu

        if var is None:
            if sig is None:
                raise ValueError('must give sig or var to NormalPrior')
            self.sig = sig
        else:
            if sig is not None:
                raise ValueError('cannot give both sig and var to NormalPrior')
            self.var = var

        self.unif = UniformPrior(lower, upper)

    @property
    def sig(self):
        return self.var**0.5
    @sig.setter
    def sig(self, value):
        self.var = value**2

    def __call__(self, value, allparamdct={}):
        uniflp = self.unif(value, allparamdct)
        return uniflp - 0.5*(TWOPI + self.var + (value - self.mu)**2/self.var)

    def __repr__(self):
        r = super(NormalPrior, self).__repr__()
        vrs = ': mu={0}, sig={1}, '.format(self.mu, self.sig) + repr(self.unif).split(':')[-1][1:-1]
        return r.split(' at ')[0] + vrs.format(self) + '>'

    def sample(self, n):
        res = np.ones(n) * np.nan

        # resample until it's within the bounds
        while np.any(np.isnan(res)):
            res[np.isnan(res)] = np.random.randn(np.sum(np.isnan(res))) * self.var**0.5 + self.mu

            msk = np.ones(len(res), dtype=bool)
            if self.unif.lower is not None:
                msk = msk & (res > self.unif.lower)
            if self.unif.upper is not None:
                msk = msk & (res < self.unif.upper)
            res[~msk] = np.nan

        return res


class ScaleFreePrior(Prior):
    """
    Jeffry's prior for a scale parameter, goes like 1/val.  An improper prior
    unless lower and upper are set.
    """
    def __init__(self, lower=None, upper=None):
        if lower is None or lower <= 0:
            lower = 0
        self.lower = lower
        self.upper = upper

    def __call__(self, value, allparamdct={}):
        if value <= self.lower or value >= self.upper:
            return MINF
        else:
            return -np.log(value)

    def sample(self, n):
        if self.lower is not None and self.upper is not None:
            N = np.log(self.upper) - np.log(self.lower)
            return self.lower * np.exp(N*np.random.rand(n))
        else:
            raise ValueError("Scale-free prior is improper, can't sample")

class EmpiricalPrior(Prior):
    """
    Prior defined by the histogram of an array - uses gaussian kde to estimate
    the density
    """
    def __init__(self, samples, lower=None, upper=None, kdebw=None):
        from scipy.stats.kde import gaussian_kde
        self.samples = np.array(samples, copy=False).flatten()
        self.kde = gaussian_kde(self.samples, kdebw)
        self.unif = UniformPrior(lower, upper)

    def __call__(self, value, allparamdct={}):
        uniflp = self.unif(value, allparamdct)
        return uniflp + np.log(self.kde(value))

    def sample(self, n):
        return self.kde.resample(n)


class DeltaPrior(Prior):
    """
    A delta function prior - only one value is allowed
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, value, allparamdct={}):
        if value != self.value:
            return MINF
        else:
            return 0.0

    def sample(self, n):
        return np.ones(n) * self.value
