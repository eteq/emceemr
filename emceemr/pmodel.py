from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc

import numpy as np
from astropy import modeling

__all__ = ['ProbabilisticModel']


class ProbabilisticModel(modeling.Model):
    @abc.abstractmethod
    def lnprob_model(self, meanmodel_value, data):
        raise NotImplementedError


class GaussianProbabilisticModel(ProbabilisticModel):
    data_sig = modeling.Parameter('data_sig', 'Standard deviation of the data around the mean model')

    _GAUSS_TERM = -0.5*np.log(2*np.pi)
    def lnprob_model(self, meanmodel_value, data):
        return -0.5 * (((data - mu) / sig) ** 2 + 2 * np.log(sig)) + self.GAUSS_TERM
