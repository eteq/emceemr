from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from dataclasses import dataclass
from typing import Any
from types import MappingProxyType
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from scipy import stats

from astropy import modeling

MINF = -np.inf


__all__ = ['ProbabilisticModel', 'ProbabilisticParameter']


@dataclass
class ProbabilisticParameter:
    name: str
    description: str = ''
    prior: stats._distn_infrastructure.rv_generic = None
    posterior_samples: Any = None


class ParameterMapping(Mapping):
    def __init__(self, parent, param_names):
        self._parent = parent
        self._param_names = tuple(param_names)

    def __len__(self):
        return len(self._param_names)

    def __iter__(self):
        return iter(self._param_names)

    def __getitem__(self, nm):
        if nm not in self._param_names:
            raise KeyError(nm)
        return getattr(self._parent, nm)

    def __repr__(self):
        return str(dict(self))


class ProbabilisticModel:
    """
    A container for a mean model (astropy model) plus a noise model.

    Parameters
    ----------
    mean_model : astropy.modeling model
        An astropy model representing the mean model
    data_distribution : astropy.modeling model
        A model that takes ('mean_model_result', 'data') and yields the logpdf.
        See `noise_models.generate_logpdfmodel_class`.
    """

    def __init__(self, mean_model, data_distribution):
        self.mean_model = mean_model
        self.data_distribution = data_distribution
        self.cross_prior = None

        parameters = []
        self._mean_parameters = []
        self._data_parameters = []
        for param_name in mean_model.param_names:
            p = ProbabilisticParameter(param_name, description=f'mean model parameter "{param_name}"')
            self._mean_parameters.append(p.name)
            parameters.append(p)

        for param_name in data_distribution.param_names:
            p = ProbabilisticParameter(param_name, description=f'data distribution parameter "{param_name}"')
            self._data_parameters.append(p.name)
            parameters.append(p)

        param_names = []
        for p in parameters:
            setattr(self, p.name, p)
            param_names.append(p.name)
        self.parameters = ParameterMapping(self, param_names)

    def __repr__(self):
        cname = self.__class__.__name__
        params_str = ', '.join(self.parameters.keys())
        return f'<{cname}({params_str})>'

    def ln_priors(self, parameter_values, datax, datay):
        lpri = 0
        for p, val in zip(self.parameters.values(), parameter_values):
            if p.prior is not None:
                lpri += p.prior.logpdf(val)
                if lpri == MINF:
                    return lpri
        if self.cross_prior is not None:
            lpri += self.cross_prior(*parameter_values)
        return lpri

    def ln_likelihood(self, parameter_values, datax, datay):
        model_val = self.mean_model.evaluate(datax, *parameter_values[:len(self._mean_parameters)])
        return self.data_distribution.evaluate(model_val, datay, *parameter_values[-len(self._data_parameters):])

    def ln_prob(self, parameter_values, datax, datay):
        lpri = self.ln_priors(parameter_values, datax, datay)
        if lpri == MINF:
            return lpri
        else:
            llike = self.ln_likelihood(parameter_values, datax, datay)
            return lpri + np.sum(llike, axis=-1)
