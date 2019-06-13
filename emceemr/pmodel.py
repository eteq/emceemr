from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from dataclasses import dataclass
from typing import Any
from types import MappingProxyType
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from scipy import stats, special

from astropy import modeling, uncertainty
from astropy import units as u

import emcee

MINF = -np.inf


__all__ = ['ProbabilisticModel', 'ProbabilisticParameter']


@dataclass
class ProbabilisticParameter:
    name: str
    description: str = ''
    prior: stats._distn_infrastructure.rv_generic = None
    posterior_samples: Any = None
    unit: u.Unit = None


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
        self.cross_priors = None # a term in the prior that's not a single-variable prior.
        self.sampler = None
        self.nburnin = 0

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
                if callable(p.prior):
                    lpri += p.prior(val)
                else:
                    lpri += p.prior.logpdf(val)
                if lpri == MINF:
                    return lpri
        if self.cross_priors is not None:
            lpri += self.cross_priors(*parameter_values)
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

    def initialize_ball(self, size=None, stds=1e-6):
        """
        uses the values in the mean model as the centers
        """
        if size is None:
            size = self.sampler.k
        vals = []
        for nm in self.parameters:
            if nm in self.mean_model.param_names:
                vals.append(getattr(self.mean_model, nm).value)
            elif nm in self.data_distribution.param_names:
                vals.append(getattr(self.data_distribution, nm).value)
            else:
                assert False, "This should be impossible!  Something very wrong with the class initialization"
        if np.array(stds).shape == ():
            stds = [stds]*len(self.parameters)
        return emcee.utils.sample_ball(vals, stds, size=size)

    def initialize_priors(self, size=None):
        if size is None:
            size = self.sampler.k

        p0s = []
        for p in self.parameters.values():
            if p.prior is None:
                raise ValueError('Cannot use priors to initialize a model without priors!')
            p0s.append(p.prior.rvs(size))
        return np.array(p0s).T

    def _make_ensemble_sampler(self, nwalkers=None, **kwargs):
        if nwalkers is None:
            nwalkers = len(self.parameters)*4
        self.sampler = emcee.EnsembleSampler(nwalkers, len(self.parameters), self.ln_prob, **kwargs)
        return self.sampler

    def ensemble_sample(self, datax, datay, nsamples, initialization, nwalkers=None, **kwargs):
        self._make_ensemble_sampler(nwalkers, args=[datax, datay])

        if initialization == 'priors':
            p0 = self.initialize_priors(self.sampler.k)
        elif initialization == 'ball':
            p0 = self.initialize_ball(self.sampler.k)
        else:
            p0 = initialization

        self.sampler.run_mcmc(p0, nsamples, **kwargs)
        self._update_posteriors()

        return self.sampler

    @property
    def nburnin(self):
        return self._nburnin
    @nburnin.setter
    def nburnin(self, val):
        self._nburnin = val
        self._update_posteriors()

    @property
    def burnin_chains(self):
        return self.sampler.chain[:, :self.nburnin, :]

    @property
    def sample_chains(self):
        return self.sampler.chain[:, self.nburnin:, :]

    def _update_posteriors(self):
        if self.sampler is None:
            return

        for i, p in enumerate(self.parameters.values()):
            if p.unit is None:
                p.posterior_samples = self.sample_chains[..., i]
                p.burnin_samples = self.burnin_chains[..., i]
            else:
                p.posterior_samples = p.posterior_samples * p.unit
                p.burnin_samples = p.burnin_samples * p.unit

    def plot_chains(self, incl_burnin=True, plot_width=5, plot_height=4):
        from matplotlib import pyplot as plt

        if self.nburnin and incl_burnin:
            fig, axs = plt.subplots(len(self.parameters), 2, figsize=(plot_width*2, plot_height*len(self.parameters)))
            axs[0, 0].set_title('burnin')
            axs[0, 1].set_title('samples')
        else:
            fig, axs = plt.subplots(len(self.parameters), 1, figsize=(plot_width, plot_height*len(self.parameters)))

        for axi, p in zip(axs, self.parameters.values()):
            if hasattr(axi, 'shape'):
                axi[0].plot(p.burnin_samples.T)
                axi[0].set_xlim(0, self.nburnin)
                axi[0].set_ylabel(p.name)

                axi[1].plot(p.posterior_samples.T)
                axi[1].set_xlim(0, p.posterior_samples.shape[1])
            else:
                axi.plot(p.posterior_samples.T)
                axi.set_ylabel(p.name)
                axi.set_xlim(0, p.posterior_samples.shape[1])
        return fig

    _default_corner_quantiles = [special.erfc(2**-0.5), .5, special.erf(2**-0.5)]
    def plot_corner(self, **kwargs):
        from corner import corner

        kwargs.setdefault('labels', list(self.parameters.keys()))
        kwargs.setdefault('quantiles', self._default_corner_quantiles)
        kwargs.setdefault('show_titles', True)

        schains = self.sample_chains
        flatchain = schains.reshape(schains.size//len(self.parameters), len(self.parameters))

        return corner(flatchain, **kwargs)