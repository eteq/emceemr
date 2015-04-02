from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import abc

import re

import numpy as np

import emcee
import triangle

__all__ = ['Model']

MINF = -np.inf


class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, priors):
        if not getattr(self, 'param_names', None):
            raise ValueError('must have param_names')

        try:
            priordct = dict(priors)
        except ValueError:
            # assume it's a list
            priordct = None
        if priordct is None:
            if len(priors) != len(self.param_names):
                raise ValueError('need an equal number of priors and parameters')
        else:
            priors = []
            for nm in self.param_names:
                try:
                    priors.append(priordct.pop(nm))
                except KeyError:
                    raise KeyError('Could not find entry "{0}" in prior dictionary'.format(nm))
            if len(priordct) > 0:
                raise ValueError('some prior dictionary elements not set: '
                                 '{0}'.format(priordct.keys()))
        self.priors = priors

    def get_prior(self, paramnm):
        return dict(zip(self.param_names, self.priors))[paramnm]

    @abc.abstractmethod
    def lnprob(self, *params):
        raise NotImplementedError

    def cross_priors(self, *args):
        """
        Computes any prior terms that mix parameters. Returns log(pri)
        """
        return 0

    def __call__(self, params):
        """
        returns log(prob)
        """

        lnpri = 0
        for arg, pri in zip(params, self.priors):
            lnpri += pri(arg)
            if lnpri == MINF:
                return MINF
        cpri = self.cross_priors(*params)
        if cpri == MINF:
            return MINF

        return lnpri + cpri + np.sum(self.lnprob(*params))

    def args_to_param_dict(self, args):
        """
        Translate an argument list into a dictionary mapping names to
        parameters.
        """
        return zip(self.param_names, args)

    def get_sampler(self, nwalkers=None, **kwargs):
        """
        nwalkers defaults to 4x parameters.  Remaining kwargs go into
        emcee.EnsembleSampler
        """
        if nwalkers is None:
            nwalkers = self.default_walkers

        return emcee.EnsembleSampler(nwalkers, len(self.param_names), self, **kwargs)

    def initalize_params(self, cendct={}, stddevdct={}, nwalkers=None):
        if nwalkers is None:
            nwalkers = len(self.param_names)*4

        cendct = dict(cendct)
        stddevdct = dict(stddevdct)

        iparams = []
        for n in self.param_names:
            cen = cendct.pop(n, None)
            if cen is None:
                try:
                    iparams.append(self.get_prior(n).initialize(nwalkers))
                except Exception as e:
                    raise ValueError('Problem sampling from parameter "{0}"'.format(n), e)
            else:
                iparams.append(np.random.randn(nwalkers) * stddevdct.pop(n, 1e-3))
        if cendct:
            raise ValueError('had unused cendct values {0}'.format())

        return np.array(iparams).T

    def initialize_and_sample(self, iters, burnin=None, cendct={}, stddevdct={},
                              nwalkers=None, **kwargs):
        if nwalkers is None:
            nwalkers = self.default_walkers

        iparams = self.initalize_params(cendct, stddevdct, nwalkers)
        self.last_sampler = sampler = self.get_sampler(nwalkers=None, **kwargs)

        if burnin:
            try:
                self.last_sampler = None
                iparams = sampler.run_mcmc(iparams, burnin)[0]
            finally:
                self.last_sampler = sampler
            self.last_burnin_chain = sampler.chain
            sampler.reset()

        try:
            self.last_sampler = None
            sampler.run_mcmc(iparams, iters)
        finally:
            self.last_sampler = sampler

        return sampler

    def triangle_plot(self, sampler=None, chainstoinclude='all', **kwargs):
        if sampler is None:
            sampler = self.last_sampler

        if chainstoinclude == 'all':
            msk = slice(None)
        elif isinstance(chainstoinclude, basestring):
            rex = re.compile(chainstoinclude)
            msk = np.array([bool(rex.match(pnm)) for pnm in self.param_names])
            chains = sampler.flatchain[:, msk]
            kwargs.setdefault('labels', np.array(self.param_names)[msk])
        else:
            # assume its a list of parameters
            chainstoinclude = list(chainstoinclude)
            msk = []
            for pnm in self.param_names:
                if pnm in chainstoinclude:
                    msk.append(True)
                    chainstoinclude.remove(pnm)
                else:
                    msk.append(False)
            msk = np.array(msk)
        chains = sampler.flatchain[:, msk]
        kwargs.setdefault('labels', np.array(self.param_names)[msk])

        triangle.corner(chains, **kwargs)

    def plot_chains(self, sampler=None, incl_burnin=True):
        from matplotlib import pyplot as plt

        if sampler is None:
            sampler = self.last_sampler
        if incl_burnin is True:
            burnin_ch = self.last_burnin_chain
        elif incl_burnin is False or incl_burnin is None:
            burnin_ch = None
        else:
            burnin_ch = incl_burnin

        subplot_rows = len(self.param_names)
        subplot_cols = 1 + int(burnin_ch is not None)
        for i, nm in enumerate(self.param_names):
            plt.subplot(subplot_rows, subplot_cols, i * subplot_cols + 1)
            if burnin_ch is not None:
                if i == 0:
                    plt.title('burnin')
                burnin_ch = self.last_burnin_chain.T[i]
                plt.plot(burnin_ch)
                plt.axvline(0, ls='--', c='k')
                plt.subplot(subplot_rows, subplot_cols, i * subplot_cols + 2)
                if i == 0:
                    plt.title('samples')
            ch = sampler.chain.T[i]
            plt.plot(ch)
            plt.ylabel(nm)

    def get_name_chain(self, nm, sampler=None):
        if sampler is None:
            sampler = self.last_sampler
            chain = sampler.chain
        elif sampler == 'burnin':
            chain = self.last_burnin_chain
        elif hasattr(sampler, 'chain'):
            chain = sampler.chain
        else:
            chain = sampler

        idx = self.param_names.index(nm)
        return chain[:, :, idx]

    @property
    def default_walkers(self):
        return len(self.param_names)*4
