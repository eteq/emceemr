from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import abc

import numpy as np
import emcee

__all__ = ['Model']

MINF = -np.inf


class Model(object):
    __metaclass__ = abc.ABCMeta
    has_blobs = False

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
                if self.has_blobs:
                    return MINF, None
                else:
                    return MINF
        cpri = self.cross_priors(*params)
        if cpri == MINF:
            if self.has_blobs:
                return MINF, None
            else:
                return MINF

        if self.has_blobs:
            lnp, blob = self.lnprob(*params)
            return lnpri + cpri + np.sum(lnp), blob
        else:
            return lnpri + cpri + np.sum(self.lnprob(*params))

    def args_to_param_dict(self, args):
        """
        Translate an argument list into a dictionary mapping names to
        parameters.
        """
        return zip(self.param_names, args)

    def get_sampler(self, nwalkers='4p', **kwargs):
        """
        If a string with the letter p, means that many times the number of
        parameters. Anything else that's not string-convertable is invalid.

        Remaining kwargs go into emcee.EnsembleSampler
        """
        if isinstance(nwalkers, str):
            nwalkers = int(nwalkers.replace('p', '')) * len(self.param_names)
        else:
            nwalkers = int(nwalkers)

        return emcee.EnsembleSampler(nwalkers, len(self.param_names), self, **kwargs)

    def initalize_params(self, nwalkers):
        iparams = [self.get_prior(n).initialize(nwalkers) for n in self.param_names]
        return np.array(iparams).T

    def initialize_and_sample(self, iters, burnin=None, nwalkers='4p',
                                    use_progress_bar=True, **kwargs):
        sampler = self.get_sampler(nwalkers=nwalkers, **kwargs)
        iparams = self.initalize_params(sampler.k)

        if burnin:
            iparams = _run_mcmc_maybe_progress_bar(sampler, iparams, burnin,
                                                   None, None, use_progress_bar)[0]
            sampler.burnin_chain = sampler.chain
            sampler.burnin_lnprob = sampler.lnprobability
            sampler.reset()

        _run_mcmc_maybe_progress_bar(sampler, iparams, iters, None, None, use_progress_bar)

        return sampler

def _run_mcmc_maybe_progress_bar(sampler, p0, iters, rstate0, lnprob0, use_progress_bar):
    if use_progress_bar:
        from astropy.utils.console import ProgressBar

        try:
            use_ipy_widget = get_ipython().config['IPKernelApp']['parent_appname'] == 'ipython-notebook'
        except:
            # means we're not in IPython
            use_ipy_widget = False

        with ProgressBar(iters, ipython_widget=use_ipy_widget) as bar:
            #run_mcmc is esentially the same as this for loop, but without the update
            for results in sampler.sample(p0, rstate0, lnprob0, iterations=iters):
                bar.update()
            sampler._last_run_mcmc_result = results[:3]
    else:
        results = sampler.run_mcmc(p0, iters, rstate0, lnprob0)

    return results
