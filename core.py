from __future__ import division, print_function

import abc

import re

import numpy as np
from matplotlib import pyplot as plt

import emcee
import triangle

MINF = -np.inf


#<--------sampler/emcee stuff-------->
class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, priors):
        if not getattr(self, 'param_names', None):
            raise ValueError('must have param_names')

        try:
            priordct = dict(priors)
        except ValueError:
            #assume it's a list
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
                    iparams.append(self.get_prior(n).sample(nwalkers))
                except Exception as e:
                    raise ValueError('Problem sampling from parameter "{0}"'.format(n), e)
            else:
                iparams.append(np.random.randn(nwalkers) * stddevdct.pop(n, 1e-3))
        if cendct:
            raise ValueError('had unused cendct values {0}'.format())

        return np.array(iparams).T

    def initialize_and_sample(self, iters, burnin=None, cendct={}, stddevdct={}, nwalkers=None, **kwargs):
        if nwalkers is None:
            nwalkers = self.default_walkers

        iparams = self.initalize_params(cendct, stddevdct, nwalkers)
        self.last_sampler = self.get_sampler(nwalkers=None)

        if burnin:
            iparams = self.last_sampler.run_mcmc(iparams, burnin)[0]
            self.last_burnin_chain = self.last_sampler.chain
            self.last_sampler.reset()

        self.last_sampler.run_mcmc(iparams, iters)

        return self.last_sampler

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
            #assume its a list of parameters
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
                #plt.gca().set_color_cycle(None)
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


class LineModel(Model):
    param_names = tuple('linecen, linesig, totflux, noisesig'.split(', '))

    def __init__(self, v, data, priors, polydeg=None, legendre_poly=False, baselinevdata=None):
        self.v = v
        self.data = data

        if baselinevdata is None:
            self.v_baseline = self.data_baseline = None
        else:
            self.v_baseline, self.data_baseline = baselinevdata

        if polydeg is None:
            # try to determine the polydeg from the largest 'poly#' # if priors
            # are given as a dictionary
            try:
                polydeg = 0
                for nm in priors.keys():
                    if nm.startswith('poly'):
                        deg = int(nm[4:]) + 1
                        if deg > polydeg:
                            polydeg = deg
            except AttributeError as e:
                if "object has no attribute 'keys'" in str(e):
                    polydeg = 0
                else:
                    raise

        if polydeg:
            self.param_names = list(self.param_names)
            for i in reversed(range(polydeg)):
                self.param_names.append('poly'+str(i))
            if legendre_poly:
                self.lpoly = np.polynomial.Legendre([0]*polydeg, domain=(self.v[0], self.v[-1]))
            else:
                self.lpoly = None
        super(LineModel, self).__init__(priors)

    def baseline(self, v, polyargs):
        if len(polyargs) > 0:
            if self.lpoly is None:
                return np.polyval(polyargs, v)
            else:
                self.lpoly.coef = np.array(polyargs[::-1])
                return self.lpoly(v)
        else:
            return 0

    def model(self, v, linecen, linesig, totflux, noisesig, *polyargs):
        if v is None:
            v = self.v

        exparg = -0.5*((v-linecen)/linesig)**2
        return totflux*(2*np.pi)**-0.5*np.exp(exparg) / linesig + self.baseline(v, polyargs)

    def lnprob(self, linecen, linesig, totflux, noisesig, *polyargs):
        model = self.model(self.v, linecen, linesig, totflux, noisesig, *polyargs)
        lnp = -0.5*((self.data - model)/noisesig)**2 - np.log(noisesig)

        if self.v_baseline is None:
            return lnp
        else:
            baselinelnp = -0.5*((self.data_baseline - self.baseline(self.v_baseline, polyargs))/noisesig)**2 - np.log(noisesig)
            return np.concatenate((lnp, baselinelnp))

    def plot_model(self, sampler=None, perc=50, plot_model=True, plot_data=True,
                   plot_baseline_data=False, plot_baseline=True, msk=None,
                   data_smoothing=None):
        from scipy import hanning

        if sampler is None:
            sampler = self.last_sampler

        if msk is None:
            msk = slice(None)

        params = np.percentile(sampler.flatchain[msk], perc, axis=0)
        if data_smoothing:
            smoothing_window = hanning(data_smoothing*2 + 3)
            smoothing_window = smoothing_window/np.sum(smoothing_window)
        else:
            smoothing_window = None

        if plot_data:
            if smoothing_window is None:
                data = self.data
            else:
                data = np.convolve(self.data, smoothing_window, 'same')
            plt.step(self.v, data, c='k', where='mid')

        if plot_baseline_data:
            edges = list(np.where(np.diff(self.v_baseline)<0)[0]+1)
            edges.append(None)
            edges.insert(0, None)
            for idx1, idx2 in zip(edges[:-1], edges[1:]):
                if smoothing_window is None:
                    bdata = self.data_baseline
                else:
                    bdata = np.convolve(self.data_baseline, smoothing_window, 'same')
                plt.step(self.v_baseline[idx1:idx2], bdata[idx1:idx2], c='g', where='mid')

        if plot_baseline:
            plt.plot(self.v, self.baseline(self.v, params[4:]), c='b', ls=':', lw=2)

        if plot_model:
            plt.plot(self.v, self.model(self.v, *params), c='r', lw=2)


