from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .core import Model

__all__ = ['LineModel']


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
            dmm_baseline = self.data_baseline - self.baseline(self.v_baseline, polyargs)
            baselinelnp = -0.5*(dmm_baseline/noisesig)**2 - np.log(noisesig)
            return np.concatenate((lnp, baselinelnp))

    def plot_model(self, sampler=None, perc=50, plot_model=True, plot_data=True,
                   plot_baseline_data=False, plot_baseline=True, msk=None,
                   data_smoothing=None):
        from matplotlib import pyplot as plt
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
            edges = list(np.where(np.diff(self.v_baseline) < 0)[0]+1)
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
