from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import abstractmethod
from scipy import special

import numpy as np

from .core import Model

__all__ = ['LineModel', 'GaussianLineModel', 'LorentzianLineModel',
           'VoigtLineModel', 'SersicModel']


class LineModel(Model):
    """
    A model for a spectral line with an optional polynomial continuum/baseline
    (called the "background" here). This is an abstract class - concrete classes
    need the model method.
    """
    param_names = tuple('linecen, linewidth, totflux, noisesig'.split(', '))

    def __init__(self, x, data, priors, polydeg=None, legendre_poly=False,
                       backgroundxdata=None):
        self.x = x
        self.data = data

        if backgroundxdata is None:
            self.x_background = self.data_background = None
        else:
            self.x_background, self.data_background = backgroundxdata

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

    def background(self, v, polyargs):
        if len(polyargs) > 0:
            if self.lpoly is None:
                return np.polyval(polyargs, v)
            else:
                self.lpoly.coef = np.array(polyargs[::-1])
                return self.lpoly(v)
        else:
            return 0

    @abstractmethod
    def model(self, x, linecen, linewidth, totflux, noisesig, *polyargs):
        raise NotImplementedError

    def lnprob(self, linecen, linewidth, totflux, noisesig, *polyargs):
        model = self.model(self.x, linecen, linewidth, totflux, noisesig, *polyargs)
        lnp = -0.5*((self.data - model)/noisesig)**2 - np.log(noisesig)

        if self.x_background is None:
            return lnp
        else:
            dmm_background = self.data_background - self.background(self.x_background, polyargs)
            backgroundlnp = -0.5*(dmm_background/noisesig)**2 - np.log(noisesig)
            return np.concatenate((lnp, backgroundlnp))

    def plot_model(self, sampler=None, perc=50, plot_model=True, plot_data=True,
                   plot_background_data=False, plot_background=True, msk=None,
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
            plt.step(self.x, data, c='k', where='mid')

        if plot_background_data:
            edges = list(np.where(np.diff(self.x_background) < 0)[0]+1)
            edges.append(None)
            edges.insert(0, None)
            for idx1, idx2 in zip(edges[:-1], edges[1:]):
                if smoothing_window is None:
                    bdata = self.data_background
                else:
                    bdata = np.convolve(self.data_background, smoothing_window, 'same')
                plt.step(self.x_background[idx1:idx2], bdata[idx1:idx2], c='g', where='mid')

        if plot_background:
            plt.plot(self.x, self.background(self.x, params[4:]), c='b', ls=':', lw=2)

        if plot_model:
            plt.plot(self.x, self.model(self.x, *params), c='r', lw=2)


class GaussianLineModel(LineModel):
    def model(self, x, linecen, linewidth, totflux, noisesig, *polyargs):
        if x is None:
            x = self.x

        exparg = -0.5*((x-linecen)/linewidth)**2
        return totflux*(2*np.pi)**-0.5*np.exp(exparg) / linewidth + self.background(x, polyargs)


class LorentzianLineModel(LineModel):
    def model(self, x, linecen, linewidth, totflux, noisesig, *polyargs):
        if x is None:
            x = self.x

        return totflux/(1+((x-linecen)/linewidth)**2)/np.pi/linewidth + self.background(x, polyargs)


class VoigtLineModel(LineModel):
    param_names = tuple('linecen, gausssig, lorentzwidth, totflux, noisesig'.split(', '))

    def model(self, x, linecen, gausssig, lorentzwidth, totflux, noisesig, *polyargs):
        if x is None:
            x = self.x

        z = (x + 1j*lorentzwidth)*2**-0.5/gausssig
        line = special.erfcx(-1j*z).real*(2*np.pi)**-0.5/gausssig
        return totflux*line + self.background(x, polyargs)

    def lnprob(self, linecen, gausssig, lorentzwidth, totflux, noisesig, *polyargs):
        model = self.model(self.x, linecen, gausssig, lorentzwidth, totflux, noisesig, *polyargs)
        lnp = -0.5*((self.data - model)/noisesig)**2 - np.log(noisesig)

        if self.x_background is None:
            return lnp
        else:
            dmm_background = self.data_background - self.background(self.x_background, polyargs)
            backgroundlnp = -0.5*(dmm_background/noisesig)**2 - np.log(noisesig)
            return np.concatenate((lnp, backgroundlnp))


class SersicModel(Model):
    param_names = 'flux, n, reffmaj, ellipticity, cenx, ceny, theta'.split(', ')
    def __init__(self, xg, yg, dat, ivar, priors):
        self.xg = xg
        self.yg = yg
        self.dat = dat
        self.ivar = ivar
        self.infiniteivar = ~np.isfinite(self.ivar)

        super(SersicModel, self).__init__(priors)

    def lnprob(self, flux, n, reffmaj, ellipticity, cenx, ceny, theta):
        mod = self.get_model(flux, n, reffmaj, ellipticity, cenx, ceny, theta)
        ddat = self.dat - mod
        res = -0.5 * (self.ivar * ddat * ddat)  # + np.log(2*np.pi*self.ivar))  #just a constant
        res[self.infiniteivar] = 0
        return res

    def get_model(self, flux, n, reffmaj, ellipticity, cenx, ceny, theta):
        dx = self.xg - cenx
        dy = self.yg - ceny
        if theta != 0:
            thrad = np.radians(theta)
            rotmat = [[np.cos(thrad), -np.sin(thrad)],
                      [np.sin(thrad), np.cos(thrad)]]
            dxy = np.array([dx, dy], copy=False)
            dx, dy = np.dot(rotmat, dxy.reshape(2, -1)).reshape(dxy.shape)

        reffmin = reffmaj * (1 - ellipticity)
        # rreff = np.hypot(dx/reffmin, dy/reffmaj)
        dx = dx / reffmin
        dy = dy / reffmaj
        rreff = np.sqrt(dx*dx + dy*dy)

        bn = self.bn(n)
        # normalize to make the integral over the areal profile to infinity = 1
        N = bn**(2*n) / (2*np.pi * reffmaj*reffmin * n * special.gamma(2*n))
        return flux * N * np.exp(-bn * rreff**(1/n))

        # more expressive way:
        # N = 2*np.pi*reffmaj*reffmin * n * special.gamma(2*n) * ebn * bn**(-2*n)
        # return flux * np.exp(-bn*(rreff**(1/n)-1)) / N

    @staticmethod
    def bn(n):
        return special.gammaincinv(2 * n, 0.5)
