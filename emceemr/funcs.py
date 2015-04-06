from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import re

import numpy as np

import triangle

__all__ = ['triangle_plot', 'plot_chains', 'get_chain_by_name', 'get_percentiles']


def triangle_plot(model, sampler, chainstoinclude='all', **kwargs):
    if chainstoinclude == 'all':
        msk = slice(None)
    elif isinstance(chainstoinclude, basestring):
        rex = re.compile(chainstoinclude)
        msk = np.array([bool(rex.match(pnm)) for pnm in model.param_names])
        chains = sampler.flatchain[:, msk]
        kwargs.setdefault('labels', np.array(model.param_names)[msk])
    else:
        # assume its a list of parameters
        chainstoinclude = list(chainstoinclude)
        msk = []
        for pnm in model.param_names:
            if pnm in chainstoinclude:
                msk.append(True)
                chainstoinclude.remove(pnm)
            else:
                msk.append(False)
        msk = np.array(msk)
    chains = sampler.flatchain[:, msk]
    kwargs.setdefault('labels', np.array(model.param_names)[msk])

    triangle.corner(chains, **kwargs)


def plot_chains(model, sampler, incl_burnin=True):
    from matplotlib import pyplot as plt

    if incl_burnin is True:
        burnin_ch = sampler.burnin_chain
    elif incl_burnin is False or incl_burnin is None:
        burnin_ch = None
    else:
        burnin_ch = incl_burnin

    subplot_rows = len(model.param_names)
    subplot_cols = 1 + int(burnin_ch is not None)
    for i, nm in enumerate(model.param_names):
        plt.subplot(subplot_rows, subplot_cols, i * subplot_cols + 1)
        if burnin_ch is not None:
            if i == 0:
                plt.title('burnin')
            burnin_ch = sampler.burnin_chain.T[i]
            plt.plot(burnin_ch)
            plt.axvline(0, ls='--', c='k')
            plt.subplot(subplot_rows, subplot_cols, i * subplot_cols + 2)
            if i == 0:
                plt.title('samples')
        ch = sampler.chain.T[i]
        plt.plot(ch)
        plt.ylabel(nm)


def get_chain_by_name(model, sampler, nm, incl_burnin=True):
    idx = model.param_names.index(nm)
    chain = sampler.chain

    if incl_burnin is True:
        burnin_ch = sampler.burnin_chain
    elif incl_burnin is False or incl_burnin is None:
        burnin_ch = None
    else:
        burnin_ch = incl_burnin

    if burnin_ch is not None:
        chain = np.concatenate((burnin_ch, chain), axis=1)

    return chain[:, :, idx]


def get_percentiles(sampler, fracs, individualchains=False):
    """
    Get fractional samples from the chains.  I.e. .1 is the 10th percentile,
    .5 is the median, etc.

    First axis is parameters, last is over `fracs`
    """
    if individualchains:
        chain = sampler.chain
    else:
        chain = sampler.flatchain

    return np.percentile(chain, np.array(fracs)*100, axis=0).T


def continue_sampling(sampler, iters):
    lastparams = sampler.chain[:, -1, :].T
    return sampler.run_mcmc(lastparams, iters,
                            lnprob0=sampler.lnprobability[:, -1],
                            rstate0=sampler.random_state)


def reassign_burnin(sampler, burninidx):
    if hasattr(sampler, 'burnin_chain'):
        fullchain = np.concatenate((sampler.burnin_chain, sampler.chain), axis=1)
        fulllnprobs = np.concatenate((sampler.burnin_lnprob, sampler.lnprobability), axis=1)
    else:
        fullchain = sampler.chain
        fulllnprobs = sampler.lnprobability

    bislc = (slice(None), slice(None, burninidx), slice(None))
    sampleslc = (slice(None), slice(burninidx, None), slice(None))

    sampler.burnin_chain = fullchain[bislc]
    sampler._chain = fullchain[sampleslc]

    sampler.burnin_lnprob = fulllnprobs[bislc]
    sampler._lnprob = fulllnprobs[sampleslc]
