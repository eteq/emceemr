from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


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
