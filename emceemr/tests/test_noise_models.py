import numpy as np

from scipy import stats

from astropy.utils import find_current_module

from ..noise_models import generate_logpdfmodel_class


def test_logpdfmodel_generator():
    Mod = generate_logpdfmodel_class(stats.beta(1, 2, scale=3))
    assert Mod.name == 'BetaLogPDFModel'

    mod = Mod()
    assert mod(1) == stats.beta.logpdf(1, 1, 2, scale=3)
    assert Mod.__module__ == find_current_module(1).__name__

    Mod2 = generate_logpdfmodel_class(stats.norm(scale=1), scale_param_name='sigma')
    mod2 = Mod2(prob_loc=[0, 1])
    mod2.sigma = 1.5

    val1 = stats.norm.logpdf(0, loc=0, scale=1.5)
    val2 = stats.norm.logpdf(0, loc=1, scale=1.5)

    assert np.all(mod2(0) == np.array([val1, val2]))
