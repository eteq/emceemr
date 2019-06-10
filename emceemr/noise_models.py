import collections

from astropy import modeling
from astropy.utils import find_current_module

__all__ = ['generate_logpdfmodel_class']

def generate_logpdfmodel_class(frozen_distribution,
                               scale_param_name='prob_scale',
                               location_param_name='prob_loc',
                               data_param_name='prob_loc'):

    dist = frozen_distribution.dist
    other_params, loc, scale = dist._parse_args(*frozen_distribution.args, **frozen_distribution.kwds)
    other_param_names = [] if dist.shapes is None else dist.shapes.split(', ')

    assert len(other_params) == len(other_param_names), 'This stats object seems messed up - the names of the parameters do not match the number of parameters'

    # order matters here!  most go "other parameters in order, loc, scale"
    params = []
    for nm, p in zip(other_param_names, other_params):
        params.append(modeling.Parameter(name=nm, default=p))
    params.append(modeling.Parameter(name=location_param_name, default=loc, description='distribution location parameter'))
    params.append(modeling.Parameter(name=scale_param_name, default=scale, description='distribution scale parameter'))

    params = collections.OrderedDict(((p.name, p) for p in params))

    members = {}

    def tester(*args, **kwargs):
        print(args, kwargs)
        return dist.logpdf(*args, **kwargs)

    mod = find_current_module(2)
    modname = mod.__name__ if mod else '__main__'

    members = {
        '__module__': str(modname),
        'distribution': dist,
        'inputs': ('x',),
        'outputs': ('logpdf',),
        'evaluate': staticmethod(dist.logpdf),
    }
    members.update(params)

    return type(dist.name.title() + 'LogPDFModel', (modeling.FittableModel,), members)