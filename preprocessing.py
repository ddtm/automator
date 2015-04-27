from copy import deepcopy
from itertools import product
from string import Template

EXPERIMENT_BASE_DEFAULTS = {
    'unroll': 'zip',
    'weights': '',
    'model': {'values': {}},
    'solver': {'values': {}},
    'watch': [],
    'command': None
}

def merge_dicts(*args):
    def merge(a, b, path=[]):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    merge(a[key], b[key], path + [str(key)])
                elif a[key] != b[key]:
                    a[key] = b[key]
            else:
                if isinstance(b[key], dict):
                    a[key] = deepcopy(b[key])
                else:
                    a[key] = b[key]
        return a
    return reduce(merge, args, {})

def preprocess_experiments(data):
    experiments = deepcopy(data['experiments'])
    defaults = data['defaults']

    fill_defaults(experiments, defaults)
    experiments = unroll_values(experiments)

    return experiments

def fill_defaults(experiments, defaults):
    for i in xrange(len(experiments)):
        experiments[i] = merge_dicts(EXPERIMENT_BASE_DEFAULTS, defaults, experiments[i])

def unroll_values(experiments):
    def get_path_template_dict(d):
        path_template_d = {}
        for k1 in ['model', 'solver']:
            for k2, v in d[k1]['values'].items():
                if isinstance(v, float):
                    v = '{:g}'.format(v)
                path_template_d[k1 + '_' + k2] = v
        return path_template_d

    unrolled_experiments = []
    for e in experiments:
        mode = e['unroll']
        if mode == 'product':
            generator = product
        elif mode == 'zip':
            generator = zip
        else:
            raise NotImplementedError()

        to_unroll = []
        for k1 in ['model', 'solver']:
            to_unroll.extend([(k1, k2, e[k1]['values'][k2]) 
                              for k2 in e[k1]['values'].keys() 
                              if isinstance(e[k1]['values'][k2], list)])

        if len(to_unroll) == 0:
            path_template_d = get_path_template_dict(e)
            e['path'] = Template(e['path']).substitute(path_template_d)
            unrolled_experiments.append(e)
            continue

        for t in generator(*[x[2] for x in to_unroll]):
            flat_e = deepcopy(e)
            for i, v in enumerate(t):
                k1 = to_unroll[i][0]
                k2 = to_unroll[i][1]
                flat_e[k1]['values'][k2] = v

            path_template_d = get_path_template_dict(flat_e)
            flat_e['path'] = Template(flat_e['path']).substitute(path_template_d)
            unrolled_experiments.append(flat_e)

    return unrolled_experiments
