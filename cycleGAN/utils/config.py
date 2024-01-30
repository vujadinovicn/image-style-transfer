PARAM_CONFIG = {
    'n_epochs': 84,
    'dim_A': 3,
    'dim_B': 3,
    'display_step': 200,
    'batch_size': 1,
    'lr': 0.0002,
    'load_shape': 286,
    'target_shape': 256,
    'device': 'cuda'
}

def flatten(config, except_keys=('bin_conf')):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield (key, value)
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))