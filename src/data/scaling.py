from fastai.basics import np, pd
SCALING_METHODS = {'arcsinh': lambda x: np.arcsinh(x / 5), 'log': lambda x: np.log(x + 1),
                   'div_100': lambda x: .001 * x}  # TODO: Move to own analysis branch
INV_SCALING_METHODS = {'arcsinh': lambda x: 5 * np.sinh(x), 'log': lambda x: np.exp(x) - 1,
                       'div_100': lambda x: 1000. * x}  # TODO: Move to own analysis branch


def apply_scaling(fcs: pd.DataFrame, scaling: dict, methods: dict = SCALING_METHODS):
    for m, cols in scaling.items():
        fcs[cols] = methods[m](fcs[cols])
    return fcs
