from fastai.basics import pd, plt, torch
from fastai.metrics import F1Score


def calc_tryout_metrics(hparams_tryouts, result_dir, metric):
    results = {}
    for param, values in hparams_tryouts.items():
        res = {}
        for v in values:
            v_str = '-'.join([str(v_) for v_ in v]) if isinstance(v, tuple) else str(v)
            df = pd.read_feather(f'{result_dir}/{param}_{v_str}/df.ftr')
            res.update({v_str: calc_metric(df, metric)})
        results.update({param: pd.DataFrame(res)})
    return results


def calc_metric(df, metric):
    result = []
    cols = [str(i) for i in range(10) if str(i) in df.columns]
    pred_cols = [f'{c}_pred' for c in cols]
    for _, fold_df in df.groupby('fold'):
        targ, pred = fold_df[cols].values, fold_df[pred_cols].values
        targ, pred = torch.from_numpy(targ), torch.from_numpy(pred)
        result.append(metric(targ.argmax(dim=1), pred.argmax(dim=1)))
    return result


def plot_tryout_metrics(metrics, x_labs):
    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    fs = fig.subfigures(nrows=3, ncols=2)
    for f, (param, result) in zip(fs.flatten(), metrics.items()):
        ax = f.subplots(nrows=1, ncols=1)
        x = 1 - result.columns.values.astype(float) if 'beta' in param else result.columns.values
        mean, std = result.mean(axis=0), result.std(axis=0)
        ax.plot(x, mean)
        ax.fill_between(x, mean - std, mean + std, facecolor='lightblue', alpha=.3)
        ax.set_xlabel(x_labs[param])
        ax.set_ylabel('F1 score (unweighted)')
        ax.set_ylim(.75, .85)
        if 'beta' in param:
            beta_log_scale(ax)
        # if param == 'n_context_events':
        #     ax.set_xscale('log')
        #     #ax.set_xlim(100, 2000)
    return fig


def beta_log_scale(ax):
    ax.set_xscale('log')
    a = ax.get_xticks().tolist()
    a = [1 - e for e in a]
    ax.set_xticklabels(a)
    ax.invert_xaxis()


if __name__ == '__main__':
    from paper_figs.utils import start_usetex

    start_usetex()

    x_labels = {'gamma': f'Focal loss parameter $\gamma$',
                'beta': r'Loss weight parameter $\beta_{loss}$',
                'oversample_beta': r'Sampling weight parameter $\beta_{sampling}$',
                'n_context_events': r'Number of context events $n_{context}$',
                'n_filter': r'Number of filters $n_{filter}$',
                'n_layer': r'Number of layers $n_{layer}$'}
    metric = F1Score(average='macro')
    result_dir = '/home/lfisch/Projects/gatenet_old/data/results/flowcap/hparams_NDD'

    # hparams_tryouts = {'oversample_beta': [.1, .9, .99, .999, .9999],
    #                    'beta': [.1, .9, .99, .999, .9999],
    #                    'gamma': [1, 2, 5, 10],
    #                    'n_context_events': [100, 200, 500, 1000, 2000]}
    hparams_tryouts = {'oversample_beta': [.1, .9, .99, .999, .9999],
                       'beta': [.1, .9, .99, .999, .9999],
                       'gamma': [1, 2, 5, 10],
                       'n_context_events': [100, 200, 500, 1000, 2000],
                       'n_filter': [(1024, 512), (1024, 512, 256), (1024, 512, 256, 128), (1024, 512, 256, 128, 64),
                                    (512, 256, 128), (1024, 512, 256), (2048, 1024, 512), (4096, 2048, 1024)]}

    tryout_metrics = calc_tryout_metrics(hparams_tryouts, result_dir, metric)
    df_layers = tryout_metrics['n_filter'][['1024-512', '1024-512-256', '1024-512-256-128', '1024-512-256-128-64']]
    df_layers = df_layers.rename(columns={'1024-512': '2', '1024-512-256': '3',
                                          '1024-512-256-128': '4', '1024-512-256-128-64': '5'})
    df_filters = tryout_metrics['n_filter'][['512-256-128', '1024-512-256', '2048-1024-512', '4096-2048-1024']]
    df_filters = df_filters.rename(columns={'512-256-128': '512', '1024-512-256': '1024',
                                            '2048-1024-512': '2048', '4096-2048-1024': '4096'})
    tryout_metrics.update({'n_filter': df_filters, 'n_layer': df_layers})

    figure = plot_tryout_metrics(tryout_metrics, x_labels)
    figure.savefig('hparams_new2.pdf')
    figure.savefig('hparams_new2.png')
    # figure.show()
