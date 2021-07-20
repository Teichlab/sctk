"""
Plotting related functions
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import anndata
import scanpy as sc
if sc.__version__.startswith('1.4'):
    from scanpy.plotting._tools.scatterplots import plot_scatter
else:
    plot_scatter = sc.pl.embedding

from ._diffexp import extract_de_table
from ._utils import pseudo_bulk, sc_warn

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def expression_colormap(background_level=0.01):
    """Returns a nice color map for highlighting gene expression
    """
    background_nbin = int(100 * background_level)
    reds = plt.cm.Reds(np.linspace(0, 1, 100-background_nbin))
    greys = plt.cm.Greys_r(np.linspace(0.7, 0.8, background_nbin))
    palette = np.vstack([greys, reds])
    return LinearSegmentedColormap.from_list('expression', palette)


def make_palette(n, cmap=None, hide_first=False, hide_last=False, hide_color='#E9E9E910'):
    """Returns a color palette with specified number of colors
    """
    i = int(hide_first)
    j = int(hide_last)
    if cmap is None:
        if sc.__version__.startswith('1.4'):
            sc_default_10 = rcParams['axes.prop_cycle'].by_key()['color']
            sc_default_26 = sc.plotting.palettes.default_26
            sc_default_64 = sc.plotting.palettes.default_64
            palette = (sc_default_10[0:(n-i-j)] if n <= 10 + i + j else
                    sc_default_10 + sc_default_26[0:(n-i-j)] if n <= 36 + i + j else
                    sc_default_10 + sc_default_26 + sc_default_64[0:(n-i-j)] if n <= 100 + i + j else
                    ['grey'] * n)
        else:
            sc_default_20 = sc.plotting.palettes.default_20
            sc_default_28 = sc.plotting.palettes.default_28
            sc_default_102 = sc.plotting.palettes.default_102
            palette = (sc_default_20[0:(n-i-j)] if n <= 20 + i + j else
                    sc_default_20 + sc_default_28[0:(n-i-j)] if n <= 48 + i + j else
                    sc_default_102[0:(n-i-j)] if n<= 102 + i + j else
                    ['grey'] * n)
    else:
        color_map = plt.get_cmap(cmap)
        palette = [to_hex(color_map(k)) for k in range(n-i-j)]

    if hide_first:
        palette.insert(0, hide_color)
    if hide_last:
        palette.append(hide_color)
    return palette


def clear_colors(ad, slots=None):
    color_slots = [k for k in ad.uns.keys() if k.endswith('_colors')]
    if isinstance(slots, str):
        slots = [slots]
    elif slots is None:
        slots = [s.replace('_colors', '') for s in color_slots]
    if isinstance(slots, (tuple, list)):
        slots = [s for s in slots if f'{s}_colors' in color_slots]
    for s in slots:
        del ad.uns[f'{s}_colors']


def _is_numeric(x):
    return x.dtype.kind in ('i', 'f')


def abline(slope=1, intercept=0, ax=None, log=False, min_x=1):
    """Plot a line from slope and intercept"""
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    if log:
        if slope >= 0:
            x_vals[0] = max(min_x, (min_x - intercept)/slope, x_vals[0])
        else:
            x_vals[0] = max(min_x, x_vals[0])
            x_vals[1] = min((min_x - intercept)/slope, x_vals[1])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', c='k')


def cross_table(adata, x, y, normalise=None, exclude_x=None, exclude_y=None, subset=None, sort_index=False, include_nan=False, stylize=None, plot=False, cluster=False, figsize=(4,4), **kwargs):
    """Make a cross table comparing two categorical annotations
    """
    x_attr = adata.obs[x]
    y_attr = adata.obs[y]
    if include_nan:
        x_attr = x_attr.astype(str).astype('category')
        y_attr = y_attr.astype(str).astype('category')
    if subset is not None:
        x_attr = x_attr[subset]
        y_attr = y_attr[subset]
    assert not _is_numeric(x_attr.values), f'Can not operate on numerical {x}'
    assert not _is_numeric(y_attr.values), f'Can not operate on numerical {y}'
    crs_tbl = pd.crosstab(x_attr, y_attr, dropna=False)
    if normalise == 'x':
        x_sizes = x_attr.groupby(x_attr).size().values
        crs_tbl = (crs_tbl.T / x_sizes * 100).round(2).T
    elif normalise == 'y':
        y_sizes = x_attr.groupby(y_attr).size().values
        crs_tbl = (crs_tbl / y_sizes * 100).round(2)
    elif normalise == 'xy':
        x_sizes = x_attr.groupby(x_attr).size().values
        crs_tbl = (crs_tbl.T / x_sizes * 100).T
        y_sizes = crs_tbl.sum(axis=0)
        crs_tbl = (crs_tbl / y_sizes * 100).round(2)
    elif normalise == 'yx':
        y_sizes = x_attr.groupby(y_attr).size().values
        crs_tbl = (crs_tbl / y_sizes * 100).round(2)
        x_sizes = crs_tbl.sum(axis=1)
        crs_tbl = (crs_tbl.T / x_sizes * 100).round(2).T
    if exclude_x:
        crs_tbl = crs_tbl.loc[~crs_tbl.index.isin(exclude_x)].copy()
    if exclude_y:
        crs_tbl = crs_tbl.loc[:, ~crs_tbl.columns.isin(exclude_y)].copy()
    if sort_index:
        crs_tbl = crs_tbl.sort_index()
    if stylize is not None:
        return crs_tbl.style.background_gradient(cmap='viridis', axis=stylize)
    if plot:
        if cluster:
            sn.clustermap(crs_tbl, linewidths=0.01, cmap='viridis_r', figsize=figsize, **kwargs)
        else:
            set_figsize(figsize)
            sn.heatmap(crs_tbl, linewidths=0.01, cmap='viridis_r', **kwargs)
    return crs_tbl


def set_figsize(dim):
    if len(dim) == 2 and (isinstance(dim[0], (int, float))
            and isinstance(dim[1], (int, float))):
        rcParams.update({'figure.figsize': dim})
    else:
        raise ValueError(f'Invalid {dim} value, must be an iterable of '
                          'length two in the form of (width, height).')


def plot_df_heatmap(
        df,
        cmap='viridis',
        title=None,
        figsize=(7, 7),
        rotation=90,
        save=None,
        **kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df, cmap=cmap, aspect='auto', **kwargs)
    if 0 < rotation < 90:
        horizontalalignment = 'right'
    else:
        horizontalalignment = 'center'
    plt.xticks(
        range(len(df.columns)),
        df.columns,
        rotation=rotation,
        horizontalalignment=horizontalalignment,
    )
    plt.yticks(range(len(df.index)), df.index)
    if title:
        fig.suptitle(title)
    fig.colorbar(im)
    if save:
        plt.savefig(fname=save, bbox_inches='tight', pad_inches=0.1)


def _log_hist(x, bins=50, min_x=None, max_x=None, xlim=None, ax=None):
    x_copy = x.copy()
    if max_x is None:
        max_x = x[np.isfinite(x)].max()
    x_copy[x > max_x] = max_x
    if min_x is None:
        min_x1 = x[x>0].min()
        min_x = 10** (np.log10(min_x1) - (np.log10(max_x) - np.log10(min_x1))/(bins - 1))
    x_copy[x < min_x] = min_x

    log_bins = np.logspace(np.log10(min_x), np.log10(max_x), bins)
    if ax:
        ax.hist(x_copy, bins=log_bins)
        ax.set_xscale('log')
    else:
        fig, ax = plt.subplots()
        ax.hist(x_copy, bins=log_bins)
        ax.set_xscale('log')
        return ax


def dotplot2(
        adata, keys, groupby=None, min_group_size=0, min_presence=0, use_raw=None,
        mean_only_expressed=False, second_key_dependent_fraction=False, coexpression=False,
        vmin=0, vmax=1, dot_min=None, dot_max=None, color_map='Reds', swap_axis=False,
        legend_loc='right', title='', title_loc='top', title_size=None, omit_xlab=False, omit_ylab=False,
        xtickslabels=None, ytickslabels=None, ax=None, save=None, save_dpi=80, return_data=False, **kwargs):
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, (list, tuple)):
        raise ValueError(f'keys must be a tuple/list of obs keys or var_names')

    if second_key_dependent_fraction or coexpression:
        if len(keys) != 2:
            raise ValueError('Exactly two keys requied if `second_key_dependent_fraction=True`')

    cmap = plt.get_cmap(color_map)
    if use_raw or (use_raw is None and adata.raw):
        ad = adata.raw
    else:
        ad = adata

    if groupby is None:
        groups = pd.Series(np.ones(adata.n_obs)).astype(str).astype('category')
        n_group = 1
    elif groupby in adata.obs.columns:
        groups = adata.obs[groupby]
        if groups.dtype.name != 'category':
            groups = groups.astype('category')
    else:
        raise ValueError(f'{groupby} not found')
    groups.cat.remove_unused_categories(inplace=True)
    grouping = list(groups.cat.categories)

    n_obs = ad.shape[0]
    n_key = len(keys)
    y = np.zeros((n_key, n_obs))
    for i in range(n_key):
        key = keys[i]
        if key in adata.obs.columns:
            y[i] = adata.obs[key].values
        elif key in ad.var_names:
            if sc.__version__.startswith('1.4'):
                y[i] = ad[:, key].X
            else:
                y[i] = ad.obs_vector(key)
        else:
            sc_warn(f'{key} not found')
            y[i] = np.zeros(n_obs)

    df = pd.DataFrame(y.T, columns=keys)
    df['group'] = groups.reset_index(drop=True)
    df['group'].cat.set_categories(grouping, inplace=True)
    group_size = df.groupby('group').agg('size')
    if min_group_size:
        df = df.loc[df['group'].isin(group_size.index[group_size.values >= min_group_size]), :]
        df['group'].cat.remove_unused_categories(inplace=True)
    n_group = df['group'].cat.categories.size


    if second_key_dependent_fraction:
        exp_cnt = df.groupby('group')[keys].apply(lambda g: ((g>vmin).sum(axis=1)==2).sum()).values
        frac = df.groupby('group')[keys].apply(lambda g: ((g>vmin).sum(axis=1)==2).astype(int).mean())
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby('group')[[keys[0]]].apply(lambda g: g.mean(axis=0)).values
        avg1 = df.groupby('group')[[keys[0]]].apply(lambda g: g.sum(axis=0) / (g>vmin).sum(axis=0)).fillna(0).values
        avg = avg1 if mean_only_expressed else avg0
        keys = ['|'.join(keys)]
        n_key = 1
    elif coexpression:
        exp_cnt = df.groupby('group')[keys].apply(lambda g: ((g>vmin).sum(axis=1)==2).sum()).values
        frac = df.groupby('group')[keys].apply(lambda g: ((g>vmin).sum(axis=1)==2).astype(int).mean())
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby('group')[keys].apply(lambda g: g.min(axis=1).mean()).values
        avg1 = df.groupby('group')[keys].apply(lambda g: g.min(axis=1).sum() / max(1, (((g>vmin).sum(axis=1)==2)).sum())).fillna(0).values
        avg = avg1 if mean_only_expressed else avg0
        keys = ['+'.join(keys)]
        n_key = 1
    else:
        exp_cnt = df.groupby('group')[keys].apply(lambda g: (g>vmin).sum(axis=0)).values
        frac = df.groupby('group')[keys].apply(lambda g: (g>vmin).mean(axis=0))
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby('group')[keys].apply(lambda g: g.mean(axis=0)).values
        avg1 = df.groupby('group')[keys].apply(lambda g: g.sum(axis=0) / (g>vmin).sum(axis=0)).fillna(0).values
        avg = avg1 if mean_only_expressed else avg0

    frac[exp_cnt<min_presence] = 0
    avg[exp_cnt<min_presence] = 0

    data = pd.DataFrame({'gene': np.repeat(keys, n_group), 'group': np.tile(grouping, n_key), 'frac': frac.T.flatten(), 'avg': avg.T.flatten()})
    if return_data:
        return data

    if dot_max is None:
        dot_max = np.ceil(np.max(frac) * 10) / 10
    else:
        if dot_max < 0 or dot_max > 1:
            raise ValueError("`dot_max` value has to be between 0 and 1")
    if dot_min is None:
        dot_min = 0
    else:
        if dot_min < 0 or dot_min > 1:
            raise ValueError("`dot_min` value has to be between 0 and 1")

    if dot_min != 0 or dot_max != 1:
        # clip frac between dot_min and  dot_max
        frac = np.clip(frac, dot_min, dot_max)
        old_range = dot_max - dot_min
        # re-scale frac between 0 and 1
        frac = ((frac - dot_min) / old_range)

    frac = frac.T.flatten()
    avg = avg.T.flatten()

    dot_sizes = (frac * 10) ** 2
    normalize = Normalize(vmin=vmin, vmax=vmax)
    dot_colors = cmap(normalize(avg))

    legend_loc = 'none' if ax else legend_loc
    fig_width = 0.4 + (n_key if swap_axis else n_group) * 0.25 + 0.25 * int(legend_loc == 'right')
    fig_height = 0.4 + (n_group if swap_axis else n_key) * 0.2 + 0.25 * int(legend_loc == 'bottom')

    rcParams.update({'figure.figsize': (fig_width, fig_height)})
    if legend_loc == 'right':
        fig, axs = plt.subplots(ncols=2, nrows=1, gridspec_kw={'width_ratios': [fig_width-0.25, 0.25], 'wspace': 0.25/(n_key if swap_axis else n_group)})
    elif legend_loc == 'bottom':
        fig, axs = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [fig_height-0.25, 0.25], 'hspace': 0.25/(n_group if swap_axis else n_key)})
    elif not ax:
        fig, axs = plt.subplots(ncols=1, nrows=1)
        axs = [axs]
    else:
        axs = [ax]
    main = axs[0]

    if legend_loc == 'bottom':
        main.xaxis.tick_top()
    if swap_axis:
        main.scatter(y=np.tile(range(n_group), n_key), x=np.repeat(np.arange(n_key)[::-1], n_group), color=dot_colors[::-1], s=dot_sizes[::-1], cmap=cmap, norm=None, edgecolor='none', **kwargs)
        main.set_xticks(range(n_key))
        if xtickslabels is None:
            xtickslabels = keys
        main.set_xticklabels(xtickslabels, rotation=270)
        main.set_xlim(-0.5, n_key - 0.5)
        main.set_yticks(range(n_group))
        if ytickslabels is None:
            ytickslabels = group_label[::-1]
        main.set_yticklabels(ytickslabels)
        main.set_ylim(-0.5, n_group - 0.5)
    else:
        main.scatter(x=np.tile(range(n_group), n_key), y=np.repeat(np.arange(n_key), n_group), color=dot_colors, s=dot_sizes, cmap=cmap, norm=None, edgecolor='none', **kwargs)
        main.set_yticks(range(n_key))
        if ytickslabels is None:
            ytickslabels = keys
        main.set_yticklabels(ytickslabels)
        main.set_ylim(-0.5, n_key - 0.5)
        main.set_xticks(range(n_group))
        if xtickslabels is None:
            xtickslabels = group_label
        main.set_xticklabels(xtickslabels, rotation=270)
        main.set_xlim(-0.5, n_group - 0.5)
    if title:
        if title_loc == 'top':
            main.set_title(title, fontsize=15)
        elif title_loc == 'right':
            title_ax = axs[1] if legend_loc == 'right' else main
            ylab_position = 'left' if legend_loc == 'right' else 'right'
            ylab_pad = 0 if legend_loc == 'right' else 20
            title_ax.yaxis.set_label_position(ylab_position)
            if title_size is None:
                title_size = min(15, 100*fig_height/len(title))
            title_ax.set_ylabel(title, rotation=270, labelpad=ylab_pad, fontsize=title_size)
    if omit_xlab:
        main.tick_params(axis='x', bottom=False, labelbottom=False)
    if omit_ylab:
        main.tick_params(axis='y', left=False, labelleft=False)

    if legend_loc in ('right', 'bottom'):
        diff = dot_max - dot_min
        if 0.2 < diff <= 0.6:
            step = 0.1
        elif 0.06 < diff <= 0.2:
            step = 0.05
        elif 0.03 < diff <= 0.06:
            step = 0.02
        elif diff <= 0.03:
            step = 0.01
        else:
            step = 0.2
        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
        if dot_min != 0 or dot_max != 1:
            fracs_values = ((fracs_legends - dot_min) / old_range)
        else:
            fracs_values = fracs_legends
        size = (fracs_values * 10) ** 2
        color = [cmap(normalize(value)) for value in np.repeat(vmin+(vmax-vmin)*0.8, len(size))]

        # plot size bar
        size_legend = axs[1]
        labels = ["{:.0%}".format(x) for x in fracs_legends]
        if dot_max < 1:
            labels[-1] = ">=" + labels[-1]

        if legend_loc == 'bottom':
            size_legend.scatter(y=np.repeat(0, len(size)), x=range(len(size)), s=size, color='grey')
            size_legend.set_xticks(range(len(size)))
            size_legend.set_xticklabels(["{:.0%}".format(x) for x in fracs_legends], rotation=270)
            # remove y ticks and labels
            size_legend.tick_params(axis='x', bottom=False, labelbottom=True, pad=-2)
            size_legend.tick_params(axis='y', left=False, labelleft=False)
            xmin, xmax = size_legend.get_xlim()
            size_legend.set_xlim(xmin-((n_key if swap_axis else n_group)-1)*0.75, xmax+0.5)
        else:
            size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, color='grey')
            size_legend.set_yticks(range(len(size)))
            size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
            # remove x ticks and labels
            size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True, pad=-2)
            size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
            ymin, ymax = size_legend.get_ylim()
            size_legend.set_ylim(ymin - ((n_group if swap_axis else n_key)-1)*0.75, ymax+0.5)

        # remove surrounding lines
        size_legend.spines['right'].set_visible(False)
        size_legend.spines['top'].set_visible(False)
        size_legend.spines['left'].set_visible(False)
        size_legend.spines['bottom'].set_visible(False)
        size_legend.grid(False)

    if save:
        fig.savefig(save, bbox_inches='tight', dpi=save_dpi)
        plt.close()

    if ax:
        return fig_width, fig_height

    return main

def dotplot_combined_coexpression(
    ad, genes, groupby, groups=None, merge=False, title='', save=None, save_dpi=200, **kwargs
):
    print(save)
    n_gene = len(genes)
    assert n_gene == 2, 'genes must be '

    fig, ax = plt.subplots(ncols=n_gene + 1, gridspec_kw={'wspace': 0 if merge else 0.05})
    fW, fH = 0, 0
    if groups is not None and len(groups) != len(ad.obs[groupby].unique()):
        ad = ad[ad.obs[groupby].isin(groups), :]
    for i, gene in enumerate(genes):
        w, h = dotplot2(
            ad, gene, groupby=groupby, swap_axis=True, ax=ax[i], omit_xlab=False, omit_ylab=i>0, title='', **kwargs
        )
        if merge:
            if i < n_gene - 1:
                ax[i].spines['right'].set_visible(False)
            else:
                ax[i].spines['left'].set_visible(False)
        fW += w
        fH = max(fH, h)
    i += 1
    w,h = dotplot2(
        ad, genes, groupby=groupby, coexpression=True, ax=ax[i], swap_axis=True,
        title=title, title_loc='right', omit_xlab=False, omit_ylab=True, **kwargs
    )
    fW += w
    fH = max(fH, h)

    fig.set_figwidth(fW)
    fig.set_figheight(fH)
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=save_dpi)
        plt.close()


def plot_qc(adata, groupby=None, groupby_only=False, one_per_line=False, **kwargs):
    old_figsize = rcParams.get('figure.figsize')
    qc_metrics = ['log1p_n_counts', 'log1p_n_genes'] + adata.obs.columns[adata.obs.columns.str.startswith('percent_')].to_list()
    if 'scrublet_score_z' in adata.obs.columns:
        qc_metrics.append('scrublet_score_z')
    for qmt in qc_metrics:
        if qmt not in adata.obs.columns:
            raise ValueError(f'{qmt} not found.')
    if groupby:
        if one_per_line:
            rcParams.update({'figure.figsize': (old_figsize[0],(1.1*old_figsize[1]-0.1)*len(qc_metrics))})
            fig, axs = plt.subplots(ncols=1, nrows=len(qc_metrics), sharex=True)
            for i, qcm in enumerate(qc_metrics):
                sc.pl.violin(adata, keys=qcm, groupby=groupby, rotation=90, stripplot=False, show=False, ax=axs[i], **kwargs)
        else:
            axs = sc.pl.violin(adata, keys=qc_metrics, groupby=groupby, rotation=90, stripplot=False, show=False, **kwargs)
        axs[0].axhline(y=np.log1p(1000), linewidth=0.5, color='k')
        axs[0].axhline(y=np.log1p(500), linewidth=0.5, color='k')
        axs[0].axhline(y=np.log1p(250), linewidth=0.5, color='k')
        axs[1].axhline(y=np.log1p(200), linewidth=0.5, color='k')
        axs[1].axhline(y=np.log1p(100), linewidth=0.5, color='k')
        axs[1].axhline(y=np.log1p(50), linewidth=0.5, color='k')
    if not groupby_only:
        g = sc.pl.violin(adata, keys=qc_metrics, multi_panel=True, stripplot=False, show=False)
        g.facet_axis(0, 0).axhline(y=np.log1p(1000), linewidth=0.5, color='k')
        g.facet_axis(0, 0).axhline(y=np.log1p(500), linewidth=0.5, color='k')
        g.facet_axis(0, 0).axhline(y=np.log1p(250), linewidth=0.5, color='k')
        g.facet_axis(0, 1).axhline(y=np.log1p(200), linewidth=0.5, color='k')
        g.facet_axis(0, 1).axhline(y=np.log1p(100), linewidth=0.5, color='k')
        g.facet_axis(0, 1).axhline(y=np.log1p(50), linewidth=0.5, color='k')
        rcParams.update({'figure.figsize': (15,3)})
        fig, ax = plt.subplots(ncols=4, nrows=1)
        ax[0].hexbin(x=adata.obs['log1p_n_counts'], y=adata.obs['percent_mito'], cmap='binary')
        ax[0].set_xlabel('log1p_n_counts')
        ax[0].set_ylabel('percent_mito')
        ax[1].hexbin(x=adata.obs['log1p_n_counts'], y=adata.obs['percent_top50'], cmap='binary')
        ax[1].set_xlabel('log1p_n_counts')
        ax[1].set_ylabel('percent_top50')
        ax[2].hexbin(x=adata.obs['percent_mito'], y=adata.obs['percent_ribo'], cmap='binary')
        ax[2].set_xlabel('percent_mito')
        ax[2].set_ylabel('percent_ribo')
        ax[3].hexbin(x=adata.obs['log1p_n_counts'], y=adata.obs['log1p_n_genes'], cmap='binary')
        ax[3].set_xlabel('log1p_n_counts')
        ax[3].set_ylabel('log1p_n_genes')
        rcParams.update({'figure.figsize': old_figsize})


def plot_metric_by_rank(
        adata,
        subject='cell',
        metric='n_counts',
        kind='rank',
        nbins=50,
        order=True,
        decreasing=True,
        ax=None,
        title=None,
        hpos=None,
        vpos=None,
        logx=True,
        logy=True,
        swap_axis=False,
        **kwargs,
):
    """Plot metric by rank from top to bottom"""
    kwargs['c'] = kwargs.get('c', 'black')
    metric_names = {
        'n_counts': 'nUMI', 'n_genes': 'nGene', 'n_cells': 'nCell', 'percent_mito': 'fracMT', 'percent_viral': 'fracViral'}
    if subject not in ('cell', 'gene'):
        print('`subject` must be "cell" or "gene".')
        return

    if metric == 'percent_mito' and 'percent_mito' not in adata.obs.columns:
        if 'mito' in adata.var.columns:
            k_mt = adata.var.mito.astype(bool).values
        else:
            k_mt = adata.var_names.str.startswith('MT-')
        if sum(k_mt) > 0:
            adata.obs['percent_mito'] = np.squeeze(
                np.asarray(adata.X[:, k_mt].sum(axis=1))) / adata.obs['n_counts'] * 100

    if not ax:
        fig, ax = plt.subplots()

    if kind == 'rank':
        order_modifier = -1 if decreasing else 1

        if subject == 'cell':
            if metric not in adata.obs.columns:
                if metric == 'n_counts':
                    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
                if metric == 'n_genes':
                    adata.obs['n_genes'] = (adata.X>0).sum(axis=1).A1
            if swap_axis:
                x = 1 + np.arange(adata.shape[0])
                y = adata.obs[metric].values
                if order is not False:
                    k = np.argsort(order_modifier * y) if order is True else order
                    y = y[k]
            else:
                y = 1 + np.arange(adata.shape[0])
                x = adata.obs[metric].values
                if order is not False:
                    k = np.argsort(order_modifier * x) if order is True else order
                    x = x[k]
        else:
            if metric not in adata.var.columns:
                if metric == 'n_counts':
                    adata.var['n_counts'] = adata.X.sum(axis=0).A1
            if swap_axis:
                x = 1 + np.arange(adata.shape[1])
                y = adata.var[metric].values
                if order is not False:
                    k = np.argsort(order_modifier * y) if order is True else order
                    y = y[k]
            else:
                y = 1 + np.arange(adata.shape[1])
                x = adata.var[metric].values
                if order is not False:
                    k = np.argsort(order_modifier * x) if order is True else order
                    x = x[k]

        if kwargs['c'] is not None and not isinstance(kwargs['c'], str):
            kwargs_c = kwargs['c'][k]
            del kwargs['c']
            ax.scatter(x, y, c=kwargs_c, **kwargs)
        else:
            marker = kwargs.get('marker', '-')
            ax.plot(x, y, marker, **kwargs)

        if logy:
            ax.set_yscale('log')
        if swap_axis:
            ax.set_xlabel('Rank')
        else:
            ax.set_ylabel('Rank')

    elif kind == 'hist':
        del kwargs['c']
        if subject == 'cell':
            value = adata.obs[metric]
            logbins = np.logspace(np.log10(np.min(value[value>0])), np.log10(np.max(adata.obs[metric])), nbins)
            h = ax.hist(value[value>0], logbins, **kwargs)
            y = h[0]
            x = h[1]
        else:
            value = adata.var[metric]
            logbins = np.logspace(np.log10(np.min(value[value>0])), np.log10(np.max(adata.var[metric])), nbins)
            h = ax.hist(value[value>0], logbins, **kwargs)
            y = h[0]
            x = h[1]

        ax.set_ylabel('Count')

    if hpos is not None:
        ax.hlines(hpos, xmin=x.min(), xmax=x.max(), linewidth=1, colors='red')
    if vpos is not None:
        ax.vlines(vpos, ymin=y.min(), ymax=y.max(), linewidth=1, colors='green')
    if logx:
        ax.set_xscale('log')
    if swap_axis:
        ax.set_ylabel(metric_names.get(metric, metric))
    else:
        ax.set_xlabel(metric_names.get(metric, metric))
    if title:
        ax.set_title(title)
    ax.grid(linestyle='--')
    if 'label' in kwargs:
        ax.legend(loc='upper right' if decreasing else 'upper left')

    return ax


def plot_embedding(
        adata, groupby, basis='umap', color=None, annot=True, min_group_size=0,
        highlight=None, size=None, save=None, savedpi=300, figsize=(4,4), **kwargs):
    set_figsize(figsize)
    if f'X_{basis}' not in adata.obsm.keys():
        raise KeyError(f'"X_{basis}" not found in `adata.obsm`.')
    if min_group_size:
        group_sizes = adata.obs[groupby].value_counts()
        good_groups = group_sizes[group_sizes >= min_group_size].index
        adata = adata[adata.obs[groupby].isin(good_groups)].copy()
    if isinstance(groupby, (list, tuple)):
        groupby = groupby[0]
    if groupby not in adata.obs.columns:
        raise KeyError(f'"{groupby}" not found in `adata.obs`.')
    if adata.obs[groupby].dtype.name != 'category':
        if isinstance(adata.obs[groupby][0], (str, bool, np.bool_)) and adata.obs[groupby].unique().size < 100:
            adata.obs[groupby] = adata.obs[groupby].astype(str).astype('category')
        else:
            sc_warn(f'"{groupby}" is not categorical.')
            plot_scatter(adata, basis=basis, color=groupby, **kwargs)
    groups = adata.obs[groupby].copy()
    categories = list(adata.obs[groupby].cat.categories)
    rename_dict1 = {ct: f'{i:^5d} {ct} (n={(adata.obs[groupby]==ct).sum()})' for i, ct in enumerate(categories)}
    restore_dict1 = {f'{i:^5d} {ct} (n={(adata.obs[groupby]==ct).sum()})': ct for i, ct in enumerate(categories)}
    rename_dict2 = {ct: f'{i:^5d} {ct}' for i, ct in enumerate(categories)}
    restore_dict2 = {f'{i:^5d} {ct}': ct for i, ct in enumerate(categories)}

    size_ratio = 1.2

    ad = adata
    marker_size = size
    kwargs['show'] = False
    kwargs['save'] = False
    kwargs['frameon'] = kwargs.get('frameon', None)
    kwargs['legend_loc'] = 'right margin'

    color = groupby if color is None else color
    offset = 0 if 'diffmap' in basis else -1
    xi, yi = 1, 2
    if 'components' in kwargs:
        xi, yi = components
    xi += offset
    yi += offset

    adata.uns[f'{color}_colors'] = make_palette(adata.obs[color].cat.categories.size, kwargs.get('palette', None))

    if highlight:
        k_nohl = np.where(~adata.obs[groupby].cat.categories.isin(highlight))[0]
        for k in k_nohl:
            ad.uns[f'{color}_colors'][k] = ad.uns[f'{color}_colors'][k] + '05'

    if annot == 'full':
        adata.obs[groupby].cat.rename_categories(rename_dict1, inplace=True)
        if highlight:
            ad.obs[groupby].cat.rename_categories(rename_dict1, inplace=True)
    elif annot in (None, False, 'none'):
        kwargs['title'] = ''
        kwargs['legend_loc'] = None
    else:
        adata.obs[groupby].cat.rename_categories(rename_dict2, inplace=True)
        if highlight:
            ad.obs[groupby].cat.rename_categories(rename_dict2, inplace=True)

    try:
        ax = plot_scatter(ad, basis=basis, color=color, size=marker_size, **kwargs)
    finally:
        if annot == 'full':
            adata.obs[groupby].cat.rename_categories(restore_dict1, inplace=True)
        elif annot not in (None, False, 'none'):
            adata.obs[groupby].cat.rename_categories(restore_dict2, inplace=True)
    if annot not in (None, False, 'none'):
        centroids = pseudo_bulk(adata, groupby, use_rep=f'X_{basis}', FUN=np.median).T
        fontsize = kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else 9
        texts = [ax.text(x=row[xi], y=row[yi], s=f'{i:d}', fontsize=fontsize-1.5, fontweight='bold') for i, row in centroids.reset_index(drop=True).iterrows() if row[0].astype(str) != 'nan']
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, text_from_points=False, autoalign=False)
        version_scale = 1 if sc.__version__.startswith('1.4') else 1
        ax.legend(
            loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=int(np.ceil(len(categories)/int(figsize[1]*3)/8*fontsize)),
            fontsize=fontsize*1.1, markerscale=3*(fontsize/11)*version_scale, handletextpad=-1.9, labelspacing=9/fontsize
        )
    if save:
        plt.savefig(fname=save, dpi=savedpi, bbox_inches='tight', pad_inches=0.1)
    if 'ax' not in kwargs.keys():
        return ax


def highlight(adata, basis, groupby, groups=None, color=None, prefix_dict=None, hide_rest=False, figsize=(4, 4), hide_color='#F0F0F0', **kwargs):
    set_figsize(figsize)
    old_obs = adata.obs.copy()
    if isinstance(prefix_dict, dict):
        groups = {k: list(adata.obs[groupby].cat.categories[adata.obs[groupby].cat.categories.str.startswith(v)]) for k, v in prefix_dict.items()}
    if groups is None:
        groups = list(adata.obs[groupby].cat.categories)
    if isinstance(groups, (list, tuple)):
        new_obs = pd.get_dummies(adata.obs[groupby])[groups].astype(int)
        if color is not None:
            if color in adata.obs.columns:
                new_obs = new_obs * adata.obs[color].values.astype(float)[:, np.newaxis]
            elif color in adata.raw.var_names:
                new_obs = new_obs * adata.raw[:, color].X.toarray()
        adata.obs = new_obs
        try:
            kwargs['color_map'] = 'Reds'
            axs = plot_scatter(adata, basis=basis, color=groups, legend_loc=None, show=False, **kwargs)
            for i, ax in enumerate(axs):
                ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
                ax.set_xlabel('')
                ax.set_ylabel('')
                plt.gcf().axes[-(i+1)].remove()

        finally:
            adata.obs = old_obs
            clear_colors(adata)
    elif isinstance(groups, dict):
        new_obs = adata.obs[[groupby]].copy()
        for grp_name, grp in groups.items():
            new_obs[grp_name] = new_obs[groupby].astype(str)
            new_obs.loc[~new_obs[groupby].isin(grp), grp_name] = 'others'
            new_obs[grp_name] = new_obs[grp_name].astype('category').cat.reorder_categories(['others'] + grp)
            adata.uns[f'{grp_name}_colors'] = make_palette(len(grp)+1, kwargs.get('palette', None), hide_first=True, hide_color=hide_color)
        adata.obs = new_obs
        try:
            if 'palette' in kwargs:
                del kwargs['palette']
            kwargs['title'] = [f'{groupby}: {grp_name}' for grp_name in groups.keys()]
            plot_scatter(adata, basis=basis, color=list(groups.keys()), **kwargs)
        finally:
            adata.obs = old_obs
            clear_colors(adata)


def plot_genes(adata, genes, basis='umap', gene_symbols=None, title_func=None, color_map=None, figsize=(2,2), xlim=None, ylim=None, save=None, **kwargs):
    if color_map is None:
        color_map = expression_colormap(0.01)
    var_df = adata.var if adata.raw is None else adata.raw.var
    var_names = var_df[gene_symbols].values if gene_symbols in var_df.columns else var_df.index
    not_found = [g for g in genes if g not in var_names]
    found = [g for g in genes if g in var_names]
    n = len(found)
    sc_warn(f'{n} genes found')
    sc_warn(f'{",".join(not_found)} not found')
    if title_func:
        kwargs['title'] = list(map(title_func, found))
    set_figsize(figsize)
    ncols = kwargs.pop('ncols', None) or int(24/figsize[0])
    axs = plot_scatter(
        adata, basis=basis, color=found, color_map=color_map,
        ncols=ncols, wspace=0, hspace=0.2,
        show=False, **kwargs
    )
    if n < 2:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(xlim)
        plt.gcf().axes[-(i+1)].remove()
    if save:
        plt.savefig(fname=save, bbox_inches='tight', pad_inches=0.1)


def plot_markers(
    adata: anndata.AnnData,
    groupby: str,
    mks: pd.DataFrame,
    n_genes: int = 5,
    kind: str = 'dotplot',
    remove_genes: list = [],
    **kwargs
):
    df = mks.reset_index()[['index', 'top_frac_group']].rename(columns={'index': 'gene', 'top_frac_group': 'cluster'})
    var_tb = adata.raw.var if kwargs.get('use_raw', None) == True or adata.raw else adata.var
    remove_gene_set = set()
    for g_cat in remove_genes:
        if g_cat in var_tb.columns:
            remove_gene_set |= set(var_tb.index[var_tb[g_cat].values])
    df = df[~df.gene.isin(list(remove_gene_set))].copy()
    df1 = df.groupby('cluster').head(n_genes)
    mks_dict = defaultdict(list)
    for c, g in zip(df1.cluster, df1.gene):
        mks_dict[c].append(g)
    func = getattr(sc.pl, kind)
    if sc.__version__.startswith('1.4'):
        return func(adata, df1.gene.to_list(), groupby=groupby, **kwargs)
    else:
        return func(adata, mks_dict, groupby=groupby, **kwargs)


def plot_diffexp(
    adata,
    basis='umap',
    key='rank_genes_groups',
    top_n=4,
    extra_genes=None,
    figsize1=(4,4),
    figsize2=(2.5, 2.5),
    dotsize=None,
    dotplot=True,
    **kwargs
):
    grouping = adata.uns[key]['params']['groupby']
    de_tbl = extract_de_table(adata.uns[key])
    de_tbl = de_tbl.loc[de_tbl.genes.astype(str) != 'nan', :]
    de_genes = list(de_tbl.groupby('cluster').head(top_n)['genes'].values)
    de_clusters = list(de_tbl.groupby('cluster').head(top_n)['cluster'].astype(str).values)
    if extra_genes:
        de_genes.extend(extra_genes)
        de_clusters.extend(['known'] * len(extra_genes))

    rcParams.update({'figure.figsize': figsize1})

    if dotplot:
        sc.pl.dotplot(adata, var_names=de_genes, groupby=grouping, show=False)

    expr_cmap = expression_colormap(0.01)
    rcParams.update({'figure.figsize':figsize2})
    plot_scatter(
        adata,
        basis=basis,
        color=de_genes,
        color_map=expr_cmap,
        use_raw=True,
        size=dotsize,
        title=[f'{c}, {g}' for c, g in zip(de_clusters, de_genes)],
        show=False,
        **kwargs,
    )

    rcParams.update({'figure.figsize':figsize1})
    plot_embedding(
        adata, basis=basis, groupby=grouping, size=dotsize, show=False)
