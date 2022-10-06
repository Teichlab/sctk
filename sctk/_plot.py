"""
Plotting related functions
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, to_hex
from matplotlib.lines import Line2D
import anndata
import scanpy as sc
from ._diffexp import extract_de_table
from ._utils import pseudo_bulk, summarise_expression_by_group, sc_warn, cross_table

if sc.__version__.startswith("1.4"):
    from scanpy.plotting._tools.scatterplots import plot_scatter
else:
    plot_scatter = sc.pl.embedding

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def expression_colormap(background_level=0.01):
    """Returns a nice color map for highlighting gene expression"""
    background_nbin = int(100 * background_level)
    reds = plt.cm.Reds(np.linspace(0, 1, 100 - background_nbin))
    greys = plt.cm.Greys_r(np.linspace(0.7, 0.8, background_nbin))
    palette = np.vstack([greys, reds])
    return LinearSegmentedColormap.from_list("expression", palette)


def make_palette(n, cmap=None, hide_first=False, hide_last=False, hide_color="#E9E9E910"):
    """Returns a color palette with specified number of colors"""
    i = int(hide_first)
    j = int(hide_last)
    if cmap is None:
        if sc.__version__.startswith("1.4"):
            sc_default_10 = rcParams["axes.prop_cycle"].by_key()["color"]
            sc_default_26 = sc.plotting.palettes.default_26
            sc_default_64 = sc.plotting.palettes.default_64
            palette = (
                sc_default_10[0 : (n - i - j)]
                if n <= 10 + i + j
                else sc_default_10 + sc_default_26[0 : (n - i - j - 10)]
                if n <= 36 + i + j
                else sc_default_10 + sc_default_26 + sc_default_64[0 : (n - i - j)]
                if n <= 100 + i + j
                else ["grey"] * n
            )
        else:
            sc_default_20 = sc.plotting.palettes.default_20
            sc_default_28 = sc.plotting.palettes.default_28
            sc_default_102 = sc.plotting.palettes.default_102
            palette = (
                sc_default_20[0 : (n - i - j)]
                if n <= 20 + i + j
                else sc_default_20 + sc_default_28[0 : (n - i - j - 20)]
                if n <= 48 + i + j
                else sc_default_102[0 : (n - i - j)]
                if n <= 102 + i + j
                else sc_default_102 + sc_default_102[0 : (n - 102 - i - j)]
                if n <= 204 + i + j
                else ["grey"] * n
            )
    else:
        color_map = plt.get_cmap(cmap)
        palette = [to_hex(color_map(k)) for k in range(n - i - j)]

    if hide_first:
        palette.insert(0, hide_color)
    if hide_last:
        palette.append(hide_color)
    return palette


def clear_colors(ad, slots=None):
    color_slots = [k for k in ad.uns.keys() if k.endswith("_colors")]
    if isinstance(slots, str):
        slots = [slots]
    elif slots is None:
        slots = [s.replace("_colors", "") for s in color_slots]
    if isinstance(slots, (tuple, list)):
        slots = [s for s in slots if f"{s}_colors" in color_slots]
    for s in slots:
        del ad.uns[f"{s}_colors"]


def _is_numeric(x):
    return x.dtype.kind in ("i", "f")


def abline(slope=1, intercept=0, ax=None, log=False, min_x=1):
    """Plot a line from slope and intercept"""
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    if log:
        if slope >= 0:
            x_vals[0] = max(min_x, (min_x - intercept) / slope, x_vals[0])
        else:
            x_vals[0] = max(min_x, x_vals[0])
            x_vals[1] = min((min_x - intercept) / slope, x_vals[1])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, "--", c="k")


def heatmap(
    tbl,
    stylize=None,
    cluster=False,
    figsize=(4, 4),
    **kwargs,
):
    """plot a table as a heatmap"""
    if stylize is not None:
        return tbl.style.background_gradient(cmap="viridis", axis=stylize)

    if cluster:
        fig = sn.clustermap(tbl, linewidths=0.01, figsize=figsize, **kwargs)
    else:
        set_figsize(figsize)
        fig = sn.heatmap(tbl, linewidths=0.01, **kwargs)
    return fig


def set_figsize(dim):
    if len(dim) == 2 and (isinstance(dim[0], (int, float)) and isinstance(dim[1], (int, float))):
        rcParams.update({"figure.figsize": dim})
    else:
        raise ValueError(
            f"Invalid {dim} value, must be an iterable of "
            "length two in the form of (width, height)."
        )


def plot_df_heatmap(
    df,
    cmap="viridis",
    title=None,
    figsize=(7, 7),
    rotation=90,
    save=None,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df, cmap=cmap, aspect="auto", **kwargs)
    if 0 < rotation < 90:
        horizontalalignment = "right"
    else:
        horizontalalignment = "center"
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
        plt.savefig(fname=save, bbox_inches="tight", pad_inches=0.1)


def _log_hist(x, bins=50, min_x=None, max_x=None, ax=None):
    x_copy = x.copy()
    if max_x is None:
        max_x = x[np.isfinite(x)].max()
    x_copy[x > max_x] = max_x
    if min_x is None:
        min_x1 = x[x > 0].min()
        min_x = 10 ** (np.log10(min_x1) - (np.log10(max_x) - np.log10(min_x1)) / (bins - 1))
    x_copy[x < min_x] = min_x

    log_bins = np.logspace(np.log10(min_x), np.log10(max_x), bins)
    if ax:
        ax.hist(x_copy, bins=log_bins)
        ax.set_xscale("log")
    else:
        fig, ax = plt.subplots()
        ax.hist(x_copy, bins=log_bins)
        ax.set_xscale("log")
        return ax


def dotplot2(
    adata,
    keys,
    groupby=None,
    min_group_size=0,
    min_presence=0,
    use_raw=None,
    mean_only_expressed=False,
    second_key_dependent_fraction=False,
    coexpression=False,
    vmin=0,
    vmax=1,
    dot_min=None,
    dot_max=None,
    color_map="Reds",
    swap_axis=False,
    legend_loc="right",
    title="",
    title_loc="top",
    title_size=None,
    omit_xlab=False,
    omit_ylab=False,
    xtickslabels=None,
    ytickslabels=None,
    ax=None,
    save=None,
    save_dpi=80,
    return_data=False,
    **kwargs,
):
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, (list, tuple)):
        raise ValueError("keys must be a tuple/list of obs keys or var_names")

    if second_key_dependent_fraction or coexpression:
        if len(keys) != 2:
            raise ValueError("Exactly two keys requied if `second_key_dependent_fraction=True`")

    cmap = plt.get_cmap(color_map)
    if use_raw or (use_raw is None and adata.raw):
        ad = adata.raw
    else:
        ad = adata

    if groupby is None:
        groups = pd.Series(np.ones(adata.n_obs)).astype(str).astype("category")
        n_group = 1
    elif groupby in adata.obs.columns:
        groups = adata.obs[groupby]
        if groups.dtype.name != "category":
            groups = groups.astype("category")
    else:
        raise ValueError(f"{groupby} not found")
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
            if sc.__version__.startswith("1.4"):
                y[i] = ad[:, key].X
            else:
                y[i] = ad.obs_vector(key)
        else:
            sc_warn(f"{key} not found")
            y[i] = np.zeros(n_obs)

    df = pd.DataFrame(y.T, columns=keys)
    df["group"] = groups.reset_index(drop=True)
    df["group"].cat.set_categories(grouping, inplace=True)
    group_size = df.groupby("group").agg("size")
    if min_group_size:
        df = df.loc[df["group"].isin(group_size.index[group_size.values >= min_group_size]), :]
        df["group"].cat.remove_unused_categories(inplace=True)
    n_group = df["group"].cat.categories.size

    if second_key_dependent_fraction:
        exp_cnt = (
            df.groupby("group")[keys].apply(lambda g: ((g > vmin).sum(axis=1) == 2).sum()).values
        )
        frac = df.groupby("group")[keys].apply(
            lambda g: ((g > vmin).sum(axis=1) == 2).astype(int).mean()
        )
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby("group")[[keys[0]]].apply(lambda g: g.mean(axis=0)).values
        avg1 = (
            df.groupby("group")[[keys[0]]]
            .apply(lambda g: g.sum(axis=0) / (g > vmin).sum(axis=0))
            .fillna(0)
            .values
        )
        avg = avg1 if mean_only_expressed else avg0
        keys = ["|".join(keys)]
        n_key = 1
    elif coexpression:
        exp_cnt = (
            df.groupby("group")[keys].apply(lambda g: ((g > vmin).sum(axis=1) == 2).sum()).values
        )
        frac = df.groupby("group")[keys].apply(
            lambda g: ((g > vmin).sum(axis=1) == 2).astype(int).mean()
        )
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby("group")[keys].apply(lambda g: g.min(axis=1).mean()).values
        avg1 = (
            df.groupby("group")[keys]
            .apply(lambda g: g.min(axis=1).sum() / max(1, (((g > vmin).sum(axis=1) == 2)).sum()))
            .fillna(0)
            .values
        )
        avg = avg1 if mean_only_expressed else avg0
        keys = ["+".join(keys)]
        n_key = 1
    else:
        exp_cnt = df.groupby("group")[keys].apply(lambda g: (g > vmin).sum(axis=0)).values
        frac = df.groupby("group")[keys].apply(lambda g: (g > vmin).mean(axis=0))
        group_label = frac.index.values
        frac = frac.values
        avg0 = df.groupby("group")[keys].apply(lambda g: g.mean(axis=0)).values
        avg1 = (
            df.groupby("group")[keys]
            .apply(lambda g: g.sum(axis=0) / (g > vmin).sum(axis=0))
            .fillna(0)
            .values
        )
        avg = avg1 if mean_only_expressed else avg0

    frac[exp_cnt < min_presence] = 0
    avg[exp_cnt < min_presence] = 0

    data = pd.DataFrame(
        {
            "gene": np.repeat(keys, n_group),
            "group": np.tile(grouping, n_key),
            "frac": frac.T.flatten(),
            "avg": avg.T.flatten(),
        }
    )
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
        frac = (frac - dot_min) / old_range

    frac = frac.T.flatten()
    avg = avg.T.flatten()

    dot_sizes = (frac * 10) ** 2
    normalize = Normalize(vmin=vmin, vmax=vmax)
    dot_colors = cmap(normalize(avg))

    legend_loc = "none" if ax else legend_loc
    fig_width = 0.4 + (n_key if swap_axis else n_group) * 0.25 + 0.25 * int(legend_loc == "right")
    fig_height = 0.4 + (n_group if swap_axis else n_key) * 0.2 + 0.25 * int(legend_loc == "bottom")

    rcParams.update({"figure.figsize": (fig_width, fig_height)})
    if legend_loc == "right":
        fig, axs = plt.subplots(
            ncols=2,
            nrows=1,
            gridspec_kw={
                "width_ratios": [fig_width - 0.25, 0.25],
                "wspace": 0.25 / (n_key if swap_axis else n_group),
            },
        )
    elif legend_loc == "bottom":
        fig, axs = plt.subplots(
            ncols=1,
            nrows=2,
            gridspec_kw={
                "height_ratios": [fig_height - 0.25, 0.25],
                "hspace": 0.25 / (n_group if swap_axis else n_key),
            },
        )
    elif not ax:
        fig, axs = plt.subplots(ncols=1, nrows=1)
        axs = [axs]
    else:
        axs = [ax]
    main = axs[0]

    if legend_loc == "bottom":
        main.xaxis.tick_top()
    if swap_axis:
        main.scatter(
            y=np.tile(range(n_group), n_key),
            x=np.repeat(np.arange(n_key)[::-1], n_group),
            color=dot_colors[::-1],
            s=dot_sizes[::-1],
            cmap=cmap,
            norm=None,
            edgecolor="none",
            **kwargs,
        )
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
        main.scatter(
            x=np.tile(range(n_group), n_key),
            y=np.repeat(np.arange(n_key), n_group),
            color=dot_colors,
            s=dot_sizes,
            cmap=cmap,
            norm=None,
            edgecolor="none",
            **kwargs,
        )
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
        if title_loc == "top":
            main.set_title(title, fontsize=15)
        elif title_loc == "right":
            title_ax = axs[1] if legend_loc == "right" else main
            ylab_position = "left" if legend_loc == "right" else "right"
            ylab_pad = 0 if legend_loc == "right" else 20
            title_ax.yaxis.set_label_position(ylab_position)
            if title_size is None:
                title_size = min(15, 100 * fig_height / len(title))
            title_ax.set_ylabel(title, rotation=270, labelpad=ylab_pad, fontsize=title_size)
    if omit_xlab:
        main.tick_params(axis="x", bottom=False, labelbottom=False)
    if omit_ylab:
        main.tick_params(axis="y", left=False, labelleft=False)

    if legend_loc in ("right", "bottom"):
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
            fracs_values = (fracs_legends - dot_min) / old_range
        else:
            fracs_values = fracs_legends
        size = (fracs_values * 10) ** 2
        # color = [
        #    cmap(normalize(value)) for value in np.repeat(vmin + (vmax - vmin) * 0.8, len(size))
        # ]

        # plot size bar
        size_legend = axs[1]
        labels = ["{:.0%}".format(x) for x in fracs_legends]
        if dot_max < 1:
            labels[-1] = ">=" + labels[-1]

        if legend_loc == "bottom":
            size_legend.scatter(y=np.repeat(0, len(size)), x=range(len(size)), s=size, color="grey")
            size_legend.set_xticks(range(len(size)))
            # size_legend.set_xticklabels(labels, rotation=270)
            size_legend.set_xticklabels(["{:.0%}".format(x) for x in fracs_legends], rotation=270)
            # remove y ticks and labels
            size_legend.tick_params(axis="x", bottom=False, labelbottom=True, pad=-2)
            size_legend.tick_params(axis="y", left=False, labelleft=False)
            xmin, xmax = size_legend.get_xlim()
            size_legend.set_xlim(xmin - ((n_key if swap_axis else n_group) - 1) * 0.75, xmax + 0.5)
        else:
            size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, color="grey")
            size_legend.set_yticks(range(len(size)))
            # size_legend.set_yticklabels(labels)
            size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
            # remove x ticks and labels
            size_legend.tick_params(axis="y", left=False, labelleft=False, labelright=True, pad=-2)
            size_legend.tick_params(axis="x", bottom=False, labelbottom=False)
            ymin, ymax = size_legend.get_ylim()
            size_legend.set_ylim(ymin - ((n_group if swap_axis else n_key) - 1) * 0.75, ymax + 0.5)

        # remove surrounding lines
        size_legend.spines["right"].set_visible(False)
        size_legend.spines["top"].set_visible(False)
        size_legend.spines["left"].set_visible(False)
        size_legend.spines["bottom"].set_visible(False)
        size_legend.grid(False)

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=save_dpi)
        plt.close()

    if ax:
        return fig_width, fig_height

    return main


def dotplot_combined_coexpression(
    ad, genes, groupby, groups=None, merge=False, title="", save=None, save_dpi=200, **kwargs
):
    print(save)
    n_gene = len(genes)
    assert n_gene == 2, "genes must be "

    fig, ax = plt.subplots(ncols=n_gene + 1, gridspec_kw={"wspace": 0 if merge else 0.05})
    fW, fH = 0, 0
    if groups is not None and len(groups) != len(ad.obs[groupby].unique()):
        ad = ad[ad.obs[groupby].isin(groups), :]
    for i, gene in enumerate(genes):
        w, h = dotplot2(
            ad,
            gene,
            groupby=groupby,
            swap_axis=True,
            ax=ax[i],
            omit_xlab=False,
            omit_ylab=i > 0,
            title="",
            **kwargs,
        )
        if merge:
            if i < n_gene - 1:
                ax[i].spines["right"].set_visible(False)
            else:
                ax[i].spines["left"].set_visible(False)
        fW += w
        fH = max(fH, h)
    i += 1
    w, h = dotplot2(
        ad,
        genes,
        groupby=groupby,
        coexpression=True,
        ax=ax[i],
        swap_axis=True,
        title=title,
        title_loc="right",
        omit_xlab=False,
        omit_ylab=True,
        **kwargs,
    )
    fW += w
    fH = max(fH, h)

    fig.set_figwidth(fW)
    fig.set_figheight(fH)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=save_dpi)
        plt.close()


def dotplot3(adata, groupby, use_rep="X", order_gene_by="frac", order_group_by="avg", **kwargs):
    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import minmax_scale
    from seriate import seriate

    if order_gene_by not in ("frac", "avg", None):
        raise ValueError("`order_gene_by` must be one of 'frac', 'avg' or None")
    if order_group_by not in ("frac", "avg", None):
        raise ValueError("`order_group_by` must be one of 'frac', 'avg' or None")

    dotplot_data = summarise_expression_by_group(adata, groupby, use_rep=use_rep)

    if order_gene_by is not None:
        gene_order_df = pd.pivot(dotplot_data, "gene", "group", order_gene_by)
        gene_order = seriate(
            pdist(gene_order_df.values), approximation_multiplier=10000, timeout=10
        )
        ordered_genes = gene_order_df.index[gene_order].tolist()
    else:
        ordered_genes = gene_order_df.index.tolist()

    if order_group_by is not None:
        group_order_df = pd.pivot(dotplot_data, "gene", "group", order_group_by).loc[ordered_genes]
        group_order_df = pd.DataFrame(
            minmax_scale(group_order_df.values),
            index=group_order_df.index,
            columns=group_order_df.columns,
        )
        group_pos = np.zeros(group_order_df.shape[1], dtype=np.int32)
        for i in range(gene_order_df.shape[1]):
            x = group_order_df.iloc[:, i]
            good_indices = np.where(x >= 0.5)[0][0]
            if good_indices.size == 0:
                good_indices = np.where(x == x.max())[0]
            group_pos[i] = good_indices.mean()
        group_order = np.argsort(group_pos)
        ordered_groups = gene_order_df.columns[group_order].tolist()

    old_group_order = adata.obs[groupby].cat.categories.copy()
    try:
        if order_group_by is not None:
            adata.obs[groupby].cat.reorder_categories(ordered_groups, inplace=True)
        sc.pl.dotplot(adata, ordered_genes, groupby=groupby, **kwargs)
    finally:
        adata.obs[groupby].cat.reorder_categories(old_group_order, inplace=True)


def plot_qc_violin(
    adata,
    metrics=None,
    groupby=None,
    one_per_line=False,
    rotation=0,
    figsize=(4, 3),
    return_fig=False,
    **kwargs,
):
    kwargs["linewidth"] = kwargs.get("linewidth", 0.2)
    kwargs["width"] = kwargs.get("width", 1)
    kwargs["scale"] = kwargs.get("scale", "count")
    obs_df = adata.obs.copy()
    if "log1p_n_counts" not in obs_df.columns and "n_counts" in obs_df.columns:
        obs_df["log1p_n_counts"] = np.log1p(obs_df["n_counts"])
    if "log1p_n_genes" not in obs_df.columns and "n_genes" in obs_df.columns:
        obs_df["log1p_n_genes"] = np.log1p(obs_df["n_genes"])
    qc_metrics = ["log1p_n_counts", "log1p_n_genes"] + obs_df.columns[
        obs_df.columns.str.startswith("percent_")
    ].to_list()
    if "scrublet_score_z" in obs_df.columns:
        qc_metrics.append("scrublet_score_z")
    if isinstance(metrics, (tuple, list)):
        qc_metrics = metrics
    for qmt in qc_metrics:
        if qmt not in obs_df.columns:
            raise ValueError(f"{qmt} not found.")
    if groupby is None:
        df0 = obs_df[qc_metrics]
        groupby = "__group__"
        df0[groupby] = "all"
    else:
        df0 = obs_df[[groupby] + qc_metrics]
    df1 = pd.melt(
        df0, id_vars=[groupby], value_vars=qc_metrics, var_name="metric", value_name="value"
    )

    g = sn.catplot(
        data=df1,
        x=groupby,
        y="value",
        row="metric" if one_per_line else None,
        col="metric" if not one_per_line else None,
        sharex=one_per_line,
        sharey=False,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="violin",
        inner=None,
        palette=adata.uns.get(
            f"{groupby}_colors", make_palette(adata.obs[groupby].cat.categories.size)
        ),
        **kwargs,
    )
    g.set_xticklabels(rotation=rotation)
    for i, qmt in enumerate(qc_metrics):
        g.facet_axis(i if one_per_line else 0, 0 if one_per_line else i).set_ylabel(qmt)
        if groupby == "__group__":
            g.facet_axis(
                i if one_per_line else 0, 0 if one_per_line else i
            ).axes.get_xaxis().set_visible(False)
        else:
            if one_per_line:
                g.facet_axis(i, 0).set_title("")
                if i < len(qc_metrics) - 1:
                    g.facet_axis(i, 0).axes.get_xaxis().set_visible(False)
    if one_per_line:
        g.fig.subplots_adjust(hspace=0.05)
    if return_fig:
        return g.fig


def plot_qc_scatter(
    adata,
    metric_pairs=None,
    color_by=None,
    show_group_size=False,
    use_hexbin=False,
    figsize=(3, 3),
    wspace=0.25,
    axs=None,
    return_fig=False,
    **kwargs,
):
    obs_df = adata.obs.copy()
    n_cell = obs_df.shape[0]
    if metric_pairs is None:
        if "n_counts" not in obs_df.columns:
            raise ValueError("required metric `n_counts` missing from `obs`")
        metric_pairs = []
        if "n_genes" in obs_df.columns:
            metric_pairs.append(("log1p_n_counts", "log1p_n_genes"))
        for pct_m in (
            "percent_mito",
            "percent_ribo",
            "percent_top50",
            "percent_soup",
            "percent_spliced",
        ):
            if pct_m in obs_df.columns:
                metric_pairs.append(("log1p_n_counts", pct_m))
        if "percent_mito" in obs_df.columns and "percent_ribo" in obs_df.columns:
            metric_pairs.append(("log(percent_mito)", "percent_ribo"))
        if "percent_soup" in obs_df.columns and "percent_spliced" in obs_df.columns:
            metric_pairs.append(("log(percent_soup)", "percent_spliced"))
    n_pair = len(metric_pairs)

    def _logged_metric(m):
        return m.startswith("log1p_") or (m.startswith("log(") and m.endswith(")"))

    def _unlog(m):
        if m.startswith("log1p_"):
            return m[6:]
        if m.startswith("log(") and m.endswith(")"):
            return m[4:-1]
        return m

    scales = [
        (
            "log" if _logged_metric(mp[0]) else "linear",
            "log" if _logged_metric(mp[1]) else "linear",
        )
        for mp in metric_pairs
    ]
    metric_pairs = [(_unlog(mp[0]), _unlog(mp[1])) for mp in metric_pairs]

    old_figsize = rcParams.get("figure.figsize")
    rcParams.update({"figure.figsize": (n_pair * figsize[0], figsize[1])})
    try:
        if axs is None:
            fig, axs = plt.subplots(ncols=n_pair, nrows=1, gridspec_kw={"wspace": wspace})
            if n_pair == 1:
                axs = [axs]
        else:
            fig = None
        point_size = max(kwargs.pop("s", 1 / n_cell), 1e-5)
        kwargs["rasterized"] = kwargs.get("rasterized", True)
        if color_by in obs_df.columns:
            if obs_df[color_by].dtype.kind in ("i", "f"):
                color_var = obs_df[color_by].values
                # vmin = kwargs.pop("vmin", color_var.min())
                # vmax = kwargs.pop("vmax", color_var.max())
                # color_var = (color_var - vmin) / (vmax - vmin)
                for i, mp in enumerate(metric_pairs):
                    m1, m2 = mp
                    panel = axs[i].scatter(
                        x=obs_df[m1],
                        y=obs_df[m2],
                        c=color_var,
                        cmap=kwargs.pop("cmap", "viridis"),
                        s=point_size,
                        **kwargs,
                    )
                    axs[i].set_xlabel(m1)
                    axs[i].set_ylabel(m2)
                    axs[i].set_xscale(scales[i][0])
                    axs[i].set_yscale(scales[i][1])
                if fig is not None:
                    fig.subplots_adjust(right=1 - 0.2 / (i + 1))
                    cbar_ax = fig.add_axes([1 - 0.2 / (i + 1) * 0.75, 0.15, 0.05 / (i + 1), 0.7])
                    fig.colorbar(panel, cax=cbar_ax)
            else:
                if pd.api.types.is_categorical_dtype(obs_df[color_by]):
                    color_var = obs_df[color_by].cat.categories
                else:
                    color_var = obs_df[color_by].astype("category").cat.categories
                n_color = color_var.size
                palette = adata.uns.get(f"{color_by}_colors", make_palette(n_color))
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        label=f"{v}, n={sum(obs_df[color_by] == v)}" if show_group_size else v,
                        color="white",
                        markerfacecolor=c,
                    )
                    for c, v in zip(palette, color_var)
                ]
                alphas = np.repeat(0.1, n_color)
                alphas[0] = 1
                for i, mp in enumerate(metric_pairs):
                    m1, m2 = mp
                    for j, v in enumerate(color_var):
                        k = obs_df[color_by] == v
                        if use_hexbin:
                            cm = LinearSegmentedColormap.from_list(
                                "custom gradient", ["#FFFFFF", palette[j]], N=256
                            )
                            axs[i].hexbin(
                                x=obs_df.loc[k, m1],
                                y=obs_df.loc[k, m2],
                                mincnt=1,
                                cmap=cm,
                                alpha=alphas[j],
                                **kwargs,
                            )
                        else:
                            axs[i].scatter(
                                x=obs_df.loc[k, m1],
                                y=obs_df.loc[k, m2],
                                color=palette[j],
                                s=point_size,
                                **kwargs,
                            )
                    axs[i].set_xlabel(m1)
                    axs[i].set_ylabel(m2)
                    axs[i].set_xscale(scales[i][0])
                    axs[i].set_yscale(scales[i][1])
                axs[n_pair - 1].legend(
                    handles=legend_elements,
                    loc="center left",
                    bbox_to_anchor=[1, 0, 0, 1],
                    ncol=max(1, len(legend_elements) // 10),
                )
        else:
            for i, mp in enumerate(metric_pairs):
                m1, m2 = mp
                if use_hexbin:
                    axs[i].hexbin(
                        x=obs_df[m1],
                        y=obs_df[m2],
                        mincnt=1,
                        xscale=scales[i][0],
                        yscale=scales[i][1],
                        cmap="viridis",
                    )
                else:
                    axs[i].scatter(x=obs_df[m1], y=obs_df[m2], color="k", s=point_size)
                    axs[i].set_xscale(scales[i][0])
                    axs[i].set_yscale(scales[i][1])
                axs[i].set_xlabel(m1)
                axs[i].set_ylabel(m2)
        if return_fig:
            return fig
    finally:
        rcParams.update({"figure.figsize": old_figsize})


def plot_metric_by_rank(
    adata,
    subject="cell",
    metric="n_counts",
    kind="rank",
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
    kwargs["c"] = kwargs.get("c", "black")
    metric_names = {
        "n_counts": "nUMI",
        "n_genes": "nGene",
        "n_cells": "nCell",
        "percent_mito": "fracMT",
        "percent_viral": "fracViral",
    }
    if subject not in ("cell", "gene"):
        print('`subject` must be "cell" or "gene".')
        return
    # if metric not in metric_names:
    #    print('`metric` must be "n_counts", "n_genes", "n_cells" or "percent_mito".')
    #    return

    if metric == "percent_mito" and "percent_mito" not in adata.obs.columns:
        if "mito" in adata.var.columns:
            k_mt = adata.var.mito.astype(bool).values
        else:
            k_mt = adata.var_names.str.startswith("MT-")
        if sum(k_mt) > 0:
            adata.obs["percent_mito"] = (
                np.squeeze(np.asarray(adata.X[:, k_mt].sum(axis=1))) / adata.obs["n_counts"] * 100
            )

    if not ax:
        fig, ax = plt.subplots()

    if kind == "rank":
        order_modifier = -1 if decreasing else 1

        if subject == "cell":
            if metric not in adata.obs.columns:
                if metric == "n_counts":
                    adata.obs["n_counts"] = adata.X.sum(axis=1).A1
                if metric == "n_genes":
                    adata.obs["n_genes"] = (adata.X > 0).sum(axis=1).A1
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
                if metric == "n_counts":
                    adata.var["n_counts"] = adata.X.sum(axis=0).A1
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

        if kwargs["c"] is not None and not isinstance(kwargs["c"], str):
            kwargs_c = kwargs["c"][k]
            del kwargs["c"]
            ax.scatter(x, y, c=kwargs_c, **kwargs)
        else:
            marker = kwargs.get("marker", "-")
            ax.plot(x, y, marker, **kwargs)

        if logy:
            ax.set_yscale("log")
        if swap_axis:
            ax.set_xlabel("Rank")
        else:
            ax.set_ylabel("Rank")

    elif kind == "hist":
        del kwargs["c"]
        if subject == "cell":
            value = adata.obs[metric]
            logbins = np.logspace(
                np.log10(np.min(value[value > 0])), np.log10(np.max(adata.obs[metric])), nbins
            )
            h = ax.hist(value[value > 0], logbins, **kwargs)
            y = h[0]
            x = h[1]
        else:
            value = adata.var[metric]
            logbins = np.logspace(
                np.log10(np.min(value[value > 0])), np.log10(np.max(adata.var[metric])), nbins
            )
            h = ax.hist(value[value > 0], logbins, **kwargs)
            y = h[0]
            x = h[1]

        ax.set_ylabel("Count")

    if hpos is not None:
        ax.hlines(hpos, xmin=x.min(), xmax=x.max(), linewidth=1, colors="red")
    if vpos is not None:
        ax.vlines(vpos, ymin=y.min(), ymax=y.max(), linewidth=1, colors="green")
    if logx:
        ax.set_xscale("log")
    if swap_axis:
        ax.set_ylabel(metric_names.get(metric, metric))
    else:
        ax.set_xlabel(metric_names.get(metric, metric))
    if title:
        ax.set_title(title)
    ax.grid(linestyle="--")
    if "label" in kwargs:
        ax.legend(loc="upper right" if decreasing else "upper left")

    return ax


def plot_embedding(
    adata,
    groupby,
    basis="umap",
    color=None,
    annot=True,
    min_group_size=0,
    greyout_group=None,
    highlight_group=None,
    size=None,
    use_uns_colors=True,
    save=None,
    savedpi=300,
    figsize=(4, 4),
    **kwargs,
):
    set_figsize(figsize)
    if f"X_{basis}" not in adata.obsm.keys():
        raise KeyError(f'"X_{basis}" not found in `adata.obsm`.')
    if min_group_size:
        group_sizes = adata.obs[groupby].value_counts()
        good_groups = group_sizes[group_sizes >= min_group_size].index
        adata = adata[adata.obs[groupby].isin(good_groups)].copy()
    if isinstance(groupby, (list, tuple)):
        groupby = groupby[0]
    if groupby not in adata.obs.columns:
        raise KeyError(f'"{groupby}" not found in `adata.obs`.')
    if adata.obs[groupby].dtype.name != "category":
        if (
            isinstance(adata.obs[groupby][0], (str, bool, np.bool_))
            and adata.obs[groupby].unique().size < 100
        ):
            adata.obs[groupby] = adata.obs[groupby].astype(str).astype("category")
        else:
            sc_warn(f'"{groupby}" is not categorical.')
            plot_scatter(adata, basis=basis, color=groupby, **kwargs)
    categories = list(adata.obs[groupby].cat.categories)
    rename_dict1 = {
        ct: f"{i:^5d} {ct} (n={(adata.obs[groupby]==ct).sum()})" for i, ct in enumerate(categories)
    }
    restore_dict1 = {
        f"{i:^5d} {ct} (n={(adata.obs[groupby]==ct).sum()})": ct for i, ct in enumerate(categories)
    }
    rename_dict2 = {ct: f"{i:^5d} {ct}" for i, ct in enumerate(categories)}
    restore_dict2 = {f"{i:^5d} {ct}": ct for i, ct in enumerate(categories)}

    marker_size = size
    kwargs["show"] = False
    kwargs["save"] = False
    kwargs["frameon"] = kwargs.get("frameon", None)
    kwargs["legend_loc"] = kwargs.get("legend_loc", "right margin")

    if color is not None and color != groupby:
        use_uns_colors = False
    color = groupby if color is None else color
    offset = 0 if "diffmap" in basis else -1
    xi, yi = 1, 2
    if "components" in kwargs:
        xi, yi = kwargs["components"]
    xi += offset
    yi += offset

    if not (use_uns_colors and f"{color}_colors" in adata.uns):
        adata.uns[f"{color}_colors"] = make_palette(
            adata.obs[color].cat.categories.size, kwargs.get("palette", None)
        )
        if greyout_group is not None:
            if isinstance(greyout_group, str):
                greyout_group = [greyout_group]
            greyout_indices = np.where(
                adata.obs[color].cat.categories.isin(greyout_group)
            )[0]
            for nan_i in greyout_indices:
                adata.uns[f"{color}_colors"][nan_i] = "#E0E0E0"

        if highlight_group is not None:
            if isinstance(highlight_group, str):
                highlight_group = [highlight_group]
            nohl_indices = np.where(
                ~adata.obs[color].cat.categories.isin(highlight_group)
            )[0]
            for nohl_i in nohl_indices:
                adata.uns[f"{color}_colors"][nohl_i] = "#E0E0E0"

    if annot == "full":
        adata.obs[groupby].cat.rename_categories(rename_dict1, inplace=True)
    elif annot in (None, False, "none"):
        kwargs["title"] = ""
        kwargs["legend_loc"] = None
    else:
        adata.obs[groupby].cat.rename_categories(rename_dict2, inplace=True)

    try:
        if "ax" in kwargs:
            ax = kwargs["ax"]
            plot_scatter(adata, basis=basis, color=color, size=marker_size, **kwargs)
        else:
            ax = plot_scatter(adata, basis=basis, color=color, size=marker_size, **kwargs)
    finally:
        if annot == "full":
            adata.obs[groupby].cat.rename_categories(restore_dict1, inplace=True)
        elif annot not in (None, False, "none"):
            adata.obs[groupby].cat.rename_categories(restore_dict2, inplace=True)
    if annot not in (None, False, "none"):
        centroids = pseudo_bulk(adata, groupby, use_rep=f"X_{basis}", FUN=np.median).T
        fontsize = kwargs["legend_fontsize"] if "legend_fontsize" in kwargs else 11
        texts = [
            ax.text(x=row[xi], y=row[yi], s=f"{i:d}", fontsize=fontsize - 1.5, fontweight="bold")
            for i, row in centroids.reset_index(drop=True).iterrows()
            if row[0].astype(str) != "nan"
        ]
        from adjustText import adjust_text

        adjust_text(texts, ax=ax, text_from_points=False, autoalign=False)
        version_scale = 1 if sc.__version__.startswith("1.4") else 1
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            ncol=int(np.ceil(len(categories) / int(figsize[1] * 3) / 8 * fontsize)),
            fontsize=fontsize * 1.1,
            markerscale=3 * (fontsize / 11) * version_scale,
            handletextpad=-1.9,
            labelspacing=9 / fontsize,
        )
    if save:
        plt.savefig(fname=save, dpi=savedpi, bbox_inches="tight", pad_inches=0.1)
    if "ax" not in kwargs:
        return ax


def highlight(
    adata,
    basis,
    groupby,
    groups=None,
    color=None,
    prefix_dict=None,
    hide_rest=False,
    figsize=(4, 4),
    hide_color="#F0F0F0",
    **kwargs,
):
    set_figsize(figsize)
    old_obs = adata.obs.copy()
    if isinstance(prefix_dict, dict):
        groups = {
            k: list(
                adata.obs[groupby].cat.categories[
                    adata.obs[groupby].cat.categories.str.startswith(v)
                ]
            )
            for k, v in prefix_dict.items()
        }
    if groups is None:
        groups = list(adata.obs[groupby].cat.categories)
    if isinstance(groups, (list, tuple)):
        new_obs = pd.get_dummies(adata.obs[groupby])[groups].astype(int)
        if color is not None:
            if color in adata.obs.columns:
                new_obs = new_obs * adata.obs[color].values.astype(float)[:, np.newaxis]
            elif color in adata.raw.var_names:
                new_obs = new_obs * adata.raw[:, color].X.toarray()
        for i, grp in enumerate(groups):
            if grp in adata.var_names:
                new_grp = "c_" + grp
                new_obs.rename(columns={grp: new_grp}, inplace=True)
                groups[i] = new_grp
        adata.obs = new_obs
        try:
            kwargs["color_map"] = "Reds"
            axs = plot_scatter(
                adata, basis=basis, color=groups, legend_loc=None, show=False, **kwargs
            )
            for i, ax in enumerate(axs):
                ax.tick_params(which="both", bottom=False, top=False, left=False, right=False)
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.gcf().axes[-(i + 1)].remove()

        finally:
            adata.obs = old_obs
            clear_colors(adata)
    elif isinstance(groups, dict):
        new_obs = adata.obs[[groupby]].copy()
        for grp_name, grp in groups.items():
            new_obs[grp_name] = new_obs[groupby].astype(str)
            new_obs.loc[~new_obs[groupby].isin(grp), grp_name] = "others"
            new_obs[grp_name] = (
                new_obs[grp_name].astype("category").cat.reorder_categories(["others"] + grp)
            )
            adata.uns[f"{grp_name}_colors"] = make_palette(
                len(grp) + 1, kwargs.get("palette", None), hide_first=True, hide_color=hide_color
            )
        adata.obs = new_obs
        try:
            if "palette" in kwargs:
                del kwargs["palette"]
            kwargs["title"] = [f"{groupby}: {grp_name}" for grp_name in groups.keys()]
            plot_scatter(adata, basis=basis, color=list(groups.keys()), **kwargs)
        finally:
            adata.obs = old_obs
            clear_colors(adata)


def plot_composition(adata, composition, groupby, sample):
    """Plot composition as a bar plot

    composition: categorical variable in `obs` of which composition is of interest, e.g. cell type
    groupby    : categorical variable in `obs` of which composition is calculated within
    sample     : categorical variable in `obs` across which composition is averaged
    """
    if not ((composition in adata.obs.columns) and (groupby in adata.obs.columns)):
        raise ValueError
    df = (
        pd.merge(
            adata.obs[[sample, groupby]].drop_duplicates().set_index(sample),
            cross_table(adata, sample, composition, normalise="x"),
            left_index=True,
            right_index=True,
            how="right",
        )
        .groupby(groupby)
        .apply(np.mean)
    ) * 100
    fig, ax = plt.subplots(1, 1)
    cmap = (
        ListedColormap(adata.uns[f"{composition}_colors"])
        if f"{composition}_colors" in adata.uns.keys()
        else None
    )
    df.plot.bar(stacked=True, width=0.8, edgecolor="k", linewidth=0.5, cmap=cmap, ax=ax)
    ax.set_ylim(102, -2)
    ax.set_yticklabels([102, 100, 80, 60, 40, 20, 0])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return df, fig, ax


def plot_genes(
    adata,
    genes,
    basis="umap",
    gene_symbols=None,
    title_func=None,
    color_map=None,
    figsize=(2, 2),
    xlim=None,
    ylim=None,
    use_hexbin=False,
    save=None,
    **kwargs,
):
    if color_map is None:
        color_map = expression_colormap(0.01)
    var_df = adata.var if adata.raw is None else adata.raw.var
    var_names = var_df[gene_symbols].values if gene_symbols in var_df.columns else var_df.index
    not_found = [g for g in genes if g not in var_names]
    found = [g for g in genes if g in var_names]
    n = len(found)
    sc_warn(f"{n} genes found")
    sc_warn(f'{",".join(not_found)} not found')
    ncols = kwargs.pop("ncols", None) or int(24 / figsize[0])

    titles = list(map(title_func, found)) if title_func else found

    if use_hexbin:
        nrows = int(np.ceil(n / ncols))
        ncols = min(ncols, n)
        set_figsize((figsize[0] * ncols, figsize[1] * nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        cm = LinearSegmentedColormap.from_list("custom gradient", ["#EEEEEE", "#FF0000"], N=256)
        for i in range(n):
            ax = axs[i // ncols, i % ncols]
            ax.hexbin(
                x=adata.obsm[f"X_{basis}"][:, 0],
                y=adata.obsm[f"X_{basis}"][:, 1],
                C=adata.obs_vector(found[i]),
                cmap=cm,
                **kwargs,
            )
            ax.set_title(titles[i])
            ax.set_xticks([])
            ax.set_yticks([])
        if save:
            fig.savefig(fname=save, bbox_inches="tight", pad_inches=0.1)
    else:
        set_figsize(figsize)
        kwargs["title"] = titles
        axs = plot_scatter(
            adata,
            basis=basis,
            color=found,
            color_map=color_map,
            ncols=ncols,
            wspace=0,
            hspace=0.2,
            show=False,
            gene_symbols=gene_symbols,
            **kwargs,
        )
        if n < 2:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.tick_params(which="both", bottom=False, top=False, left=False, right=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(xlim)
            plt.gcf().axes[-(i + 1)].remove()
        if save:
            plt.savefig(fname=save, bbox_inches="tight", pad_inches=0.1)


def plot_markers(
    adata: anndata.AnnData,
    groupby: str,
    mks: pd.DataFrame,
    n_genes: int = 5,
    kind: str = "dotplot",
    remove_genes: list = [],
    **kwargs,
):
    df = mks.reset_index()[["index", "top_frac_group"]].rename(
        columns={"index": "gene", "top_frac_group": "cluster"}
    )
    var_tb = adata.raw.var if kwargs.get("use_raw", None) is True or adata.raw else adata.var
    remove_gene_set = set()
    for g_cat in remove_genes:
        if g_cat in var_tb.columns:
            remove_gene_set |= set(var_tb.index[var_tb[g_cat].values])
    df = df[~df.gene.isin(list(remove_gene_set))].copy()
    df1 = df.groupby("cluster").head(n_genes)
    mks_dict = defaultdict(list)
    for c, g in zip(df1.cluster, df1.gene):
        mks_dict[c].append(g)
    func = getattr(sc.pl, kind)
    if sc.__version__.startswith("1.4"):
        return func(adata, df1.gene.to_list(), groupby=groupby, **kwargs)
    else:
        return func(adata, mks_dict, groupby=groupby, **kwargs)


def plot_diffexp(
    adata,
    basis="umap",
    key="rank_genes_groups",
    top_n=4,
    extra_genes=None,
    figsize1=(4, 4),
    figsize2=(2.5, 2.5),
    dotsize=None,
    dotplot=True,
    **kwargs,
):
    grouping = adata.uns[key]["params"]["groupby"]
    de_tbl = extract_de_table(adata.uns[key])
    de_tbl = de_tbl.loc[de_tbl.genes.astype(str) != "nan", :]
    de_genes = list(de_tbl.groupby("cluster").head(top_n)["genes"].values)
    de_clusters = list(de_tbl.groupby("cluster").head(top_n)["cluster"].astype(str).values)
    if extra_genes:
        de_genes.extend(extra_genes)
        de_clusters.extend(["known"] * len(extra_genes))

    rcParams.update({"figure.figsize": figsize1})
    # sc.pl.rank_genes_groups(adata, key=key, show=False)

    if dotplot:
        sc.pl.dotplot(adata, var_names=de_genes, groupby=grouping, show=False)

    expr_cmap = expression_colormap(0.01)
    rcParams.update({"figure.figsize": figsize2})
    plot_scatter(
        adata,
        basis=basis,
        color=de_genes,
        color_map=expr_cmap,
        use_raw=True,
        size=dotsize,
        title=[f"{c}, {g}" for c, g in zip(de_clusters, de_genes)],
        show=False,
        **kwargs,
    )

    rcParams.update({"figure.figsize": figsize1})
    plot_embedding(adata, basis=basis, groupby=grouping, size=dotsize, show=False)


def plot_cellbender_qc(
    ad, raw_nUMI_name="n_counts_raw", nUMI_name="n_counts", title=None, ax=None, overlay_soup=False
):
    if raw_nUMI_name not in ad.obs.columns:
        raise ValueError("Cannot find raw nUMI info in `obs`")

    n_ax = 1
    train_history_found = (
        "test_epoch" in ad.uns.keys()
        and "test_elbo" in ad.uns.keys()
        and "training_elbo_per_epoch" in ad.uns.keys()
    )
    latent_gene_encoding_found = "X_latent_gene_encoding" in ad.obsm.keys()
    if train_history_found:
        n_ax += 1
    if latent_gene_encoding_found:
        n_ax += 1

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=n_ax, gridspec_kw={"wspace": 0.5})

    raw_nUMI = ad.obs[raw_nUMI_name].values
    if nUMI_name not in ad.obs.columns:
        nUMI = ad.X.sum(axis=1).A1
    else:
        nUMI = ad.obs[nUMI_name].values
    k1 = np.argsort(raw_nUMI)[::-1]
    y1 = raw_nUMI[k1]
    y2 = nUMI[k1]
    cell_prob = ad.obs["latent_cell_probability"].values
    y3 = cell_prob[k1]
    y4 = (1 - nUMI / raw_nUMI)[k1]
    x1 = np.arange(ad.n_obs) + 1
    ax[0].plot(y1 + 1, c="black", label="raw")
    ax[0].scatter(x1, y2 + 1, s=0.1, c="blue", label="CB", rasterized=True)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Rank")
    ax[0].legend()
    ax[0].set_ylabel("nUMI")
    ax0b = ax[0].twinx()
    ax0b.scatter(x1, y3, s=0.1, c="red", alpha=0.5, label="p", rasterized=True)
    ax0b.set_ylim(-0.02, 1.02)
    if overlay_soup:
        ax0c = ax[0].twinx()
        ax0c.scatter(x1, y4, s=0.1, c="yellow", alpha=0.5, label="f", rasterized=True)
        ax0c.set_ylim(-0.02, 1.02)
        ax0b.set_ylabel("Cell probability (red) / soup fraction (yellow)")
    else:
        ax0b.set_ylabel("Cell probability", color="red")

    if train_history_found:
        y5 = ad.uns["training_elbo_per_epoch"]
        x2 = ad.uns["test_epoch"]
        y6 = ad.uns["test_elbo"]
        ax[1].plot(y5, c="black", label="train")
        ax[1].scatter(x2, y6, s=2, c="blue", label="test", rasterized=True)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Elbo")
        ax[1].legend()

    if latent_gene_encoding_found:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, svd_solver="arpack")
        pcs = pca.fit_transform(ad.obsm["X_latent_gene_encoding"][cell_prob >= 0.5])
        ax[-1].hexbin(pcs[:, 0], pcs[:, 1], bins="log", cmap="viridis")
        ax[-1].set_xlabel("PC1")
        ax[-1].set_ylabel("PC2")

    if title:
        ax[0].get_figure().suptitle(title, weight="bold")

    if ax is None:
        return fig
