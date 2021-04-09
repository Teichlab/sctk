from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import anndata
import scanpy as sc


def calc_marker_stats(ad, groupby, genes=None, use_rep='raw', inplace=False, partial=False):
    if ad.obs[groupby].dtype.name != 'category':
        raise ValueError('"%s" is not categorical' % groupby)
    n_grp = ad.obs[groupby].cat.categories.size
    if n_grp < 2:
        raise ValueError('"%s" must contain at least 2 categories' % groupby)
    if use_rep == 'raw':
        X = ad.raw.X
        var_names = ad.raw.var_names.values
    else:
        X = ad.X
        var_names = ad.var_names.values
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    if genes:
        v_idx = var_names.isin(genes)
        X = X[:, v_idx]
        var_names = var_names[v_idx]

    X = normalize(X, norm='max', axis=0)
    k_nonzero = X.sum(axis=0).A1 > 0
    X = X[:, np.where(k_nonzero)[0]]
    var_names = var_names[k_nonzero]

    n_var = var_names.size
    x = np.arange(n_var)

    grp_indices = {k: g.index.values for k, g in ad.obs.reset_index().groupby(groupby, sort=False)}
    
    frac_df = pd.DataFrame({k: (X[idx, :]>0).mean(axis=0).A1 for k, idx in grp_indices.items()}, index=var_names)
    mean_df = pd.DataFrame({k: X[idx, :].mean(axis=0).A1 for k, idx in grp_indices.items()}, index=var_names)

    if partial:
        stats_df = None
    else:
        frac_order = np.apply_along_axis(np.argsort, axis=1, arr=frac_df.values)
        y1 = frac_order[:, n_grp-1]
        y2 = frac_order[:, n_grp-2]
        y3 = frac_order[:, n_grp-3] if n_grp > 2 else y2
        top_frac_grps = frac_df.columns.values[y1]
        top_fracs = frac_df.values[x, y1]
        frac_diffs = top_fracs - frac_df.values[x, y2]
        max_frac_diffs = top_fracs - frac_df.values[x, y3]
 
        mean_order = np.apply_along_axis(np.argsort, axis=1, arr=mean_df.values)
        y1 = mean_order[:, n_grp-1]
        y2 = mean_order[:, n_grp-2]
        y3 = mean_order[:, n_grp-3] if n_grp > 2 else y2
        top_mean_grps = mean_df.columns.values[y1]
        top_means = mean_df.values[x, y1]
        mean_diffs = top_means - mean_df.values[x, y2]
        max_mean_diffs = top_means - mean_df.values[x, y3]

        stats_df = pd.DataFrame({
            'top_frac_group': top_frac_grps, 'top_frac': top_fracs, 'frac_diff': frac_diffs, 'max_frac_diff': max_frac_diffs,
            'top_mean_group': top_mean_grps, 'top_mean': top_means, 'mean_diff': mean_diffs, 'max_mean_diff': max_mean_diffs
        }, index=var_names)
        stats_df['top_frac_group'] = stats_df['top_frac_group'].astype('category')
        stats_df['top_frac_group'].cat.reorder_categories(list(ad.obs[groupby].cat.categories), inplace=True)

    if inplace:
        if use_rep == 'raw':
            ad.raw.varm[f'frac_{groupby}'] = frac_df
            ad.raw.varm[f'mean_{groupby}'] = mean_df
            if not partial:
                ad.raw.var = pd.concat([ad.raw.var, stats_df], axis=1)
        else:
            ad.varm[f'frac_{groupby}'] = frac_df
            ad.varm[f'mean_{groupby}'] = mean_df
            if not partial:
                ad.var = pd.concat([ad.raw.var, stats_df], axis=1)
    else:
        return frac_df, mean_df, stats_df


def filter_marker_stats(data, use_rep='raw', min_frac_diff=0.1, min_mean_diff=0.1, max_next_frac=0.9, max_next_mean=0.95, strict=False, how='or'):
    columns = ['top_frac_group', 'top_frac', 'frac_diff', 'max_frac_diff', 'top_mean_group', 'top_mean', 'mean_diff', 'max_mean_diff']
    if isinstance(data, anndata.AnnData):
        stats_df = data.raw.var[columns] if use_rep == 'raw' else data.var[columns]
    elif isinstance(data, pd.DataFrame):
        stats_df = data[columns]
    else:
        raise ValueError('Invalid input, must be an AnnData or DataFrame')
    frac_diff = stats_df.frac_diff if strict else stats_df.max_frac_diff
    mean_diff = stats_df.mean_diff if strict else stats_df.max_mean_diff
    same_group = stats_df.top_frac_group == stats_df.top_mean_group
    meet_frac_requirement = (frac_diff >= min_frac_diff) & (stats_df.top_frac - frac_diff <= max_next_frac)
    meet_mean_requirement = (mean_diff >= min_mean_diff) & (stats_df.top_mean - mean_diff <= max_next_mean)
    if how == 'or':
        filtered = stats_df.loc[same_group & (meet_frac_requirement | meet_mean_requirement)]
    else:
        filtered = stats_df.loc[same_group & (meet_frac_requirement & meet_mean_requirement)]
    if strict:
        filtered = filtered.sort_values(['top_frac_group', 'mean_diff', 'frac_diff'], ascending=[True, False, False])
    else:
        filtered = filtered.sort_values(['top_frac_group', 'mean_diff', 'frac_diff'], ascending=[True, False, False])
    filtered['top_frac_group'] = filtered['top_frac_group'].astype('category')
    filtered['top_frac_group'].cat.reorder_categories(list(stats_df['top_frac_group'].cat.categories), inplace=True)
    return filtered


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
