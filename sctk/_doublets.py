"""
Remove doublets
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
import pandas as pd
import scanpy as sc
import anndata
from ._utils import lognorm_to_counts, _find_rep


def _calculate_z(x, upper_mad_only=True):
    med = np.median(x)
    if upper_mad_only:
        mad = np.median(x[x > med] - med) * 1.4826
    else:
        mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad


def _test_outlier(x, upper_mad_only=True):
    from statsmodels.stats.multitest import multipletests

    z = _calculate_z(x, upper_mad_only=upper_mad_only)
    pvals = 1 - norm.cdf(z, loc=0, scale=1)
    bh_pvals = multipletests(pvals, method="fdr_bh")[1]
    return pvals, bh_pvals, z


def run_scrublet(
    adata,
    resolution_function=None,
    use_rep="X",
    reverse_lognorm=False,
    inplace=True,
    ignore_error=True,
):
    if resolution_function is None:
        resolution_function = lambda x: np.maximum(np.maximum(np.log10(x) - 1, 0) ** 2, 0.1)

    old_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 1

    use_calculated = False
    X = _find_rep(adata, use_rep=use_rep, return_var_names=False)

    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    if reverse_lognorm:
        X = lognorm_to_counts(X)

    import scrublet as scr

    try:
        scrub = scr.Scrublet(X)
        ds, _ = scrub.scrub_doublets(verbose=False)
        ds_z = _calculate_z(ds)

        adata_copy = anndata.AnnData(X=X, obs=adata.obs)
        sc.pp.filter_genes(adata_copy, min_cells=3)
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)
        sc.pp.highly_variable_genes(
            adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True
        )
        sc.pp.scale(adata_copy, zero_center=False)
        sc.pp.pca(adata_copy, svd_solver="arpack", zero_center=False)
        sc.pp.neighbors(adata_copy, n_pcs=30)
        sc.tl.leiden(adata_copy, resolution=1)
        for clst in np.unique(adata_copy.obs["leiden"]):
            clst_size = sum(adata_copy.obs["leiden"] == clst)
            sc.tl.leiden(
                adata_copy,
                restrict_to=("leiden", [clst]),
                resolution=resolution_function(clst_size),
                key_added="leiden_R",
            )
            adata_copy.obs["leiden"] = adata_copy.obs["leiden_R"]
        clst_meds = []
        for clst in np.unique(adata_copy.obs["leiden"]):
            k = adata_copy.obs["leiden"] == clst
            clst_med = np.median(ds[k])
            adata_copy.obs.loc[k, "cluster_scrublet_score"] = clst_med
            clst_meds.append(clst_med)
        clst_meds = np.array(clst_meds)
        pvals, bh_pvals, z_scores = _test_outlier(clst_meds)
        for i, clst in enumerate(np.unique(adata_copy.obs["leiden"])):
            k = adata_copy.obs["leiden"] == clst
            adata_copy.obs.loc[k, "pval"] = pvals[i]
            adata_copy.obs.loc[k, "bh_pval"] = bh_pvals[i]
        sc.settings.verbosity = old_verbosity

        df = adata_copy.obs[["cluster_scrublet_score", "bh_pval"]].copy()
        del adata_copy

        df["scrublet_score"] = ds
        df["scrublet_score_z"] = ds_z
        df["scrublet_done"] = True

    except ValueError:
        if ignore_error:
            n = adata.n_obs
            ones = np.ones(n)
            zeros = np.zeros(n)
            df = pd.DataFrame(
                {
                    "cluster_scrublet_score": zeros,
                    "bh_pval": ones,
                    "scrublet_score": zeros,
                    "scrublet_score_z": zeros,
                },
                index=adata.obs_names,
            )
        df["scrublet_done"] = False

    if inplace:
        adata.obs["scrublet_score"] = df["scrublet_score"].values
        adata.obs["scrublet_score_z"] = df["scrublet_score_z"].values
        adata.obs["cluster_scrublet_score"] = df["cluster_scrublet_score"].values
        adata.obs["bh_pval"] = df["bh_pval"].values
        adata.obs["scrublet_done"] = df["scrublet_done"].values
    else:
        return df
