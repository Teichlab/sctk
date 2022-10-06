"""
scanpy pca
"""

import logging
import scanpy as sc
from ._obj_utils import write_embedding


def pca(adata, key_added=None, remove_genes=None, export_embedding=None, **kwargs):
    """
    Wrapper function for sc.pp.pca, for supporting named slot
    """
    if isinstance(remove_genes, (tuple, list)):
        pass
    elif isinstance(remove_genes, str):
        remove_genes = [remove_genes]
    else:
        remove_genes = []
    if remove_genes:
        adata.var["hvg_full"] = adata.var["highly_variable"].copy()
        for g in remove_genes:
            if g in adata.var.columns and (
                adata.var[g].dtype.kind == "b" or isinstance(adata.var[g][0], bool)
            ):
                adata.var["highly_variable"] = adata.var["highly_variable"] & ~adata.var[g]

    # n_comps may be greater than the number of cells (n_obs), which will
    # produce an error. Additional logic may be required in future for very
    # small gene numbers (adata.n_vars).

    if "n_comps" in kwargs and kwargs["n_comps"] is not None:
        if kwargs["n_comps"] > adata.n_obs:
            logging.warning("n_comps exceeds cell number, resetting to %d", adata.n_obs)
            kwargs["n_comps"] = adata.n_obs

    # omit "svd_solver" to let scanpy choose automatically
    if "svd_solver" in kwargs and kwargs["svd_solver"] == "auto":
        del kwargs["svd_solver"]

    if key_added:
        if "X_pca" in adata.obsm.keys():
            adata.obsm["X_pca_bkup"] = adata.obsm["X_pca"]
        sc.pp.pca(adata, **kwargs)
        pca_key = f"X_pca_{key_added}"
        adata.obsm[pca_key] = adata.obsm["X_pca"]
        del adata.obsm["X_pca"]
        if "X_pca_bkup" in adata.obsm.keys():
            adata.obsm["X_pca"] = adata.obsm["X_pca_bkup"]
            del adata.obsm["X_pca_bkup"]
    else:
        sc.pp.pca(adata, **kwargs)
        pca_key = "X_pca"

    if export_embedding is not None:
        write_embedding(adata, pca_key, export_embedding, key_added=key_added)
