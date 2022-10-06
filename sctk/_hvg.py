"""
scanpy hvg
"""

import numpy as np
import pandas as pd
import scanpy as sc
from ._utils import sc_warn


def hvg(
    adata,
    mean_limits=(0.0125, 3),
    disp_limits=(0.5, float("inf")),
    subset=False,
    by_batch=None,
    n_hvg=None,
    **kwargs,
):
    """
    Wrapper function for sc.highly_variable_genes(), mainly to support searching
    by batch and pooling.
    """

    # Check for n_top_genes beeing greater than the total genes

    if "n_top_genes" in kwargs and kwargs["n_top_genes"] is not None:
        kwargs["n_top_genes"] = min(adata.n_vars, kwargs["n_top_genes"])

    if by_batch and isinstance(by_batch, (list, tuple)) and by_batch[0]:
        kwargs["batch_key"] = by_batch[0]

    sc.pp.highly_variable_genes(
        adata,
        min_disp=disp_limits[0],
        max_disp=disp_limits[1],
        min_mean=mean_limits[0],
        max_mean=mean_limits[1],
        inplace=True,
        **kwargs,
    )

    if by_batch and isinstance(by_batch, (list, tuple)) and by_batch[0]:
        batch_name = by_batch[0]
        min_n = by_batch[1]
        k_hvg = adata.var.highly_variable_nbatches
        if n_hvg is not None and isinstance(n_hvg, int):
            n_cum = k_hvg.value_counts().sort_index(ascending=False).cumsum()
            if np.all(n_cum < n_hvg):
                min_n = 1
            else:
                min_n = n_cum[n_cum > n_hvg].index[0]
        sc_warn(f"n_hvg = {(k_hvg >= min_n).sum()} found in at least {min_n} batches")
        if subset:
            adata = adata[:, k_hvg.values >= min_n]
        else:
            adata.var["highly_variable"] = k_hvg >= min_n
