"""
scanpy diffexp
"""

import numpy as np
import pandas as pd
import scanpy as sc
from ._utils import sc_warn


def diffexp(
    adata,
    use_raw=True,
    n_genes=None,
    key_added=None,
    logreg_param=None,
    filter_params=None,
    save=None,
    **kwargs,
):
    """
    Wrapper function for sc.tl.rank_genes_groups.
    """
    parameters = ["min_in_group_fraction", "max_out_group_fraction", "min_fold_change"]
    if isinstance(filter_params, (tuple, list)) and len(filter_params) == 3:
        filter_params = {parameters[i]: filter_params[i] for i in range(3)}
    elif isinstance(filter_params, dict) and np.all(
        pd.Series(list(filter_params.keys())).isin(parameters)
    ):
        pass
    else:
        if filter_params is not None:
            sc_warn("Unsupported data format for `filter_params`, reset to None")
        filter_params = None

    if adata.raw is None:
        use_raw = False

    if n_genes is None:
        n_genes = adata.raw.shape[1] if use_raw else adata.shape[1]

    if logreg_param and isinstance(logreg_param, dict):
        for key, val in logreg_param.items():
            kwargs[key] = val

    key_added = f"rank_genes_groups_{key_added}" if key_added else "rank_genes_groups"

    sc.tl.rank_genes_groups(adata, use_raw=use_raw, n_genes=n_genes, key_added=key_added, **kwargs)

    de_tbl = extract_de_table(adata.uns[key_added])

    if isinstance(filter_params, dict):
        sc.tl.filter_rank_genes_groups(
            adata,
            key=key_added,
            key_added=key_added + "_filtered",
            use_raw=use_raw,
            **filter_params,
        )
        de_tbl = extract_de_table(adata.uns[key_added + "_filtered"])

    if save:
        de_tbl.to_csv(save, sep="\t", header=True, index=False)

    return de_tbl


def diffexp_paired(adata, groupby, pair, **kwargs):
    """
    Restrict DE to between a pair of clusters, return both up and down genes
    """
    test, ref = pair
    de_key = f"de.{test}-{ref}"
    up_de = diffexp(
        adata,
        key_added=de_key,
        groupby=groupby,
        groups=[test],
        reference=ref,
        **kwargs,
    )
    ref, test = pair
    de_key = f"de.{test}-{ref}"
    down_de = diffexp(
        adata,
        key_added=de_key,
        groupby=groupby,
        groups=[test],
        reference=ref,
        **kwargs,
    )
    return up_de, down_de


def extract_de_table(de_dict):
    """
    Extract DE table from adata.uns
    """
    if de_dict["params"]["method"] == "logreg":
        requested_fields = ("scores",)
    else:
        requested_fields = (
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
        )
    gene_df = _recarray_to_dataframe(de_dict["names"], "genes")[["cluster", "rank", "genes"]]
    gene_df["ref"] = de_dict["params"]["reference"]
    gene_df = gene_df[["cluster", "ref", "rank", "genes"]]
    de_df = pd.DataFrame(
        {
            field: _recarray_to_dataframe(de_dict[field], field)[field]
            for field in requested_fields
            if field in de_dict
        }
    )
    de_tbl = gene_df.merge(de_df, left_index=True, right_index=True)
    de_tbl = de_tbl.loc[de_tbl.genes.astype(str) != "nan", :]
    return de_tbl


def _recarray_to_dataframe(array, field_name):
    return (
        pd.DataFrame(array)
        .reset_index()
        .rename(columns={"index": "rank"})
        .melt(id_vars="rank", var_name="cluster", value_name=field_name)
    )
