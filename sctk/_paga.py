"""
scanpy paga
"""

import numpy as np
import scanpy as sc
from ._obj_utils import (
    _backup_default_key,
    _delete_backup_key,
    _rename_default_key,
    _set_default_key,
    _restore_default_key,
)

if sc.__version__.startswith("1.4"):
    from scanpy.plotting._tools.scatterplots import plot_scatter
else:
    plot_scatter = sc.pl.embedding


def paga(
    adata,
    use_graph="neighbors",
    key_added=None,
    **kwargs,
):
    """
    Wrapper function for sc.tl.paga, for supporting named slot
    """
    _set_default_key(adata.uns, "neighbors", use_graph)
    _backup_default_key(adata.uns, "paga")

    sc.tl.paga(adata, **kwargs)

    _restore_default_key(adata.uns, "neighbors", use_graph)

    if key_added:
        paga_key = f"paga_{key_added}"
        _rename_default_key(adata.uns, "paga", paga_key)
    else:
        _delete_backup_key(adata.uns, "paga")

    return adata


def plot_paga(
    adata,
    use_key="paga",
    basis=None,
    layout=None,
    init_pos=None,
    legend_loc="on data",
    color=None,
    size=None,
    title=None,
    show=None,
    save=None,
    **kwargs,
):
    """Make PAGA plot"""
    if basis is not None and f"X_{basis}" in adata.obsm.keys():
        ax = plot_scatter(
            adata,
            basis=basis,
            color=color,
            legend_loc=legend_loc,
            size=size,
            title=None,
            save=False,
            show=False,
        )

        grouping = adata.uns[use_key]["groups"]
        categories = list(adata.obs[grouping].cat.categories)
        obsm = adata.obsm[f"X_{basis}"]
        group_pos = np.zeros((len(categories), 2))
        for i, label in enumerate(categories):
            offset = 1 if basis.startswith("diffmap") else 0
            _scatter = obsm[adata.obs[grouping] == label, (0 + offset) : (2 + offset)]
            x_pos, y_pos = np.median(_scatter, axis=0)
            group_pos[i] = [x_pos, y_pos]

        _set_default_key(adata.uns, "paga", use_key)
        kwargs["node_size_scale"] = 0
        kwargs["fontsize"] = 1
        kwargs["pos"] = group_pos
        kwargs["color"] = None
        try:
            sc.pl.paga(
                adata,
                ax=ax,
                title=title,
                show=show,
                save=save,
                **kwargs,
            )
        finally:
            _restore_default_key(adata.uns, "paga", use_key)
    else:
        _set_default_key(adata.uns, "paga", use_key)
        try:
            sc.pl.paga(
                adata, layout=layout, init_pos=init_pos, title=title, show=show, save=save, **kwargs
            )
        finally:
            _restore_default_key(adata.uns, "paga", use_key)
