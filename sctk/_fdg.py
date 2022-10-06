"""
scanpy fdg
"""

import scanpy as sc
from ._obj_utils import (
    _backup_obsm_key,
    _delete_obsm_backup_key,
    _rename_obsm_key,
    write_embedding,
)


def fdg(
    adata,
    use_graph="neighbors",
    layout="fa",
    key_added=None,
    random_state=0,
    export_embedding=None,
    **kwargs,
):
    """
    Wrapper function for sc.tl.draw_graph, for supporting named slot of fdg
    embeddings.
    """
    adj_mat = None
    if use_graph != "neighbors":
        if use_graph not in adata.uns.keys():
            raise KeyError(f'"{use_graph}" is not a valid key of `.uns`.')
        if "connectivities" in adata.uns[use_graph].keys():
            adj_mat = adata.uns[use_graph]["connectivities"]
        elif getattr(adata, "obsp") and f"{use_graph}_connectivities" in adata.obsp.keys():
            adj_mat = adata.obsp[f"{use_graph}_connectivities"]
        else:
            raise KeyError(f'connectivities graph not found for "{use_graph}".')

    _backup_obsm_key(adata, f"X_draw_graph_{layout}")

    sc.tl.draw_graph(
        adata,
        layout=layout,
        random_state=random_state,
        adjacency=adj_mat,
        **kwargs,
    )

    fdg_key = f"X_draw_graph_{layout}"
    if key_added:
        fdg_key = f"X_draw_graph_{layout}_{key_added}"
        _rename_obsm_key(adata, f"X_draw_graph_{layout}", fdg_key)
    else:
        _delete_obsm_backup_key(adata, f"X_draw_graph_{layout}")

    if export_embedding is not None:
        write_embedding(adata, fdg_key, export_embedding, key_added=key_added)
