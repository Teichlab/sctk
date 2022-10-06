"""
scanpy diffmap
"""

import scanpy as sc
from ._obj_utils import (
    _set_default_key,
    _restore_default_key,
    _backup_obsm_key,
    _delete_obsm_backup_key,
    _rename_obsm_key,
    write_embedding,
)


def diffmap(
    adata,
    use_graph="neighbors",
    key_added=None,
    export_embedding=None,
    **kwargs,
):
    """
    Wrapper function for sc.tl.diffmap, for supporting named slot
    """
    _set_default_key(adata.uns, "neighbors", use_graph)
    _backup_obsm_key(adata, "X_diffmap")

    sc.tl.diffmap(adata, **kwargs)

    _restore_default_key(adata.uns, "neighbors", use_graph)

    diffmap_key = "X_diffmap"
    if key_added:
        diffmap_key = f"{diffmap_key}_{key_added}"
        _rename_obsm_key(adata, "X_diffmap", diffmap_key)
    else:
        _delete_obsm_backup_key(adata, "X_diffmap")

    if export_embedding is not None:
        write_embedding(adata, diffmap_key, export_embedding, key_added=key_added)
