"""
scanpy dpt
"""

import numpy as np
import scanpy as sc
from ._obj_utils import (
    _set_default_key,
    _restore_default_key,
    _backup_default_key,
    _delete_backup_key,
    _rename_default_key,
    _set_obsm_key,
    _restore_obsm_key,
)


def dpt(
    adata,
    root=None,
    use_graph="neighbors",
    use_diffmap="X_diffmap",
    key_added=None,
    **kwargs,
):
    """
    Wrapper function for sc.tl.dpt, for support named slot
    """
    if root is None or not (isinstance(root, (list, tuple)) and len(root) == 2):
        root = (None, None)
    if "iroot" not in adata.uns.keys() and root[0] is None:
        raise ValueError(
            "Annotate your data with root cell first, i.e. "
            'boolean vector `.uns["iroot"]` is required.'
        )
    if root[0] is not None:
        adata.uns["iroot"] = np.random.choice(np.flatnonzero(adata.obs[root[0]] == root[1]))

    _set_default_key(adata.uns, "neighbors", use_graph)
    _set_obsm_key(adata, "X_diffmap", use_diffmap)

    _backup_default_key(adata.obs, "dpt_pseudotime")

    sc.tl.dpt(adata, **kwargs)

    _restore_default_key(adata.uns, "neighbors", use_graph)
    _restore_obsm_key(adata, "X_diffmap", use_diffmap)

    if key_added:
        dpt_key = f"dpt_pseudotime_{key_added}"
        _rename_default_key(adata.obs, "dpt_pseudotime", dpt_key)
    else:
        _delete_backup_key(adata.obs, "dpt_pseudotime")
