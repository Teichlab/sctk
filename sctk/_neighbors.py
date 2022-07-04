"""
scanpy neighbors
"""

import scanpy as sc
from ._obj_utils import (
    _backup_default_key,
    _delete_backup_key,
    _rename_default_key,
)


def neighbors(adata, n_neighbors=15, key_added=None, **kwargs):
    """
    Wrapper function for sc.pp.neighbors(), for supporting multiple n_neighbors
    """
    if not isinstance(n_neighbors, (list, tuple)):
        _backup_default_key(adata.uns, "neighbors")

        if sc.__version__.startswith("1.4"):
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, **kwargs)

            if key_added:
                nb_key = f"neighbors_{key_added}"
                _rename_default_key(adata.uns, "neighbors", nb_key)
            else:
                _delete_backup_key(adata.uns, "neighbors")
        else:
            if key_added:
                nb_key = f"neighbors_{key_added}"
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, key_added=nb_key, **kwargs)
    else:
        for i, n_nb in enumerate(n_neighbors):
            if key_added is None:
                graph_key = f"k{n_nb}"
            elif not isinstance(key_added, (list, tuple)):
                graph_key = f"{key_added}_k{n_nb}"
            elif len(key_added) == len(n_neighbors):
                graph_key = key_added[i]
            else:
                raise ValueError(
                    "`key_added` can only be None, a scalar, or an "
                    "iterable of the same length as `n_neighbors`."
                )
            neighbors(
                adata,
                n_neighbors=n_nb,
                key_added=graph_key,
                **kwargs,
            )
