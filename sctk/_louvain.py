"""
scanpy louvain
"""

import scanpy as sc
from ._obj_utils import write_cluster


def louvain(adata, resolution, use_graph=None, key_added=None, export_cluster=None, **kwargs):
    """
    Wrapper function for sc.tl.louvain, for supporting multiple resolutions.
    """
    keys = []
    if "restrict_to" in kwargs and not (
        isinstance(kwargs["restrict_to"], (list, tuple))
        and len(kwargs["restrict_to"]) == 2
        and kwargs["restrict_to"][0]
    ):
        del kwargs["restrict_to"]
    adj_mat = None
    if use_graph:
        if use_graph not in adata.uns:
            raise KeyError(f'"{use_graph}" is not a valid key of `.uns`.')
        adj_mat = adata.uns[use_graph]["connectivities"]
    if not isinstance(resolution, (list, tuple)):
        if key_added is not None and not key_added.startswith("louvain_"):
            key_added = f"louvain_{key_added}"
        elif key_added is None:
            key_added = "louvain"
        sc.tl.louvain(
            adata, resolution=resolution, adjacency=adj_mat, key_added=key_added, **kwargs
        )
        keys.append(key_added)
    else:
        for i, res in enumerate(resolution):
            res_key = str(res).replace(".", "_")
            if key_added is None:
                graph_key = ("_" + use_graph) if use_graph else ""
                key = f"louvain{graph_key}_r{res_key}"
            elif not isinstance(key_added, (list, tuple)):
                key = f"louvain_{key_added}_r{res_key}"
            elif len(key_added) == len(resolution):
                key = key_added[i]
            else:
                raise ValueError(
                    "`key_added` can only be None, a scalar, or an "
                    "iterable of the same length as `resolution`."
                )
            keys.extend(
                louvain(
                    adata,
                    resolution=res,
                    use_graph=use_graph,
                    key_added=key,
                    **kwargs,
                )
            )

    if export_cluster:
        write_cluster(adata, keys, export_cluster)

    return keys
