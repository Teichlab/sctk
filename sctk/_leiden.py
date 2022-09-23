"""
scanpy leiden
"""

import re
import numpy as np
import scanpy as sc
from ._obj_utils import write_cluster


def leiden(
    adata,
    resolution,
    flavor=None,
    use_graph=None,
    key_added=None,
    export_cluster=None,
    legacy=False,
    **kwargs,
):
    """
    Wrapper function for sc.tl.leiden, for supporting multiple resolutions.
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
        if sc.__version__.startswith("1.4") or legacy:
            adj_mat = adata.uns[use_graph]["connectivities"]
        else:
            ckey = adata.uns[use_graph]["connectivities_key"]
            adj_mat = adata.obsp[ckey]
    if not isinstance(resolution, (list, tuple)):
        if key_added is not None and not key_added.startswith("leiden_"):
            key_added = f"leiden_{key_added}"
        elif key_added is None:
            key_added = "leiden"
        if flavor == "rapids":
            kwargs["flavor"] = "rapids"
        sc.tl.leiden(adata, resolution=resolution, adjacency=adj_mat, key_added=key_added, **kwargs)
        keys.append(key_added)
    else:
        for i, res in enumerate(resolution):
            res_key = str(res).replace(".", "_")
            if key_added is None:
                graph_key = ("_" + use_graph) if use_graph else ""
                key = f"leiden{graph_key}_r{res_key}"
            elif not isinstance(key_added, (list, tuple)):
                key = f"leiden_{key_added}_r{res_key}"
            elif len(key_added) == len(resolution):
                key = key_added[i]
            else:
                raise ValueError(
                    "`key_added` can only be None, a scalar, or an "
                    "iterable of the same length as `resolution`."
                )
            keys.extend(
                leiden(
                    adata,
                    resolution=res,
                    flavor=flavor,
                    use_graph=use_graph,
                    key_added=key,
                    **kwargs,
                )
            )

    if export_cluster:
        write_cluster(adata, keys, export_cluster)

    return keys


def leiden_shredding(
    adata,
    resolution=[0.5, 0.1],
    use_graph=None,
    level1_groups=None,
    key_added=None,
    min_cluster_size=10,
    max_cluster_size=1000,
    **kwargs,
):
    # resolution_function = lambda x: np.maximum(np.maximum(np.log10(x)-1, 0)**2, 0.1)
    if not isinstance(resolution, (list, tuple)):
        resolution = [resolution, resolution]
    adj_mat = None
    if use_graph:
        if use_graph not in adata.uns:
            raise KeyError(f'"{use_graph}" is not a valid key of `.uns`.')
        if sc.__version__.startswith("1.4"):
            adj_mat = adata.uns[use_graph]["connectivities"]
        else:
            ckey = adata.uns[use_graph]["connectivities_key"]
            adj_mat = adata.obsp[ckey]
    if key_added is not None and not key_added.startswith("leiden_"):
        key_added = f"leiden_{key_added}"
    elif key_added is None:
        key_added = "leiden"
    if level1_groups and level1_groups in adata.obs.columns:
        adata.obs[key_added] = adata.obs[level1_groups].copy()
    else:
        sc.tl.leiden(
            adata, adjacency=adj_mat, resolution=resolution[0], key_added=key_added, **kwargs
        )
    level1_clusters = np.unique(adata.obs[key_added])
    for clst in level1_clusters:
        clst_size = sum(adata.obs[key_added] == clst)
        if clst_size <= min_cluster_size:
            continue
        if max_cluster_size and clst_size <= max_cluster_size:
            continue
        sc.tl.leiden(
            adata,
            restrict_to=(key_added, [clst]),
            key_added="leiden_temp",
            adjacency=adj_mat,
            resolution=resolution[1] * max(1, np.sqrt(clst_size / min_cluster_size)),
            **kwargs,
        )
        adata.obs[key_added] = adata.obs["leiden_temp"]
    del adata.obs["leiden_temp"]


def leiden_shredding2(
    adata,
    adjacency=None,
    resolution=0.5,
    initial_groupby=None,
    restrict_to=None,
    key_added=None,
    level=0,
    min_cluster_size=20,
    max_cluster_size=200,
    **kwargs,
):
    tmp_key = f"{key_added}_{level}"
    if initial_groupby and initial_groupby in adata.obs.columns:
        adata.obs[tmp_key] = adata.obs[initial_groupby].copy()
    else:
        sc.tl.leiden(
            adata,
            adjacency=adjacency,
            restrict_to=restrict_to,
            resolution=resolution,
            key_added=tmp_key,
            **kwargs,
        )
    current_clusters = np.unique(adata.obs[tmp_key])
    if restrict_to is not None:
        _, parent_group = restrict_to
        parent_group = parent_group[0]
        clst_pattern = re.compile(f"{re.escape(parent_group)}(,[0-9]+)+$")
        current_clusters = [clst for clst in current_clusters if clst_pattern.match(clst)]
    clst_sizes =  adata.obs[tmp_key].value_counts().loc[current_clusters]
    if len(current_clusters) < 2 and clst_sizes.max() > max_cluster_size:
        # require re-do with increased resolution
        return 1
    if (
        min_cluster_size is not None
        and (level > 0 or initial_groupby is None)
        and clst_sizes.min() < min_cluster_size
        and (clst_sizes < min_cluster_size).sum() > 1
        and len(current_clusters) > 2
    ):
        # require re-do with decreased resolution
        return -1
    for clst in current_clusters:
        clst_size = sum(adata.obs[tmp_key] == clst)
        res = resolution
        if max_cluster_size is not None and clst_size > max_cluster_size:
            # require another round of leiden
            max_res, min_res = None, None
            while True:
                ret = leiden_shredding2(
                    adata,
                    adjacency=adjacency,
                    resolution=res,
                    restrict_to=(tmp_key, [clst]),
                    key_added=key_added,
                    min_cluster_size=min_cluster_size,
                    max_cluster_size=max_cluster_size,
                    level=level + 1,
                    **kwargs
                )
                if ret == 0:
                    break

                if ret > 0:
                    if max_res is None:
                        next_res = res + 0.1 if res >= 0.1 else res * 2
                    else:
                        next_res = (res + max_res) / 2
                    min_res = res
                else:
                    if min_res is None:
                        next_res = res - 0.1 if res >= 0.2 else res / 2
                    else:
                        next_res = (res + min_res) / 2
                    max_res = res

                # Don't search forever
                if np.abs(next_res - res) < 0.01:
                    print("stop search")
                    break

                res = next_res

            next_tmp_key = f"{key_added}_{level + 1}"
            adata.obs[tmp_key] = adata.obs[next_tmp_key]
            del adata.obs[next_tmp_key]

    if level == 0:
        adata.obs[key_added] = adata.obs[tmp_key]
        del adata.obs[tmp_key]

    return 0
