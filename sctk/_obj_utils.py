"""
Provide helper functions for constructing sub-commands
"""

import scanpy as sc
import pandas as pd

sc_warn = sc.logging.warn if sc.__version__.startswith("1.4") else sc.logging.warning


def write_cluster(adata, keys, cluster_fn, sep="\t"):
    """Export cell clustering as a text table"""
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for key in keys:
        if key not in adata.obs.keys():
            raise KeyError(f"{key} is not a valid `.uns` key")
    adata.obs[keys].reset_index(level=0).rename(columns={"index": "cells"}).to_csv(
        cluster_fn, sep=sep, header=True, index=False
    )


def write_embedding(adata, key, embed_fn, n_comp=None, sep="\t", key_added=None):
    """Export cell embeddings as a txt table"""
    if key_added:
        if embed_fn.endswith(".tsv"):
            embed_fn = embed_fn[0:-4]
        embed_fn = f"{embed_fn}_{key_added}.tsv"
    if key not in adata.obsm.keys():
        raise KeyError(f"{key} is not a valid `.obsm` key")
    mat = adata.obsm[key].copy()
    if n_comp is not None and mat.shape[1] >= n_comp:
        mat = mat[:, 0:n_comp]
    pd.DataFrame(mat, index=adata.obs_names).to_csv(embed_fn, sep=sep, header=False, index=True)


# The functions below handles slot key.
#
# Default keys are those read and written by scanpy functions by default, e.g
# "X_pca", "neighbors", "louvain", etc.
#
# Of them, `obsm_key` specifically refers to those used for embedding, e.g
# "X_pca", "X_tsne", "X_umap", etc.
#
# The approach for supplying a non-standard key to a function as input is:
# if the function only reads the value in the default key, we first backup the
# value in the default key, then write the value of the non-standard key into
# the standard key, run the funtion, and finally restore the value of the
# default key from backup and delete the backup.
#
# The approach for writting the results of a function to a non-standard key is:
# if the function only writes to the default key, we first backup the value in
# the default key, run the function, copy the value of the default key to the
# desired non-standard key, and finally restore the value of the default key
# from backup and delete the backup.
#
# Specical treatment for obsm_key is needed, as the underlying data type is not
# a python dictionary but a numpy array.


def _backup_default_key(slot, default):
    if default in slot.keys():
        bkup_key = f"{default}_bkup"
        if bkup_key in slot.keys():
            sc_warn(f"overwrite existing {bkup_key}")
        slot[bkup_key] = slot[default]


def _restore_default_key(slot, default, key=None):
    if key != default:
        bkup_key = f"{default}_bkup"
        if bkup_key in slot.keys():
            slot[default] = slot[bkup_key]
            del slot[bkup_key]


def _delete_backup_key(slot, default):
    bkup_key = f"{default}_bkup"
    if bkup_key in slot.keys():
        del slot[bkup_key]


def _set_default_key(slot, default, key):
    if key != default:
        if key not in slot.keys():
            raise KeyError(f"{key} does not exist")
        _backup_default_key(slot, default)
        slot[default] = slot[key]


def _rename_default_key(slot, default, key):
    if not default in slot.keys():
        raise KeyError(f"{default} does not exist")
    slot[key] = slot[default]
    del slot[default]
    _restore_default_key(slot, default)


def _backup_obsm_key(adata, key):
    if key in adata.obsm_keys():
        bkup_key = f"{key}_bkup"
        if bkup_key in adata.obsm_keys():
            sc_warn(f"overwrite existing {bkup_key}")
        adata.obsm[bkup_key] = adata.obsm[key]


def _restore_obsm_key(adata, key, new_key=None):
    if new_key != key:
        bkup_key = f"{key}_bkup"
        if bkup_key in adata.obsm_keys():
            adata.obsm[key] = adata.obsm[bkup_key]
            del adata.obsm[bkup_key]


def _delete_obsm_backup_key(adata, key):
    bkup_key = f"{key}_bkup"
    if bkup_key in adata.obsm_keys():
        del adata.obsm[bkup_key]


def _set_obsm_key(adata, key, new_key):
    if new_key != key:
        if new_key not in adata.obsm_keys():
            raise KeyError(f"{new_key} does not exist")
        _backup_obsm_key(adata, key)
        adata.obsm[key] = adata.obsm[new_key]


def _rename_obsm_key(adata, from_key, to_key):
    if not from_key in adata.obsm_keys():
        raise KeyError(f"{from_key} does not exist")
    adata.obsm[to_key] = adata.obsm[from_key]
    del adata.obsm[from_key]
    _restore_obsm_key(adata, from_key)
