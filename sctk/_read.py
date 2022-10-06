"""
Provides read_10x()
"""

import os
from tempfile import mkdtemp
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import anndata


def read_10x(
    input_10x_h5,
    input_10x_mtx,
    genome="hg19",
    var_names="gene_symbols",
    extra_obs=None,
    extra_var=None,
):
    """
    Wrapper function for sc.read_10x_h5() and sc.read_10x_mtx(), mainly to
    support adding extra metadata
    """
    if input_10x_h5 is not None:
        adata = sc.read_10x_h5(input_10x_h5, genome=genome)
    elif input_10x_mtx is not None:
        adata = sc.read_10x_mtx(input_10x_mtx, var_names=var_names)

    if extra_obs:
        obs_tbl = pd.read_csv(extra_obs, sep="\t", header=0, index_col=0)
        adata.obs = adata.obs.merge(
            obs_tbl,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )

    if extra_var:
        var_tbl = pd.read_csv(extra_var, sep="\t", header=0, index_col=0)
        adata.var = adata.var.merge(
            var_tbl,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )
    return adata


def read_10x_atac(input_10x_mtx, extra_obs=None, extra_var=None):
    """
    Wrapper function for sc.read_10x_mtx() to read 10x ATAC data
    """
    matrix_mtx = os.path.join(input_10x_mtx, "matrix.mtx")
    barcodes_tsv = os.path.join(input_10x_mtx, "barcodes.tsv")
    peaks_bed = os.path.join(input_10x_mtx, "peaks.bed")
    if not os.path.exists(peaks_bed):
        raise FileNotFoundError

    tmp_dir = mkdtemp()
    genes_tsv = os.path.join(tmp_dir, "genes.tsv")

    os.symlink(os.path.abspath(matrix_mtx), os.path.join(tmp_dir, "matrix.mtx"))
    os.symlink(os.path.abspath(barcodes_tsv), os.path.join(tmp_dir, "barcodes.tsv"))
    with open(peaks_bed) as f, open(genes_tsv, "w") as fw:
        for line in f:
            fields = line.rstrip().split("\t")
            peak_id = "_".join(fields)
            print(f"{peak_id}\t{peak_id}", file=fw)

    adata = sc.read_10x_mtx(tmp_dir)
    adata.var.rename(columns={"gene_ids": "peak"}, inplace=True)

    os.remove(os.path.join(tmp_dir, "matrix.mtx"))
    os.remove(os.path.join(tmp_dir, "barcodes.tsv"))
    os.remove(genes_tsv)
    os.removedirs(tmp_dir)

    if extra_obs:
        obs_tbl = pd.read_csv(extra_obs, sep="\t", header=0, index_col=0)
        adata.obs = adata.obs.merge(
            obs_tbl,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )

    if extra_var:
        var_tbl = pd.read_csv(extra_var, sep="\t", header=0, index_col=0)
        adata.var = adata.var.merge(
            var_tbl,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )
    return adata


def read_cellbender(
    input_h5,
    remove_zero=True,
    remove_nan=True,
    train_history=False,
    latent_gene_encoding=False,
    add_suffix=None,
):
    """
    Read cellbender output h5 generated from mtx input
    """
    import scipy.sparse as sp

    f = h5py.File(input_h5, "r")
    if "matrix" in f:
        mat = f["matrix"]
        feat = mat["features"]
        feat_name = feat["name"][()]
        vardict = {
            "gene_ids": feat["id"][()].astype(str),
            "feature_type": feat["feature_type"][:].astype(str),
        }
    elif "background_removed" in f:
        mat = f["background_removed"]
        vardict = {
            "gene_ids": mat["genes"][()].astype(str),
        }
        feat_name = mat["gene_names"][()]
    else:
        raise ValueError("The data doesn't look like cellbender output")
    n_var, n_obs = tuple(mat["shape"][()])
    cols = ["latent_cell_probability", "latent_RT_efficiency"]
    if "barcode_indices_for_latents" in mat:
        bidx = mat["barcode_indices_for_latents"][()]
        obsdict = {}
        for x in cols:
            val = np.empty(n_obs)
            val.fill(np.nan)
            val[bidx] = mat[x][()]
            obsdict[x] = val
        if latent_gene_encoding:
            lge = mat["latent_gene_encoding"][()]
            obsm = np.empty((n_obs, lge.shape[1]))
            obsm.fill(np.nan)
            obsm[bidx, :] = lge
    else:
        obsdict = {x: mat[x] for x in cols}
        if latent_gene_encoding:
            obsm = mat["latent_gene_encoding"][()]
    barcodes = np.array(
        [b[:-2] if b.endswith("-1") else b for b in mat["barcodes"][()].astype(str)]
    )
    ad = anndata.AnnData(
        X=sp.csr_matrix(
            (mat["data"][()], mat["indices"][()], mat["indptr"][()]),
            shape=(n_obs, n_var),
        ),
        var=pd.DataFrame(vardict, index=feat_name.astype(str)),
        obs=pd.DataFrame(obsdict, index=barcodes),
        uns={
            "target_false_positive_rate": mat["target_false_positive_rate"][()],
            "test_elbo": list(mat["test_elbo"]),
            "test_epoch": list(mat["test_epoch"]),
            "training_elbo_per_epoch": list(mat["training_elbo_per_epoch"]),
        }
        if train_history
        else {},
    )
    ad.var_names_make_unique()
    if latent_gene_encoding:
        ad.obsm["X_latent_gene_encoding"] = obsm

    mask_nan = np.isnan(ad.obs.latent_cell_probability)
    mask_0 = ad.X.sum(axis=1).A1 <= 0

    mask_remove = np.zeros(n_obs).astype(bool)
    if remove_nan:
        mask_remove = mask_remove | mask_nan
    if remove_zero:
        mask_remove = mask_remove | mask_0
    idx_remove = np.where(mask_remove)[0]

    idx_sort = pd.Series(np.argsort(ad.obs_names))
    idx_sort = idx_sort[~idx_sort.isin(idx_remove)]

    ad1 = ad[idx_sort.values].copy()
    del ad

    if add_suffix:
        ad1.obs_names = ad1.obs_names.astype(str) + add_suffix

    return ad1


def read_h5ad(input_h5ad, component="all", **kwargs):
    if component == "all":
        return sc.read(input_h5ad, **kwargs)
    elif component == "raw":
        with h5py.File(input_h5ad, "r") as f:
            raw_dict = anndata._io.utils.read_attribute(f["/raw"])
            obs = anndata._io.h5ad.read_dataframe(f["/obs"])
            return anndata.AnnData(obs=obs, **raw_dict)
    elif isinstance(component, str):
        component = "/" + component if not component.startswith("/") else component
        if component in {"/obs", "/var"}:
            with h5py.File(input_h5ad, "r") as f:
                df = anndata._io.h5ad.read_dataframe(f[component])
                if not isinstance(f[component], h5py.Group):
                    for cats_name in f["/uns"].keys():
                        if not cats_name.endswith("_categories"):
                            continue
                        name = cats_name.replace("_categories", "")
                        cats = f[f"/uns/{cats_name}"][:]
                        if isinstance(cats, (str, int)):
                            cats = [cats]
                        if name in df.columns:
                            df[name] = pd.Categorical.from_codes(df[name].values, cats)
                return df
        elif component.startswith("/uns/neighbors") and len(component.split("/")) == 3:
            with h5py.File(input_h5ad, "r") as f:
                neighbor = anndata._io.h5ad.read_attribute(f[component])
                if "connectivities_key" in neighbor.keys():
                    c_key = neighbor["connectivities_key"]
                    neighbor["connectivities"] = anndata._io.h5ad.read_attribute(
                        f[f"/obsp/{c_key}"]
                    )
                if "distances_key" in neighbor.keys():
                    d_key = neighbor["distances_key"]
                    neighbor["distances"] = anndata._io.h5ad.read_attribute(f[f"/obsp/{d_key}"])
                return neighbor
        else:
            with h5py.File(input_h5ad, "r") as f:
                return anndata._io.utils.read_attribute(f[component])
    elif isinstance(component, (list, tuple)):
        d = dict()
        for comp in component:
            d[comp] = read_h5ad(input_h5ad, component=comp, **kwargs)
        return d
    else:
        raise ValueError(
            "`component` must be a str ('all', 'raw' or absolute HDF5 path), tuple or list of str"
        )


def read_velocyto(
    mtx_path,
    spliced_mtx="spliced.mtx.gz",
    unspliced_mtx="unspliced.mtx.gz",
    ambiguous_mtx="ambiguous.mtx.gz",
):

    barcodes_tsv = os.path.join(mtx_path, "barcodes.tsv.gz")
    features_tsv = os.path.join(mtx_path, "features.tsv.gz")
    mtx_dict = {"spliced": spliced_mtx, "unspliced": unspliced_mtx, "ambiguous": ambiguous_mtx}
    ad_dict = {}
    for name, mtx in mtx_dict.items():
        matrix_mtx = os.path.join(mtx_path, mtx)
        tmp_dir = mkdtemp()
        tmp_barcodes_tsv = os.path.join(tmp_dir, "barcodes.tsv.gz")
        tmp_features_tsv = os.path.join(tmp_dir, "features.tsv.gz")
        tmp_matrix_mtx = os.path.join(tmp_dir, "matrix.mtx.gz")
        os.symlink(barcodes_tsv, tmp_barcodes_tsv)
        os.symlink(features_tsv, tmp_features_tsv)
        os.symlink(matrix_mtx, tmp_matrix_mtx)
        try:
            ad_dict[name] = sc.read_10x_mtx(tmp_dir)
        finally:
            os.remove(tmp_barcodes_tsv)
            os.remove(tmp_features_tsv)
            os.remove(tmp_matrix_mtx)
            os.rmdir(tmp_dir)
    spliced_ad = ad_dict["spliced"]
    spliced_ad.layers["unspliced"] = ad_dict["unspliced"].X.copy()
    spliced_ad.layers["ambiguous"] = ad_dict["ambiguous"].X.copy()
    del ad_dict["unspliced"], ad_dict["ambiguous"]
    return spliced_ad
