"""
Utility functions
"""

import warnings
import os
import re
import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp
import pandas as pd
import anndata
import scanpy as sc
from ._obj_utils import (
    _set_obsm_key,
    _restore_obsm_key,
    _backup_default_key,
    _rename_default_key,
    _delete_backup_key,
)


sc_warn = sc.logging.warn if sc.__version__.startswith("1.4") else sc.logging.warning


def _find_rep(adata, use_rep, return_var_names=True):
    if isinstance(use_rep, str):
        if use_rep == "raw" and adata.raw:
            X = adata.raw.X
            var_names = adata.raw.var_names.values
        elif use_rep == "X":
            X = adata.X
            var_names = adata.var_names.values
        elif use_rep in adata.layers.keys():
            X = adata.layers[use_rep]
            var_names = adata.var_names.values
        elif use_rep in adata.obsm.keys():
            X = adata.obsm[use_rep]
            var_names = np.array([f"{use_rep}{i+1}".replace("X_", "") for i in range(X.shape[1])])
        elif use_rep in adata.obs.keys():
            X = adata.obs[use_rep].values.reshape((adata.n_obs, 1))
            var_names = np.array([use_rep])
        else:
            raise ValueError("Invalid `use_rep` provided.")
    elif isinstance(use_rep, np.ndarray) and use_rep.shape[0] == adata.n_obs:
        x = use_rep
        var_names = np.arange(x.shape[1])
    elif isinstance(use_rep, pd.DataFrame) and use_rep.shape[0] == adata.n_obs:
        x = use_rep.values
        var_names = use_rep.columns.values
    else:
        raise ValueError("Invalid `use_rep` provided.")

    if return_var_names:
        return (X, var_names)
    return X


def read_list(fn, **kwargs):
    return pd.read_csv(fn, header=None, names=["x"], **kwargs).x.to_list()


def lognorm_to_counts(X, norm_sum=1e4, n_counts=None, force=False, rounding=True):
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_expm1 = np.expm1(X)
    if n_counts is not None:
        size_factor = n_counts / norm_sum
        X_counts = (X_expm1.T * sp.csr_matrix(sp.diags(size_factor))).T
        res = np.abs(X_counts.data - np.round(X_counts.data)).sum() / X_counts.data.sum()
        if res < 1e-6 or force:
            if rounding:
                X_counts.data = np.round(X_counts.data).astype(np.int32)
            return X_counts
        warnings.warn(
            f"Non-integer residuals too large (res = {res}), try inferring size_factor"
        )
    x_min = np.array([X_expm1.getrow(i).data.min() for i in range(X_expm1.shape[0])])
    size_factor = 1 / x_min
    X_counts = (X_expm1.T * sp.csr_matrix(sp.diags(size_factor))).T
    res = np.abs(X_counts.data - np.round(X_counts.data)).sum() / X_counts.data.sum()
    if res < 1e-3 or force:
        if rounding:
            X_counts.data = np.round(X_counts.data).astype(np.int32)
        return X_counts
    raise ValueError(f"Non-integer residuals too large (res = {res}), failed to recover counts")


def restore_adata(
    adata,
    restore_type="count",
    use_raw=True,
    obs_cols=None,
    var_cols=None,
    obsm_keys=None,
    use_n_counts=False,
    **kwargs,
):
    if use_raw and not adata.raw:
        raise ValueError("adata.raw not found")

    if restore_type == "norm":
        if use_raw:
            X = adata.raw.X
        else:
            X = adata.X
    elif restore_type == "count":
        n_counts = None
        if use_n_counts is True:
            if use_n_counts in adata.obs_keys():
                n_counts = adata.obs[use_n_counts].values
            elif "n_counts" in adata.obs_keys():
                n_counts = adata.obs["n_counts"].values
        if use_raw:
            X0 = adata.raw.X
        else:
            X0 = adata.X
        X = lognorm_to_counts(X0, n_counts=n_counts, **kwargs)
    else:
        raise ValueError(
            f'{restore_type}: invalid <restore_type>, choose between "norm" and "count"'
        )

    obs = adata.obs[obs_cols].copy() if obs_cols else adata.obs.copy()
    if use_raw:
        var = adata.raw.var[var_cols].copy() if var_cols else adata.raw.var.copy()
    else:
        var = adata.var[var_cols].copy() if var_cols else adata.var.copy()
    ad = anndata.AnnData(X=X, obs=obs, var=var)
    if obsm_keys:
        for ok in obsm_keys:
            if ok in adata.obsm.keys():
                ad.obsm[ok] = adata.obsm[ok]
    return ad


def _get_index_in_superset(superset, x):
    index = np.argsort(superset)
    xpos = np.searchsorted(superset[index], x)
    return index[xpos]


def expand_feature_space(ad, features):
    var_names = ad.var_names
    n_features = features.size
    if not var_names.isin(features).all():
        raise ValueError("`ad.var_names` must be a strict subset of `features`")
    k_included = _get_index_in_superset(features, var_names.values)
    k_extra = np.array([i for i in range(n_features) if i not in k_included])
    new_ad = anndata.AnnData(
        X=sp.csr_matrix((ad.X.data, ad.X.indices, ad.X.indptr), shape=(ad.n_obs, n_features)),
        obs=ad.obs,
    )
    new_ad.var_names = var_names.tolist() + list(features[k_extra])
    for layer in ad.layers.keys():
        layer_X = ad.layers[layer]
        new_ad.layers[layer] = sp.csr_matrix(
            (layer_X.data, layer_X.indices, layer_X.indptr), shape=(ad.n_obs, n_features)
        )
    return new_ad[:, features].copy()


def find_top_expressed_genes(ad, use_rep="X", groupby=None, n_genes=50, inplace=True):
    X, var_names = _find_rep(ad, use_rep)
    counts = X.sum(axis=1)
    if sp.issparse(X):
        counts = counts.A1
    X = sc.preprocessing._normalization._normalize_data(X, counts, after=100, copy=True)
    if groupby:
        raise NotImplementedError
    else:
        mean_percent = X.mean(axis=0)
        if sp.issparse(X):
            mean_percent = mean_percent.A1
        top_idx = np.argsort(mean_percent)[::-1][:n_genes]
    top_genes = var_names[top_idx]
    if inplace:
        ad.var[f"top{n_genes}"] = ad.var_names.isin(top_genes)
    else:
        return top_genes


def remove_genes(adata, var_flags):
    if isinstance(var_flags, (tuple, list)):
        pass
    elif isinstance(var_flags, str):
        var_flags = [var_flags]
    else:
        var_flags = []
    if var_flags:
        mask = np.zeros(adata.n_vars).astype(bool)
        for vf in var_flags:
            mask = mask | adata.var[vf]
        adata = adata[:, ~mask].copy()
    return adata


def project_into_PC(
    source_adata,
    target_adata,
    source_scaled=False,
    target_scaled=True,
    source_rep="X",
    target_pca="X_pca",
    use_loadings=None,
    key_added=None,
):
    """project cells from one dataset (source) into PC space of another dataset (target)"""
    if source_rep == "raw":
        src_ad = source_adata.raw.to_anndata()
    else:
        src_ad = source_adata

    tgt_ad = target_adata

    if np.any(~tgt_ad.var_names.isin(src_ad.var_names)):
        raise ValueError("`source_adata` must contain all features in `target_adata`")

    if np.any(src_ad.var_names[src_ad.var_names.isin(tgt_ad.var_names)] != tgt_ad.var_names):
        raise ValueError("features in `source_adata` must be in the same order as `target_adata`")

    tgt_X = tgt_ad.X
    tgt_X_is_dense = not sp.issparse(tgt_X)

    aux_ad = src_ad[:, src_ad.var_names.isin(tgt_ad.var_names)].copy()

    if not source_scaled:
        sc.pp.scale(aux_ad, zero_center=tgt_X_is_dense, max_value=10)

    if not target_scaled:
        tgt_X = sc.pp.scale(tgt_X, zero_center=tgt_X_is_dense, max_value=10)

    if use_loadings:
        tgt_V = use_loadings
    else:
        if not tgt_X_is_dense:
            tgt_X = tgt_X.toarray()
        tgt_V = np.linalg.lstsq(tgt_X, tgt_ad.obsm[target_pca])

    proj_pca = aux_ad.X @ tgt_V[0]
    del aux_ad, src_ad

    if key_added:
        source_adata.obsm[f"X_pca_{key_added}"] = proj_pca
        return tgt_V
    return proj_pca, tgt_V


def _is_numeric(x):
    return x.dtype.kind in ("i", "f")


def cross_table(
    adata,
    x,
    y,
    normalise=None,
    exclude_x=None,
    exclude_y=None,
    subset=None,
    sort=False,
    include_nan=False,
):
    """Make a cross table comparing two categorical annotations"""
    x_attr = adata.obs[x]
    y_attr = adata.obs[y]
    assert not _is_numeric(x_attr.values), f"Can not operate on numerical {x}"
    assert not _is_numeric(y_attr.values), f"Can not operate on numerical {y}"
    if include_nan:
        x_attr = x_attr.astype(str).astype("category")
        y_attr = y_attr.astype(str).astype("category")
    if subset is not None:
        x_attr = x_attr[subset]
        y_attr = y_attr[subset]
    selection = np.ones(x_attr.size, dtype=bool)
    if exclude_x:
        selection = selection & ~x_attr.isin(exclude_x)
    if exclude_y:
        selection = selection & ~y_attr.isin(exclude_y)
    x_attr = x_attr[selection]
    y_attr = y_attr[selection]

    if normalise in (0, "x", "xy", "row", "index"):
        norm_method = "index"
    elif normalise in (1, "y", "yx", "col", "columns"):
        norm_method = "columns"
    else:
        norm_method = False
    crs_tbl = pd.crosstab(x_attr, y_attr, dropna=False, normalize=norm_method)
    nx, ny = crs_tbl.shape
    if normalise in ("xy", "yx", "jaccard"):
        x_sizes = crs_tbl.sum(axis=1).values
        y_sizes = crs_tbl.sum(axis=0).values
        if normalise == "yx":
            normaliser = np.tile(x_sizes.reshape(nx, 1), (1, ny))
        elif normalise == "xy":
            normaliser = np.tile(y_sizes.reshape(1, ny), (nx, 1))
        else:
            normaliser = (
                np.tile(x_sizes.reshape(nx, 1), (1, ny))
                + np.tile(y_sizes.reshape(1, ny), (nx, 1))
                - crs_tbl.values
            )
        crs_tbl = (crs_tbl / normaliser).round(4)
    if sort in ("x", "index", "xy", "yx", "both"):
        crs_tbl = crs_tbl.sort_index()
    if sort in ("y", "columns", "xy", "yx", "both"):
        crs_tbl = crs_tbl.sort_columns()
    return crs_tbl


def run_celltypist(
    ad, model, use_rep="X", require_lognorm=False, min_prob=None, key_added="ctp_pred"
):
    import celltypist

    X, var_names = _find_rep(ad, use_rep)
    aux_ad = anndata.AnnData(X=X)
    aux_ad.obs_names = ad.obs_names
    aux_ad.var_names = var_names
    if require_lognorm:
        sc.pp.normalize_total(aux_ad, target_sum=1e4)
        sc.pp.log1p(aux_ad)
    pred = celltypist.annotate(aux_ad, model=model, majority_voting=True)
    ad.obs[f"{key_added}"] = pred.predicted_labels.majority_voting
    ad.obs[f"{key_added}_prob"] = pred.probability_matrix.max(axis=1)
    if min_prob is not None:
        ad.obs[f"{key_added}_uncertain"] = ad.obs[f"{key_added}_prob"] < min_prob
        ad.obs[f"{key_added}_uncertain"] = ad.obs[f"{key_added}_uncertain"].astype("category")
    del aux_ad


def run_harmony(adata, batch, theta=2.0, use_rep="X_pca", key_added="hm", random_state=0, **kwargs):
    if not isinstance(batch, (tuple, list)):
        batch = [batch]
    if not isinstance(theta, (tuple, list)):
        theta = [theta] * len(batch)
    for b in batch:
        if b not in adata.obs.columns:
            raise KeyError(f"{b} is not a valid obs annotation.")
    if use_rep not in adata.obsm.keys():
        raise KeyError(f"{use_rep} is not a valid embedding.")
    meta = adata.obs[batch].reset_index()
    embed = adata.obsm[use_rep]

    if sc.__version__.startswith("1.4"):
        # ===========
        import rpy2.robjects
        from rpy2.robjects.packages import importr

        harmony = importr("harmony")
        from rpy2.robjects import numpy2ri, pandas2ri

        numpy2ri.activate()
        pandas2ri.activate()
        set_seed = rpy2.robjects.r("set.seed")
        set_seed(random_state)
        hm_embed = harmony.HarmonyMatrix(
            embed, meta, batch, theta, do_pca=False, verbose=False, **kwargs
        )
        pandas2ri.deactivate()
        numpy2ri.deactivate()
        hm_embed = numpy2ri.ri2py(hm_embed)
        # ===========
    else:
        import harmonypy as hm

        if "max_iter_harmony" not in kwargs:
            kwargs["max_iter_harmony"] = 20
        hobj = hm.run_harmony(embed, meta, batch, theta=theta, **kwargs)
        hm_embed = hobj.Z_corr.T

    if key_added:
        adata.obsm[f"{use_rep}_{key_added}"] = hm_embed
    else:
        adata.obsm[use_rep] = hm_embed.T


def run_bbknn(adata, batch, use_rep="X_pca", key_added="bk", **kwargs):
    import bbknn

    _set_obsm_key(adata, "X_pca", use_rep)
    try:
        _backup_default_key(adata.uns, "neighbors")
        bbknn.bbknn(adata, batch_key=batch, **kwargs)
        if key_added:
            _rename_default_key(adata.uns, "neighbors", f"neighbors_{key_added}")
            if not sc.__version__.startswith("1.4"):
                _rename_default_key(adata.obsp, "distances", f"neighbors_{key_added}_distances")
                _rename_default_key(
                    adata.obsp, "connectivities", f"neighbors_{key_added}_connectivities"
                )
                adata.uns[f"neighbors_{key_added}"][
                    "distances_key"
                ] = f"neighbors_{key_added}_distances"
                adata.uns[f"neighbors_{key_added}"][
                    "connectivities_key"
                ] = f"neighbors_{key_added}_connectivities"
        else:
            _delete_backup_key(adata.uns, "neighbors")
    finally:
        _restore_obsm_key(adata, "X_pca", use_rep)


def run_phate(
    adata,
    use_rep="X",
    key_added=None,
    knn=5,
    decay=40,
    t="auto",
    n_pca=100,
    random_state=0,
    verbose=False,
    **kwargs,
):
    import phate

    try:
        data = _find_rep(adata, use_rep, return_var_names=False)
    except ValueError as e:
        if use_rep in adata.uns.keys():
            data = adata.uns[use_rep]["distances"]
            kwargs["knn_dist"] = "precomputed"
        else:
            raise e
    phate_operator = phate.PHATE(
        knn=knn, decay=decay, t=t, n_pca=n_pca, random_state=random_state, verbose=verbose, **kwargs
    )
    kc_phate = phate_operator.fit_transform(data)
    slot_name = f"X_phate_{key_added}" if key_added else "X_phate"
    adata.obsm[slot_name] = kc_phate


def run_diffQuant(
    adata,
    groupby,
    condition,
    ctrl,
    outprfx,
    trmt=None,
    groups=None,
    cofactor=[],
    n_cpu=1,
    recover_counts=True,
    dry=False,
    debug=False,
):
    if not isinstance(cofactor, (tuple, list)):
        cofactor = [cofactor]
    selected_variables = [groupby, condition]
    selected_variables.extend(cofactor)

    X = adata.raw.X.copy() if adata.raw else adata.X

    if recover_counts:
        X = lognorm_to_counts(
            X, adata.obs["n_counts"].values, rounding=True, force=(recover_counts == "force")
        )
    else:
        X.data = np.round(np.expm1(X.data))

    if groups is not None and isinstance(groups, (list, tuple)):
        k_obs = adata.obs[groupby].isin(groups)
        ad = anndata.AnnData(X=X[k_obs, :], obs=adata.obs.loc[k_obs, selected_variables].copy())
    else:
        ad = anndata.AnnData(X=X, obs=adata.obs[selected_variables].copy())
    ad.var_names = adata.raw.var_names if adata.raw else adata.var_names
    ad.obs[groupby].cat.remove_unused_categories(inplace=True)

    count_matrix = pseudo_bulk(ad, groupby=groupby, FUN=np.sum).astype(np.int32)
    metadata = ad.obs.groupby(groupby).first()
    metadata.index.name = "sample"

    import tempfile

    x_fh = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False)
    meta_fh = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False)

    count_matrix.to_csv(x_fh.name, sep="\t")
    metadata.to_csv(meta_fh.name, sep="\t")

    import subprocess as sbp

    trmt_opt = f"--trmt {trmt}" if trmt else ""
    covar_opt = f'--covar {",".join(cofactor)}' if cofactor else ""
    cmd = (
        f"diffQuant -o {outprfx} --minfrac 0.99 --nCPU {n_cpu} --factor {condition} "
        f"--ctrl {ctrl} {trmt_opt} {covar_opt} {x_fh.name} {meta_fh.name}"
    )
    if dry:
        print(cmd)
        return
    try:
        job = sbp.run(cmd, shell=True, capture_output=True)
    except sbp.CalledProcessError as e:
        print(e)
    finally:
        if debug:
            print(job.stdout.decode("utf-8"))
            print(job.stderr.decode("utf-8"))

        os.remove(x_fh.name)
        os.remove(meta_fh.name)
    de_tbl = pd.read_csv(f"{outprfx}.{trmt}-vs-{ctrl}.txt", sep="\t", index_col=0)
    return de_tbl


def run_cellphonedb(
    adata,
    groupby,
    group_size=-1,
    cpdb_path="cellphonedb",
    outpath="./cpdb",
    thread=8,
    dry=False,
    **subsample_kwargs,
):
    outdir, outname = os.path.split(outpath)
    if not outdir:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)

    import tempfile

    tmp_h5ad = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    if group_size > 0:
        adata = subsample(
            adata,
            fraction=1,
            groupby=groupby,
            max_n=group_size,
            min_n=group_size,
            **subsample_kwargs,
        )
    adata.write(tmp_h5ad.name, compression="lzf")

    tmp_obs = f"{outpath}.obs.tsv"
    adata.obs[[groupby]].to_csv(tmp_obs, sep="\t")

    import subprocess as sbp

    cmd = (
        f"{cpdb_path} method statistical_analysis --counts-data gene_name --output-format tsv"
        f" --project-name {outname} --output-path {outdir} --threads {thread}"
        f" {tmp_obs} {tmp_h5ad.name}"
    )
    try:
        if dry:
            print(cmd)
            return
        else:
            sbp.run(cmd.split())
    except sbp.CalledProcessError as e:
        print(e)
    finally:
        os.remove(tmp_h5ad.name)


def run_cellchat(
    adata,
    groupby,
    use_rep="raw",
    organism="human",
):

    X, var_names = _find_rep(adata, use_rep)

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rBase = importr("base")
    rCC = importr("CellChat")
    rMatrix = importr("Matrix")
    rPbapply = importr("pbapply")
    rPbapply.pboptions(type="none")

    cellchat_db = ro.r(f"CellChatDB.{organism}")

    if sp.issparse(X):
        X = X.tocoo()
        r_X = rMatrix.sparseMatrix(
            i=numpy2ri.py2rpy(X.col + 1),
            j=numpy2ri.py2rpy(X.row + 1),
            x=rBase.as_numeric(numpy2ri.py2rpy(X.data)),
            dims=rBase.as_integer(list(X.shape[::-1])),
            dimnames=rBase.list(
                rBase.as_character(numpy2ri.py2rpy(var_names)),
                rBase.as_character(numpy2ri.py2rpy(adata.obs_names.values)),
            ),
        )
    else:
        r_X = rBase.matrix(
            numpy2ri.py2rpy(X),
            dimnames=rBase.list(
                rBase.as_character(numpy2ri.py2rpy(var_names)),
                rBase.as_character(numpy2ri.py2rpy(adata.obs_names.values)),
            ),
        )

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_obs = ro.conversion.py2rpy(adata.obs[[groupby]])

    cellchat = rCC.createCellChat(object=r_X, meta=r_obs, group_by=groupby)

    cellchat_db_secret = rCC.subsetDB(cellchat_db, search="Secreted Signaling")
    cellchat.slots["DB"] = cellchat_db_secret
    cellchat = rCC.subsetData(cellchat)
    cellchat = rCC.identifyOverExpressedGenes(cellchat)
    cellchat = rCC.identifyOverExpressedInteractions(cellchat)
    cellchat = rCC.computeCommunProb(cellchat)
    cellchat = rCC.filterCommunication(cellchat, min_cells=10)
    cellnet = rCC.subsetCommunication(cellchat)

    cellnet = pandas2ri.rpy2py(cellnet)
    return cellnet


def run_celltype_composition_analysis(
    adata,
    sample,
    celltype,
    cat_covar=[],
    num_covar=[],
    extra_term=None,
    outprfx=None,
):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rBase = importr("base")
    rUtils = importr("utils")
    rMatrix = importr("Matrix")
    rTibble = importr("tibble")
    rSCTKR = importr("sctkr")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_obs = ro.conversion.py2rpy(adata.obs)

    ctca_result = rSCTKR.CellTypeCompositionAnalysis(
        rTibble.as_tibble(r_obs),
        sample,
        celltype,
        cat_covar,
        num_covar,
        extra_term=extra_term,
        save=outprfx,
    )

    return ctca_result


def _run_NMF(args, X=None, kwargs=None):
    from sklearn.decomposition import non_negative_factorization

    if not isinstance(kwargs, dict):
        kwargs = {}
    k, seed = args
    _, H, _ = non_negative_factorization(X, n_components=k, random_state=seed, **kwargs)
    return H


def run_cNMF(
    adata,
    K,
    use_rep="X",
    rep_type="count",
    n_iter=100,
    random_state=0,
    density_threshold=2,
    n_worker=1,
    use_cache=False,
    output_prefix="cNMF",
    **kwargs,
):
    """cNMF as in Kotliar et al, re-implemented by nh3"""
    if isinstance(K, int):
        K = [K]
    K = np.array(K).astype(int)

    default_NMF_kwargs = {
        "init": "random",
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 1e-4,
        "max_iter": 1000,
        "alpha_W": 0.0,
        "alpha_H": "same",
        "l1_ratio": 0.0,
    }
    for k, v in default_NMF_kwargs.items():
        kwargs[k] = kwargs.get(k, v)

    if "highly_variable" not in adata.var:
        raise ValueError("`adata` must have highly variable genes identified.")
    hvg = adata.var_names[adata.var.highly_variable.values]

    X, var_names = _find_rep(adata, use_rep)
    var_names = pd.Index(var_names)

    if not sp.issparse(X):
        raise ValueError("Provided X must be sparse")

    if rep_type not in ("count", "lognorm"):
        raise ValueError("`rep_type` must be 'count' or 'lognorm'.")

    # import tempfile
    from functools import partial
    from multiprocessing.pool import Pool
    import sklearn
    from sklearn.decomposition import non_negative_factorization
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.cluster import KMeans

    if rep_type == "count":
        tpm_X = sklearn.preprocessing.normalize(X, norm="l1", axis=1) * 1e6
    else:
        tpm_X = np.expm1(X) * 1e2
    norm_X = sklearn.preprocessing.scale(tpm_X, with_mean=False, axis=0)
    if np.all(var_names.isin(hvg)):
        norm_X_hvg = norm_X
    else:
        norm_X_hvg = norm_X[:, var_names.isin(hvg)]

    n_runs = K.size * n_iter
    np.random.seed(seed=random_state)
    nmf_seeds = np.random.randint(low=1, high=(2 ** 32) - 1, size=n_runs)
    n_neighbors = int(0.3 * n_iter)

    run_NMF = partial(_run_NMF, X=norm_X_hvg, kwargs=kwargs)

    for i, k in enumerate(K):
        spectral_cache = f"{output_prefix}.spectral.k{k}.combined.csv.gz"
        consensus_spectra_csv = f"{output_prefix}.spectra.k_{k}.consensus.csv.gz"
        consensus_usage_csv = f"{output_prefix}.usage.k_{k}.consensus.csv.gz"
        consensus_stats_csv = f"{output_prefix}.stats.k_{k}.consensus.csv"
        gene_spectra_csv = f"{output_prefix}.gene_spectra.k_{k}.csv.gz"
        gene_spectra_tpm_csv = f"{output_prefix}.gene_spectra_tpm.k_{k}.csv.gz"

        if use_cache:
            print(f"use cache {spectral_cache}")
            combined_spectra = pd.read_csv(spectral_cache, index_col=0)
        else:
            with Pool(n_worker) as pool:
                tasks = ((k, seed) for seed in nmf_seeds[i * n_iter : (i + 1) * n_iter])
                results = pool.map(run_NMF, tasks)
                # `combined_spectra` is a (k * n_inter, n_gene) dataframe
                combined_spectra = pd.DataFrame(
                    np.concatenate(results, axis=0),
                    index=[f"iter{n}_topic{m}" for n in range(n_iter) for m in range(k)],
                    columns=var_names[var_names.isin(hvg)],
                )
                combined_spectra.to_csv(spectral_cache)

        l2_spectra = (combined_spectra.T / np.linalg.norm(combined_spectra.values, 2, axis=1)).T
        topics_dist = pairwise_distances(l2_spectra.values)
        partition_order = np.argpartition(topics_dist, n_neighbors + 1)[:, : n_neighbors + 1]
        distance_to_nearest_neighbors = topics_dist[
            np.arange(topics_dist.shape[0])[:, None], partition_order
        ]
        local_density = pd.Series(
            distance_to_nearest_neighbors.sum(1) / (n_neighbors),
            index=l2_spectra.index,
        )
        del partition_order, distance_to_nearest_neighbors
        density_filter = local_density.values < density_threshold
        l2_spectra = l2_spectra.loc[density_filter, :]

        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans_cluster_labels = pd.Series(
            kmeans_model.fit_predict(l2_spectra) + 1, index=l2_spectra.index
        )
        median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()
        median_spectra = (median_spectra.T / median_spectra.sum(axis=1)).T
        median_spectra.to_csv(consensus_spectra_csv)

        stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric="euclidean")

        refit_kwargs = kwargs.copy()
        refit_kwargs.update(dict(H=median_spectra.values.astype(np.float32), update_H=False))
        refit_W, _, _ = non_negative_factorization(
            norm_X_hvg, n_components=k, random_state=1, **refit_kwargs
        )
        rf_usage = pd.DataFrame(
            refit_W,
            index=adata.obs_names,
            columns=median_spectra.index,
        )
        rf_usage.to_csv(consensus_usage_csv)

        rf_pred_norm_counts = rf_usage.dot(median_spectra)
        prediction_error = ((norm_X_hvg.toarray() - rf_pred_norm_counts) ** 2).values.sum()

        consensus_stats = pd.DataFrame(
            [k, density_threshold, stability, prediction_error],
            index=["k", "local_density_threshold", "stability", "prediction_error"],
            columns=["stats"],
        )
        consensus_stats.to_csv(consensus_stats_csv)

        gene_spectra = pd.DataFrame(
            sp.csr_matrix(np.linalg.pinv(rf_usage.values)).dot(norm_X).toarray(),
            index=rf_usage.columns,
            columns=var_names,
        )
        gene_spectra.to_csv(gene_spectra_csv)

        norm_usage = sklearn.preprocessing.normalize(rf_usage, norm="l1", axis=0)
        refit_kwargs.update(dict(H=norm_usage.T))
        refit_tpm_W, _, _ = non_negative_factorization(
            tpm_X.T, n_components=k, random_state=1, **refit_kwargs
        )
        gene_spectra_tpm = pd.DataFrame(
            refit_tpm_W,
            index=adata.var_names,
            columns=median_spectra.index,
        )
        gene_spectra_tpm.to_csv(gene_spectra_tpm_csv)
        print(f"k = {k} done")


def write_10x_h5(
    adata,
    outfile,
    data_type="gex",
    genome="GRCh38",
    feature_id=None,
    feature_name=None,
    feature_type=None,
    target_sets=None,
    other_feature_tags=None,
    data_version=2,
    software_version="3.0.0",
):
    """Write an AnnData to an HDF5 that Cellranger produces
    * Parameters
        + adata : AnnData
        An AnnData object
        + outfile : str
        Name of the output h5
        + data_type : str
        Data type, can be "gex" or "atac", default "gex"
        + genome : str
        Genome name, default "GRCh38"
        + feature_id : str
        Column in `var` that contains feature ids, by default "gene_ids" for
        "gex" or "peak" for "atac"
        + feature_name : str
        Column in `var` that contains feature names, by default `var_names`
        + feature_type : str
        Either column in `var` that contains feature types, or the actual
        feature type common to all features, by default "Gene Expression" for
        "gex" or "Peaks" for "atac"
        + target_sets : list of str
        A list of boolean columns in `var` that flag targeted features, only
        applicable to targeted gene expression assay
        + other_feature_tags : list of str
        Other columns in `var` to save
        + data_version : numeric
        Version of the assay, default 2
        + software_version : str
        Cellranger version, default "3.0.0"
    """
    import h5py

    DEFAULT_FEATURE_IDS = {"gex": "gene_ids", "atac": "peak"}
    DEFAULT_FEATURE_TYPES = {"gex": "Gene Expression", "atac": "Peaks"}
    DEFAULT_TAG_KEYS = {"gex": ["genome"], "atac": ["genome", "derivation"]}

    other_feature_tags = other_feature_tags or []

    n_barcode, n_feature = adata.shape

    barcode_len = max(map(len, adata.obs_names.tolist()))

    feature_type = feature_type or DEFAULT_FEATURE_TYPES[data_type]
    ftype = (
        adata.var.get(feature_type).values
        if feature_type in adata.var.columns
        else np.repeat(feature_type, n_feature)
    )
    ftype_len = max(map(len, ftype.tolist()))

    fname = adata.var.get(feature_name, adata.var_names).values
    fname_len = max(map(len, fname.tolist()))

    fid = adata.var.get(feature_id or DEFAULT_FEATURE_IDS[data_type], adata.var_names).values
    fid_len = max(map(len, fid.tolist()))

    nodes = {
        "/matrix": (h5py.Group, None),
        "/matrix/barcodes": (f"S{barcode_len}", adata.obs_names.values),
        "/matrix/data": (np.int32, adata.X.data),
        "/matrix/indices": (np.int32, adata.X.indices),
        "/matrix/indptr": (np.int32, adata.X.indptr),
        "/matrix/shape": (np.int64, np.array([n_feature, n_barcode])),
        "/matrix/features": (h5py.Group, None),
        "/matrix/features/feature_type": (f"S{ftype_len}", ftype),
        "/matrix/features/id": (f"S{fid_len}", fid),
        "/matrix/features/name": (f"S{fname_len}", fname),
        "/matrix/features/genome": (f"S{len(genome)}", np.repeat(genome, n_feature)),
    }

    if data_type == "atac":
        if "derivation" in adata.var.columns:
            other_feature_tags.append("derivation")
        else:
            nodes["/matrix/features/derivation"] = ("S1", np.repeat("", n_feature))

    if data_type == "gex" and target_sets is not None:
        if isinstance(target_sets, str):
            target_sets = [target_sets]
        valid_target_sets = []
        for tgt_set in target_sets:
            if adata.var.dtypes.get(tgt_set) == "bool":
                valid_target_sets.append(tgt_set)
            else:
                warnings.warn(f"Ignore invalid target set: {tgt_set}")
        if valid_target_sets:
            nodes["/matrix/features/target_sets"] = (h5py.Group, None)
            for tgt_set in valid_target_sets:
                nodes[f"/matrix/features/target_sets/{tgt_set}"] = (
                    np.int32,
                    np.where(adata.var.get(tgt_set).values)[0],
                )

    tag_keys_set = set(DEFAULT_TAG_KEYS.get(data_type, []))
    for ft in other_feature_tags:
        if ft in adata.var.columns:
            path = f"/matrix/features/{ft}"
            dv = adata.var.get(ft).values
            dt = dv.dtype
            if dt.kind == "O":
                dl = max(map(len, dv.tolist()))
                dt = f"S{dl}"
            nodes[path] = (dt, dv)
            tag_keys_set.add(ft)
    tag_keys = np.array(list(tag_keys_set))
    tag_len = max(map(len, tag_keys.tolist()), default=1)
    nodes["/matrix/features/_all_tag_keys"] = (f"S{tag_len}", tag_keys)

    with h5py.File(outfile, mode="w") as f:
        f.attrs["filetype"] = "matrix"
        f.attrs["software_version"] = software_version
        f.attrs["version"] = data_version
        for path, node in nodes.items():
            print(path)
            dtype, dvalue = node
            if dtype == h5py.Group:
                if path not in f:
                    f.create_group(path)
            else:
                f.create_dataset(path, data=dvalue.astype(dtype), dtype=dtype, compression="gzip")


def write_mtx(
    adata,
    fname_prefix="",
    var=["gene_ids"],
    obs=None,
    use_raw=False,
    output_version="v3",
    feature_type="Gene Expression",
):
    """Export AnnData object to mtx formt
    * Parameters
        + adata : AnnData
        An AnnData object
        + fname_prefix : str
        Prefix of the exported files. If not empty and not ending with '/' or '_',
        a '_' will be appended. Full names will be <fname_prefix>matrix.mtx(.gz),
        <fname_prefix>genes.tsv/(features.tsv.gz), <fname_prefix>barcodes.tsv(.gz)
        + var : list
        A list of extra column names to be exported to gene table, default ["gene_ids"]
        + obs : list
        A list of extra column names to be exported to barcode/cell table, default None
        + use_raw: boolean
        Whether to write `adata.raw` instead of `adata`, default False
        + output_version: str
        Write v2 or v3 Cellranger mtx outputs, default v3
        + feature_type: str
        Text added as the last column of "features.tsv.gz", only relevant when
        `output_version="v3"`
    """
    if fname_prefix and not (fname_prefix.endswith("/") or fname_prefix.endswith("_")):
        fname_prefix = fname_prefix + "_"
    if var is None:
        var = []
    if obs is None:
        obs = []
    adata_obs = adata.obs.copy()
    if use_raw:
        adata = adata.raw
    obs = list(set(obs) & set(adata_obs.columns))
    var = list(set(var) & set(adata.var.columns))

    mat = sp.coo_matrix(adata.X)
    n_obs, n_var = mat.shape
    n_entry = len(mat.data)
    header = "%%MatrixMarket matrix coordinate real general\n%\n{} {} {}\n".format(
        n_var, n_obs, n_entry
    )
    df = pd.DataFrame({"col": mat.col + 1, "row": mat.row + 1, "data": mat.data})
    if output_version == "v2":
        mtx_fname = fname_prefix + "matrix.mtx"
        gene_fname = fname_prefix + "genes.tsv"
        barcode_fname = fname_prefix + "barcodes.tsv"
    else:
        mtx_fname = fname_prefix + "matrix.mtx.gz"
        gene_fname = fname_prefix + "features.tsv.gz"
        barcode_fname = fname_prefix + "barcodes.tsv.gz"
    with open(mtx_fname, "a", encoding="utf8") as fh:
        fh.write(header)
        df.to_csv(fh, sep=" ", header=False, index=False)

    obs_df = adata_obs[obs].reset_index(level=0)
    obs_df.to_csv(barcode_fname, sep="\t", header=False, index=False)
    var_df = adata.var[var].reset_index(level=0)
    if not var:
        var_df["gene"] = var_df["index"]
    if output_version != "v2":
        var_df["feature_type"] = feature_type
    var_df.to_csv(gene_fname, sep="\t", header=False, index=False)


def write_table(
    adata,
    outdir=".",
    slots=("X", "obs", "var", "obsm", "varm", "raw.X", "raw.var"),
    fmt="tsv",
    transpose=False,
    compression=False,
    X_dtype=None,
    raw_X_dtype=None,
    obs_columns=None,
    var_columns=None,
    obsm_keys=None,
    varm_keys=None,
    raw_var_columns=None,
):
    if not outdir:
        raise ValueError("`outdir` cannot be empty")
    outdir = outdir.rstrip("/")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sep = "," if fmt == "csv" else "\t"
    suffix = ".gz" if compression else ""

    for slot in slots:
        if slot in ("X", "raw.X"):
            ad = adata.raw if slot == "raw.X" else adata
            dtype = raw_X_dtype if slot == "raw.X" else X_dtype
            if dtype is None:
                dtype = np.float32
            X = ad.X.T if transpose else ad.X
            if sp.issparse(X):
                X = X.toarray()
            if dtype in (int, np.int16, np.int32, np.int64):
                X = np.round(X)
            if transpose:
                df = pd.DataFrame(
                    X, index=ad.var_names.values, columns=adata.obs_names.values, dtype=dtype
                )
                df.index.name = "Gene"
            else:
                df = pd.DataFrame(
                    X, index=adata.obs_names.values, columns=ad.var_names.values, dtype=dtype
                )
                df.index.name = "Cell"
            df.to_csv(f"{outdir}/{slot}.{fmt}{suffix}", sep=sep)
        elif slot in ("obs", "var") and getattr(adata, slot) is not None:
            df_columns = obs_columns if slot == "obs" else var_columns
            if df_columns is None:
                df_columns = df.columns.to_list()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            df = getattr(adata, slot)[df_columns]
            df.index.name = "Cell" if slot == "obs" else "Gene"
            df.to_csv(f"{outdir}/{slot}.{fmt}{suffix}", sep=sep)
        elif slot == "raw.var" and adata.raw is not None and adata.raw.var is not None:
            df_columns = raw_var_columns
            if df_columns is None:
                df_columns = df.columns.to_list()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            df = adata.raw.var[df_columns]
            df.index.name = "Gene"
            df.to_csv(f"{outdir}/{slot}.{fmt}{suffix}", sep=sep)
        elif slot in ("obsm", "varm") and getattr(adata, slot) is not None:
            obj_keys = obsm_keys if slot == "obsm" else varm_keys
            if obj_keys is None:
                obj_keys = obj.keys()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            obj = getattr(adata, slot)
            for k in obj_keys:
                X = obj[k]
                ind = adata.obs_names.values if slot == "obsm" else adata.var_names.values
                col = [f"{k}_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, index=ind, columns=col, dtype=np.float32)
                df.index.name = "Cell" if slot == "obsm" else "Gene"
                df.to_csv(f"{outdir}/{slot}.{k}.{fmt}{suffix}", sep=sep)


def split_by_group(adata, groupby, out_type="dict"):
    if groupby not in adata.obs.columns:
        raise KeyError(f"{groupby} is not a valid obs annotation.")
    groups = sorted(list(adata.obs[groupby].unique()))
    if out_type == "dict":
        out_adatas = {}
        for grp in groups:
            out_adatas[grp] = adata[adata.obs[groupby] == grp, :].copy()
    elif out_type == "list":
        out_adatas = []
        for grp in groups:
            out_adatas.append(adata[adata.obs[groupby] == grp, :].copy())
    else:
        raise ValueError(f'{out_type}: unsupported type, choose from "dict" or "list".')
    return out_adatas


def show_obs_categories(ad, columns=None):
    if columns:
        columns = [k for k in columns if k in ad.obs.columns]
    else:
        columns = [k for k in ad.obs.columns if ad.obs[k].dtype.name == "category"]
    for k in columns:
        summary = ad.obs[k].value_counts(dropna=False, sort=False)
        print(f"{k}, {len(summary.to_dict())}")
        print(summary)
        print("\n")


def regroup(adata, groupby, regroups):
    if groupby not in adata.obs.columns:
        raise KeyError(f"{groupby} is not a valid obs annotation.")
    groups = adata.obs[groupby].astype(str)
    new_groups = groups.copy()
    for new_grp, old_grps in regroups.items():
        if isinstance(old_grps, (list, tuple)):
            for grp in old_grps:
                new_groups[groups == grp] = new_grp
        else:
            new_groups[groups == old_grps] = new_grp
    regroup_keys = [g for g in regroups.keys() if g in set(new_groups.unique())]
    new_groups = new_groups.astype("category")
    if len(regroup_keys) == len(new_groups.cat.categories):
        new_groups = new_groups.cat.reorder_categories(regroup_keys)
    return new_groups


def subsample(
    adata,
    fraction,
    groupby=None,
    min_n=0,
    max_n=10000,
    method="random",
    index_only=False,
    random_state=0,
):
    if method not in ("random", "top"):
        raise NotImplementedError(f"method={method} unsupported")
    if groupby:
        if groupby not in adata.obs.columns:
            raise KeyError(f"{groupby} is not a valid obs annotation.")
        groups = adata.obs[groupby].unique()
        sampled_obs_names = []
        for grp in groups:
            k = adata.obs[groupby] == grp
            grp_size = sum(k)
            ds_grp_size = int(min(max_n, max(np.ceil(grp_size * fraction), min(min_n, grp_size))))
            if method == "top":
                idx = np.argsort(-adata.obs.loc[k, "n_counts"].values)[0:ds_grp_size]
            else:
                np.random.seed(random_state)
                idx = np.random.choice(grp_size, ds_grp_size, replace=False)
            sampled_obs_names.extend(list(adata.obs_names[k][idx]))
    else:
        ds_size = int(adata.n_obs * fraction)
        np.random.seed(random_state)
        idx = np.random.choice(adata.n_obs, ds_size, replace=False)
        sampled_obs_names = adata.obs_names[idx]

    if index_only:
        return sampled_obs_names
    return adata[adata.obs_names.isin(sampled_obs_names)].copy()


def random_partition(
    adata,
    partition_size,
    groupby=None,
    method="random_even",
    key_added="partition_labels",
    random_state=0,
):
    np.random.seed(random_state)
    if groupby:
        if groupby not in adata.obs.columns:
            raise KeyError(f"{groupby} is not a valid obs annotation.")
        groups = adata.obs[groupby].unique()
        label_df = adata.obs[[groupby]].astype(str).rename(columns={groupby: key_added})
        for grp in groups:
            k = adata.obs[groupby] == grp
            grp_size = sum(k)
            n_partition = max(np.round(grp_size / partition_size).astype(int), 1)
            if method == "random":
                part_idx = np.random.randint(low=0, high=n_partition, size=grp_size)
            elif method == "random_even":
                part_sizes = list(map(len, np.array_split(np.arange(grp_size), n_partition)))
                part_idx = np.repeat(np.arange(n_partition), part_sizes)
                np.random.shuffle(part_idx)
            else:
                raise NotImplementedError(method)
            label_df.loc[k, key_added] = [f"{grp},{i}" for i in part_idx]
        adata.obs[key_added] = label_df[key_added]
    else:
        n_partition = max(np.round(adata.n_obs / partition_size).astype(int), 1)
        if method == "random":
            part_idx = np.random.randint(low=0, high=n_partition, size=adata.n_obs)
        elif method == "random_even":
            part_sizes = list(map(len, np.array_split(np.arange(adata.n_obs), n_partition)))
            part_idx = np.repeat(np.arange(n_partition), part_sizes)
            np.random.shuffle(part_idx)
        else:
            raise NotImplementedError(method)
        adata.obs[key_added] = part_idx.astype(str)


def dummy_to_categorical(mat, random_state=0):
    """Convert a sparse dummy matrix into a list of category indices, when a row
    has multiple entries, randomly assign to one of them.

    *Parameters
        + mat: csr_matrix
        A csr_matrix of ones, with cells on the rows and nhoods on the columns
        + random_state: int
        Seed for numpy RNG
    """
    np.random.seed(random_state)
    nrow = mat.shape[0]
    nhoods = []
    for i in range(nrow):
        k = (mat[i, ] == 1).indices
        if k.size == 1:
            idx = k[0]
        elif k.size > 1:
            idx = np.random.choice(k, 1)[0]
        else:
            idx = -1
        nhoods.append(idx)
    return nhoods


def pseudo_bulk(adata, groupby, use_rep="X", FUN=np.mean):
    """Make pseudo bulk data from grouped sc data"""
    grouping = adata.obs[groupby]
    if grouping.dtype.name != "category":
        grouping = grouping.astype("category")
    group_values = grouping.values
    groups = grouping.cat.categories.values
    n_group = len(groups)

    x, features = _find_rep(adata, use_rep)

    if sp.issparse(x):
        summarised = np.zeros((n_group, x.shape[1]))
        for i, grp in enumerate(groups):
            k_grp = group_values == grp
            summarised[i] = FUN(x[k_grp, :], axis=0)
    else:
        summarised = npg.aggregate_np(grouping.cat.codes, x, FUN, axis=0)
    return pd.DataFrame(summarised.T, columns=groups, index=features)


def summarise_expression_by_group(adata, groupby, genes=None, use_rep="X"):
    grouping = adata.obs[groupby]
    if grouping.dtype.name != "category":
        grouping = grouping.astype("category")
    group_values = grouping.values
    groups = grouping.cat.categories.values
    n_group = len(groups)

    x, features = _find_rep(adata, use_rep)
    if genes is not None:
        k_gene = np.in1d(features, genes).nonzero()[0]
        x = x[:, k_gene]
        features = features[k_gene]
    n_gene = features.size

    if sp.issparse(x):
        summarised_mean = np.zeros((n_group, n_gene))
        summarised_frac = np.zeros((n_group, n_gene))
        for i, grp in enumerate(groups):
            k_grp = group_values == grp
            summarised_mean[i] = np.mean(x[k_grp, :], axis=0)
            summarised_frac[i] = np.mean(x[k_grp, :] > 0, axis=0)
    else:
        summarised_mean = npg.aggregate_np(grouping.cat.codes, x, np.mean, axis=0)
        summarised_frac = npg.aggregate_np(grouping.cat.codes, x > 0, np.mean, axis=0)
    return pd.DataFrame(
        {
            "gene": np.repeat(features, n_group),
            "group": np.tile(groups, n_gene),
            "frac": summarised_frac.T.flatten(),
            "avg": summarised_mean.T.flatten(),
        }
    )


def score_msigdb_genesets(
    adata, groupby, msigdb, use_rep="raw", pattern=r"^HALLMARK_", use_pseudobulk=False, **pb_kw
):
    def read_msigdb(msigdb, pattern):
        genesets = {}
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        with open(msigdb, "r", encoding="utf8") as fh:
            for line in fh:
                fields = line.rstrip().split("\t")
                geneset_name = fields[0]
                geneset = fields[2:]
                if re.search(pattern, geneset_name):
                    genesets[geneset_name] = geneset
        print(f"read {len(genesets)} genesets")
        return genesets

    genesets = read_msigdb(msigdb, pattern)
    if use_pseudobulk:
        X = pseudo_bulk(adata, groupby, use_rep=use_rep, **pb_kw).T
        ad = anndata.AnnData(X=X)
    else:
        X = _find_rep(adata, use_rep, return_var_names=False)
        ad = anndata.AnnData(X, obs=adata.obs[[groupby]])

    for geneset_name, geneset in genesets.items():
        sc.tl.score_genes(
            ad,
            geneset,
            ctrl_size=max(50, len(geneset)),
            use_raw=(use_rep == "raw"),
            score_name=geneset_name,
        )

    del genesets

    return ad


def write_cellxgene_object(adata, output, **kwargs):
    ad = restore_adata(adata, restore_type="norm", use_raw=adata.raw, **kwargs)
    ad.write(output, compression="lzf")
    return ad
