"""
High-level pipeline functions
"""

from typing import Union, Optional, Any, Mapping, Callable, Tuple

from packaging import version
import re
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata
import scanpy as sc
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from ._hvg import hvg
from ._pca import pca
from ._neighbors import neighbors
from ._umap import umap
from ._fdg import fdg
from ._tsne import tsne
from ._leiden import leiden
from ._diffmap import diffmap
from ._plot import (
    clear_colors,
    plot_qc_violin,
    plot_qc_scatter,
)
from ._annot import (
    LR_train,
    LR_predict,
)
from ._utils import (
    restore_adata,
    sc_warn,
    cross_table,
)
from ._markers import calc_marker_stats, filter_marker_stats


class QcLowPassError(ValueError):
    pass


def calculate_qc(
    adata,
    flags={"mito": r"^MT-", "ribo": r"^RP[LS]", "hb": r"^HB"},
    extra_flags={},
    flag_only=False,
    suffix="",
    **kwargs,
) -> None:
    """
    Calculate quality control (QC) metrics for an AnnData object. The object is
    modified in-place.

    Args:
        adata: AnnData object to calculate QC metrics for.

        flags: Dictionary of QC flags and regular expression patterns to match
        gene names against.

        extra_flags: Additional QC flags and patterns to add to `flags`.

        flag_only: If True, only calculate QC flags and do not calculate other
        metrics.

        suffix: Suffix to append to QC metric names.

        **kwargs: Additional keyword arguments to pass to
        *`scanpy.pp.calculate_qc_metrics`.

    Returns:
        None.

    Raises:
        None.

    Examples:
        >>> import anndata
        >>> import numpy as np
        >>> import pandas as pd
        >>> import scanpy as sc
        >>> from my_module import calculate_qc
        >>> adata = anndata.AnnData(
        ...     X=np.random.rand(100, 100),
        ...     obs=pd.DataFrame(index=[f"cell{i}" for i in range(100)]),
        ...     var=pd.DataFrame(index=[f"gene{i}" for i in range(100)]),
        ... )
        >>> calculate_qc(adata)

    """
    if extra_flags:
        for k, v in extra_flags.items():
            flags[k] = v

    qc_vars = []
    for flag, pattern in flags.items():
        if flag not in adata.var.columns and pattern:
            pat = re.compile(pattern)
            adata.var[flag] = np.array(
                [bool(pat.search(g)) for g in adata.var_names]
            )
        if flag in adata.var.columns:
            qc_vars.append(flag)

    if flag_only:
        return

    qc_vars.extend([flag for flag in extra_flags if flag in adata.var.columns])

    qc_tbls = sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, percent_top=[50], **kwargs
    )

    adata.obs[f"n_counts{suffix}"] = qc_tbls[0]["total_counts"].values
    adata.obs[f"log1p_n_counts{suffix}"] = np.log1p(
        adata.obs[f"n_counts{suffix}"]
    )
    adata.obs[f"n_genes{suffix}"] = qc_tbls[0]["n_genes_by_counts"].values
    adata.obs[f"log1p_n_genes{suffix}"] = np.log1p(
        adata.obs[f"n_genes{suffix}"]
    )
    for metric in qc_vars:
        adata.obs[f"percent_{metric}{suffix}"] = qc_tbls[0][
            f"pct_counts_{metric}"
        ].values
        adata.obs[f"n_counts_{metric}{suffix}"] = (
            qc_tbls[0][f"pct_counts_{metric}"].values
            * qc_tbls[0]["total_counts"].values
            / 100
        )
    adata.obs[f"percent_top50{suffix}"] = qc_tbls[0][
        "pct_counts_in_top_50_genes"
    ].values
    adata.var[f"n_counts{suffix}"] = qc_tbls[1]["total_counts"].values
    adata.var[f"n_cells{suffix}"] = qc_tbls[1]["n_cells_by_counts"].values


def generate_qc_clusters(
    ad,
    metrics,
    aux_ad=None,
    n_pcs=None,
    n_neighbors=None,
    res=0.2,
    return_aux=False,
):
    if aux_ad is None:
        n_pcs = max(2, len(metrics) - 2) if n_pcs is None else n_pcs
        n_neighbors = min(max(5, int(ad.n_obs / 500)), 10)
        aux_ad = anndata.AnnData(
            ad.obs[metrics].values,
            obs=ad.obs.copy(),
            var=pd.DataFrame(index=["var_" + m for m in metrics]),
        )
        sc.pp.scale(aux_ad)
        sc.pp.pca(aux_ad, n_comps=n_pcs)
        sc.pp.neighbors(aux_ad, n_neighbors=n_neighbors)
        sc.tl.umap(aux_ad, min_dist=0.1)
    sc.tl.leiden(aux_ad, resolution=res, key_added="qc_cluster")
    ad.obs["qc_cluster"] = aux_ad.obs.qc_cluster
    ad.obsm["X_umap_qc"] = aux_ad.obsm["X_umap"]
    if return_aux:
        return aux_ad


def _scale_factor(x):
    xmin = np.min(x)
    xmax = np.max(x)
    return 5.0 / (xmax - xmin)


def fit_gaussian(
    x,
    n=10,
    threshold=0.05,
    xmin=None,
    xmax=None,
    plot=False,
    nbins=500,
    hist_bins=100,
):
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    gmm = GaussianMixture(n_components=n, random_state=0)
    x_fit = x[(x >= xmin) & (x <= xmax)]
    f = _scale_factor(x_fit)
    x_fit = x_fit * f
    gmm.fit(x_fit.reshape(-1, 1))
    while not gmm.converged_:
        gmm.fit(x_fit.reshape(-1, 1), warm_start=True)
    x0 = np.linspace(x.min(), x.max(), num=nbins)
    y_pdf = np.zeros((n, nbins))
    y_cdf = np.zeros((n, nbins))
    for i in range(n):
        y_pdf[i] = (
            norm.pdf(
                x0 * f,
                loc=gmm.means_[i, 0],
                scale=n * gmm.covariances_[i, 0, 0],
            )
            * gmm.weights_[i]
        )
        y_cdf[i] = (
            norm.cdf(
                x0 * f,
                loc=gmm.means_[i, 0],
                scale=n * gmm.covariances_[i, 0, 0],
            )
            * gmm.weights_[i]
        )
    y0 = y_pdf.sum(axis=0)
    x_peak = x0[np.argmax(y0)]
    try:
        x_left = x0[(y0 < threshold) & (x0 < x_peak)].max()
    except Exception:
        sc_warn("Failed to find lower bound, using min value instead.")
        x_left = x0.min()
    try:
        x_right = x0[(y0 < threshold) & (x0 > x_peak)].min()
    except Exception:
        sc_warn("Failed to find upper bound, using max value instead.")
        x_right = x0.max()
    if plot:
        fig, ax1 = plt.subplots()
        _ = ax1.hist(x, bins=hist_bins)
        ax2 = ax1.twinx()
        ax2.plot(x0, y0, c="k")
        ax2.hlines(
            y=threshold,
            xmin=x.min(),
            xmax=x.max(),
            linewidth=1,
            linestyle="dotted",
        )
        ax2.vlines(
            x=[xmin, xmax],
            ymin=y0.min(),
            ymax=y0.max(),
            linewidth=1,
            linestyle="dashed",
        )
        if not np.isnan(x_left):
            ax2.vlines(x=x_left, ymin=y0.min(), ymax=y0.max(), linewidth=1)
        if not np.isnan(x_right):
            ax2.vlines(x=x_right, ymin=y0.min(), ymax=y0.max(), linewidth=1)
    return x_left, x_right, gmm


def filter_qc_outlier2(adata, metrics=None, force=False):
    default_metric_params = {
        "n_counts": (1000, None, "log", "min_only", 0.1),
        "n_genes": (100, None, "log", "min_only", 0.1),
        "percent_mito": (0.01, 20, "log", "max_only", 0.1),
        "percent_ribo": (0, 100, "log", "both", 0.1),
        "percent_hb": (None, 1, "log", "max_only", 0.1),
        "percent_soup": (None, 5, "log", "max_only", 0.1),
        "percent_spliced": (50, 97.5, "log", "both", 0.1),
        "scrublet_score": (None, 0.3, "linear", "max_only", 0.95),
    }
    if metrics is None:
        metric_params = default_metric_params
    elif isinstance(metrics, (list, tuple)):
        metric_params = {
            k: v for k, v in default_metric_params.items() if k in metrics
        }
    elif isinstance(metrics, dict) and all(
        k in adata.obs.columns and len(v) == 5 for k, v in metrics.items()
    ):
        metric_params = metrics
    else:
        raise ValueError(
            "`metrics` must be a list/tuple of metric names or a dict of"
            " <name>: [<min>, <max>, <scale>, <sidedness>, <min_pass_rate>]"
        )

    n_obs = adata.n_obs

    pass_filter = {}
    for m, v in metric_params.items():
        min_x, max_x, scale, side, min_pass_rate = v
        if m not in adata.obs.columns:
            continue
        x = adata.obs[m].values.astype(np.float32)
        if scale == "log":
            x = np.log1p(x)
            min_x = np.log1p(min_x) if min_x is not None else None
            max_x = np.log1p(max_x) if max_x is not None else None
        try:
            x_low, x_high, _ = fit_gaussian(x, xmin=min_x, xmax=max_x)
        except ValueError:
            x_low = min_x if min_x is not None else x.min()
            x_high = max_x if max_x is not None else x.max()
        if side == "min_only":
            x_high = x.max()
        elif side == "max_only":
            x_low = x.min()
        else:
            pass
        m_pass = (x_low <= x) & (x <= x_high)
        if m_pass.sum() < n_obs * min_pass_rate and not force:
            if side == "min_only":
                x_low = min_x
                m_pass = min_x <= x
            elif side == "max_only":
                x_high = max_x
                m_pass = x <= max_x
            else:
                x_low, x_high = min_x, max_x
                m_pass = (min_x <= x) & (x <= max_x)
        pass_filter[m] = m_pass

        x_low_str = x_low if scale == "linear" else np.expm1(x_low)
        x_high_str = x_high if scale == "linear" else np.expm1(x_high)
        print(
            f"{m}: [{x_low_str}, {x_high_str}], {pass_filter[m].sum()}/{n_obs} passed"
        )

    all_passed = np.ones(n_obs).astype(bool)
    for m, k_pass in pass_filter.items():
        all_passed = all_passed & k_pass
    print(f"{all_passed.sum()}/{n_obs} pass")
    return all_passed


def filter_qc_outlier(
    adata,
    metrics=[
        "n_counts",
        "n_genes",
        "percent_mito",
        "percent_ribo",
        "percent_hb",
        "percent_top50",
    ],
    min_count=1000,
    min_gene=100,
    min_mito=0.01,
    max_mito=20,
    min_ribo=0,
    max_ribo=100,
    max_hb=1.0,
    min_top50=0,
    max_top50=100,
    min_pass_rate=0.6,
    onesided=False,
    force=False,
):
    k_pass = np.ones(adata.n_obs).astype(bool)

    if "n_counts" in metrics:
        try:
            x_low, x_high, _ = fit_gaussian(
                adata.obs["log1p_n_counts"].values, xmin=np.log1p(min_count)
            )
        except ValueError:
            x_low, x_high = (
                np.log1p(min_count),
                adata.obs["log1p_n_counts"].max(),
            )
        x_low = max(np.log1p(250), x_low)
        min_count = int(np.expm1(x_low))
        max_count = (
            adata.obs["n_counts"].max() if onesided else int(np.expm1(x_high))
        )
        k_count = (adata.obs["n_counts"] >= min_count) & (
            adata.obs["n_counts"] <= max_count
        )
        print(
            f"n_counts: [{min_count}, {max_count}], {k_count.sum()}/{adata.n_obs} pass"
        )
        if k_count.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("n_counts")
        k_pass = k_pass & k_count

    if "n_genes" in metrics:
        try:
            x_low, x_high, _ = fit_gaussian(
                adata.obs["log1p_n_genes"].values, xmin=np.log1p(min_gene)
            )
        except ValueError:
            x_low, x_high = np.log1p(min_gene), adata.obs["log1p_n_genes"].max()
        x_low = max(np.log1p(50), x_low)
        min_gene = int(np.expm1(x_low))
        max_gene = (
            adata.obs["n_genes"].max() if onesided else int(np.expm1(x_high))
        )
        k_gene = (adata.obs["n_genes"] >= min_gene) & (
            adata.obs["n_genes"] <= max_gene
        )
        print(
            f"n_genes: [{min_gene}, {max_gene}], {k_gene.sum()}/{adata.n_obs} pass"
        )
        if k_gene.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("n_genes")
        k_pass = k_pass & k_gene

    if "percent_mito" in metrics:
        if (adata.obs["percent_mito"].values > 0).sum() > 0:
            x_low, x_high, _ = fit_gaussian(
                np.log1p(adata.obs["percent_mito"].values),
                xmin=np.log1p(min_mito),
                xmax=np.log1p(max_mito),
            )
            max_mito = np.expm1(x_high)
        k_mito = adata.obs["percent_mito"] <= max_mito
        print(
            f"percent_mito: [0, {max_mito}], {k_mito.sum()}/{adata.n_obs} pass"
        )
        if k_mito.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("percent_mito")
        k_pass = k_pass & k_mito

    if "percent_ribo" in metrics:
        x_low, x_high, _ = fit_gaussian(
            np.log1p(adata.obs["percent_ribo"].values),
            xmin=np.log1p(min_ribo),
            xmax=np.log1p(max_ribo),
        )
        min_ribo = np.expm1(x_low)
        max_ribo = np.expm1(x_high)
        k_ribo = (adata.obs["percent_ribo"] >= min_ribo) & (
            adata.obs["percent_ribo"] <= max_ribo
        )
        print(
            f"percent_ribo: [{min_ribo}, {max_ribo}], {k_ribo.sum()}/{adata.n_obs} pass"
        )
        if k_ribo.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("percent_ribo")
        k_pass = k_pass & k_ribo

    if "percent_hb" in metrics:
        k_hb = adata.obs["percent_hb"] <= max_hb
        print(f"percent_hb: [0, {max_hb}], {k_hb.sum()}/{adata.n_obs} pass")
        k_pass = k_pass & k_hb
        if k_hb.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("percent_hb")

    if "percent_top50" in metrics:
        x_low, x_high, _ = fit_gaussian(
            adata.obs["percent_top50"].values, xmin=min_top50, xmax=max_top50
        )
        max_top50 = x_high
        min_top50 = x_low
        k_top50 = (adata.obs["percent_top50"] <= max_top50) & (
            adata.obs["percent_top50"] >= min_top50
        )
        print(
            f"percent_top50: [{min_top50}, {max_top50}], {k_top50.sum()}/{adata.n_obs} pass"
        )
        if k_top50.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("percent_top50")
        k_pass = k_pass & k_top50

    if "scrublet_score_z" in metrics:
        x_low, x_high, _ = fit_gaussian(
            adata.obs["scrublet_score_z"].values,
            n=10,
            xmin=-99,
            xmax=10,
            threshold=0.05,
        )
        max_sz = x_high
        min_sz = x_low
        k_sz = adata.obs["scrublet_score_z"] <= max_sz
        print(
            f"scrublet_score_z: [{min_sz}, {max_sz}], {k_sz.sum()}/{adata.n_obs} pass"
        )
        if k_sz.sum() < adata.n_obs * min_pass_rate and not force:
            raise QcLowPassError("scrublet_score_z")
        k_pass = k_pass & k_sz

    print(f"{k_pass.sum()}/{adata.n_obs} pass")
    return k_pass


def find_good_qc_cluster(ad, metrics=None, threshold=0.5, key_added=""):
    key_fqo2 = key_added + ("_" if key_added else "") + "fqo2"
    ad.obs[key_fqo2] = filter_qc_outlier2(ad, metrics=metrics)

    if ad.obs[key_fqo2].sum() == 0:
        ad.obs[key_fqo2] = (
            (ad.obs.n_counts >= metrics["n_counts"][0])
            & (ad.obs.n_genes >= metrics["n_genes"][0])
            & (ad.obs.percent_mito < metrics["percent_mito"][1])
        )

    if ad.obs[key_fqo2].astype(bool).sum() == 0:
        good_qc_clusters = []
    else:
        good_qc_clusters = (
            pd.crosstab(
                ad.obs.qc_cluster,
                ad.obs[key_fqo2].astype("category"),
                normalize="index",
            )
            .where(lambda x: x[1] >= threshold)
            .dropna()
            .index.tolist()
        )

    key_added = key_added if key_added else "good_qc_clusters"
    ad.obs[key_added] = ad.obs["qc_cluster"].isin(good_qc_clusters)


def get_good_sized_batch(batches, min_size=10):
    x = batches.value_counts()
    return x.index[x >= min_size].to_list()


def simple_default_pipeline(
    adata: anndata.AnnData,
    filter_only: bool = False,
    post_filter_only: bool = False,
    norm_only: bool = False,
    post_norm_only: bool = False,
    hvg_only: bool = False,
    post_hvg_only: bool = False,
    pca_only: bool = False,
    post_pca_only: bool = False,
    do_clustering: bool = False,
    zero_center: bool = None,
    do_combat: bool = False,
    batch: Union[str, Union[list, tuple], None] = None,
    batch_method: Optional[str] = "harmony",
    random_state: int = 0,
    clustering_resolution: Union[float, list, tuple] = [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ],
    use_gpu: Union[bool, str] = False,
    use_hvg: Union[list, tuple, None] = None,
    filter_kw: dict = {},
    hvg_kw: dict = {},
    rgs_kw: dict = {},
    pca_kw: dict = {},
    nb_kw: dict = {},
    umap_kw: dict = {},
    hm_kw: dict = {},
    bk_kw: dict = {},
):
    if version.parse(sc.__version__) < version.parse("1.6"):
        use_gpu = False
    if not (post_filter_only or post_norm_only or post_pca_only):
        if not np.all(
            pd.Series(
                [
                    "n_counts",
                    "n_genes",
                    "percent_mito",
                    "percent_ribo",
                    "percent_hb",
                    "percent_top50",
                ]
            ).isin(list(adata.obs.columns))
        ):
            calculate_qc(adata)
        if (adata.obs["n_counts"] == 0).sum() > 0:
            adata = adata[adata.obs["n_counts"] > 0].copy()
        k_cell = filter_qc_outlier(adata, **filter_kw)
        if batch:
            if isinstance(batch, (list, tuple)):
                for b in batch:
                    if b in adata.obs.columns:
                        batches = get_good_sized_batch(adata.obs.loc[k_cell, b])
                        k_cell = k_cell & adata.obs[b].isin(batches)
            elif isinstance(batch, str):
                batches = get_good_sized_batch(adata.obs.loc[k_cell, batch])
                k_cell = k_cell & adata.obs[batch].isin(batches)
            else:
                raise ValueError("Invalid type of `batch`")
        adata = adata[k_cell, :].copy()
        if filter_only:
            return adata

    subset_hvg = hvg_kw.get("subset", False)
    if not (post_norm_only or post_pca_only):
        if not use_hvg and hvg_kw.get("flavor", "seurat") == "seurat_v3":
            aux_ad = anndata.AnnData(
                X=adata.X, obs=adata.obs.copy(), var=adata.var.copy()
            )
            if not sc._utils.check_nonnegative_integers(aux_ad.X):
                aux_ad.X = np.round(aux_ad.X)
                n_cells = (adata.layers["counts"] > 0).sum(axis=0)
            n_cells = n_cells.A1 if sp.issparse(aux_ad.X) else n_cells
            k_gene = n_cells >= 10
            if (~k_gene).sum() > 0:
                aux_ad = aux_ad[:, k_gene].copy()
            hvg_kw["subset"] = False
            hvg(aux_ad, **hvg_kw)
            use_hvg = aux_ad.var_names[aux_ad.var.highly_variable].to_list()
            adata.uns["hvg"] = aux_ad.uns["hvg"]
            del aux_ad
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if norm_only:
            return adata

    if not post_pca_only:
        if adata.raw is None:
            adata.raw = adata
        else:
            adata.X = adata.raw.X[
                :, adata.raw.var_names.isin(adata.var_names)
            ].copy()
        if "n_counts" not in adata.var.keys():
            n_counts = np.expm1(adata.X).sum(axis=0)
            adata.var["n_counts"] = (
                n_counts.A1 if sp.issparse(adata.X) else n_counts
            )
        n_cells = (adata.X > 0).sum(axis=0)
        adata.var["n_cells"] = n_cells.A1 if sp.issparse(adata.X) else n_cells
        k_gene = adata.var["n_cells"] >= 3
        if (~k_gene).sum() > 0:
            adata = adata[:, k_gene].copy()
        if batch and do_combat:
            sc.pp.combat(adata, key=batch)
        if use_hvg:
            adata.var["highly_variable"] = adata.var_names.isin(use_hvg)
            if subset_hvg:
                adata = adata[adata.var["highly_variable"]].copy()
        elif hvg_kw.get("flavor", "seurat") != "seurat_v3":
            hvg(adata, **hvg_kw)
        if hvg_only:
            return adata
        zero_center = (
            (adata.n_obs <= 20000) if zero_center is None else zero_center
        )
        if zero_center or rgs_kw:
            adata = adata[:, adata.var.highly_variable.values].copy()
        if rgs_kw:
            sc.pp.regress_out(adata, **rgs_kw)
        sc.pp.scale(adata, zero_center=zero_center, max_value=10)
        n_comps = min(30, adata.n_obs - 1, adata.var.highly_variable.sum() - 1)
        if n_comps < 2:
            raise ValueError("n_obs or n_vars too small for pca calculation")
        pca(
            adata,
            n_comps=n_comps,
            zero_center=zero_center,
            svd_solver="arpack",
            use_highly_variable=True,
            **pca_kw,
        )
        if pca_only:
            return adata

    n_neighbors = nb_kw.pop("n_neighbors", 15)
    n_pcs = nb_kw.pop("n_pcs", 20)
    if batch:
        if batch_method == "bbknn":
            from ._utils import run_bbknn

            key_added = bk_kw.pop("key_added", "bk")
            if use_gpu == "all":
                bk_kw["metric"] = "euclidean"
            run_bbknn(adata, batch=batch, key_added=key_added, **bk_kw)
            umap(
                adata,
                method="rapids" if use_gpu == "all" else None,
                use_graph=f"neighbors_{key_added}",
                key_added=key_added,
                random_state=random_state,
                **umap_kw,
            )
            if do_clustering:
                leiden(
                    adata,
                    flavor="rapids" if use_gpu else None,
                    use_graph=f"neighbors_{key_added}",
                    resolution=clustering_resolution,
                    key_added=key_added,
                )
        else:
            from ._utils import run_harmony

            key_added = hm_kw.pop("key_added", "hm")
            run_harmony(adata, batch=batch, key_added=key_added, **hm_kw)
            neighbors(
                adata,
                method="rapids" if use_gpu else "umap",
                use_rep=f"X_pca_{key_added}",
                key_added=key_added,
                n_pcs=n_pcs,
                n_neighbors=n_neighbors,
            )
            umap(
                adata,
                method="rapids" if use_gpu == "all" else None,
                use_graph=f"neighbors_{key_added}",
                key_added=key_added,
                random_state=random_state,
                **umap_kw,
            )
            if do_clustering:
                leiden(
                    adata,
                    flavor="rapids" if use_gpu else None,
                    use_graph=f"neighbors_{key_added}",
                    resolution=clustering_resolution,
                    key_added=key_added,
                )
    else:
        sc.pp.neighbors(
            adata, use_rep="X_pca", n_pcs=n_pcs, n_neighbors=n_neighbors
        )
        umap(
            adata,
            method="rapids" if use_gpu == "all" else None,
            random_state=random_state,
            **umap_kw,
        )
        if do_clustering:
            leiden(
                adata,
                flavor="rapids" if use_gpu else None,
                resolution=clustering_resolution,
            )

    return adata


def recluster_subset(
    adata, groupby, groups, res, new_key, ad_aux=None, **kwargs
):
    kwargs["post_norm_only"] = True
    kwargs["do_clustering"] = False
    if isinstance(res, (list, tuple)):
        res = res[0]
    if "batch" in kwargs:
        graph = (
            "neighbors_bk"
            if kwargs["batch_method"] == "bbknn"
            else "neighbors_hm"
        )
    else:
        graph = "neighbors"
    k_groups = adata.obs[groupby].isin(groups)
    if ad_aux is None:
        ad = restore_adata(
            adata[k_groups].copy(), restore_type="norm", use_raw=True
        )
        ad_aux = simple_default_pipeline(ad, **kwargs)
        return_ad = True
    else:
        return_ad = False
    leiden(ad_aux, use_graph=graph, resolution=res, key_added="aux")
    adata.obs[new_key] = adata.obs[groupby].astype(str)
    adata.obs.loc[k_groups, new_key] = (
        "_".join(groups) + "," + ad_aux.obs["leiden_aux"].astype(str)
    )
    if return_ad:
        return ad_aux


def auto_zoom_in(
    ad,
    use_graph,
    groupby,
    restrict_to=None,
    start_index=1,
    min_res=0.1,
    max_res=0.7,
    leiden_kw={},
    marker_kw={"max_next_frac": 0.3},
):
    i = start_index
    if isinstance(restrict_to, (list, tuple)):
        groups = restrict_to
    else:
        groups = ad.obs[groupby].cat.categories
    prev_grp_key = groupby
    for grp in groups:
        res = min_res
        new_grp_key = f"leiden_split{i}"
        subclustered = False
        print([grp, i], end=" ")
        while True:
            leiden(
                ad,
                use_graph=use_graph,
                restrict_to=(prev_grp_key, [grp]),
                resolution=res,
                key_added=f"split{i}",
                **leiden_kw,
            )
            if (
                ad.obs[new_grp_key].cat.categories.size
                > ad.obs[prev_grp_key].cat.categories.size
            ):
                subclustered = True
                break
            elif res < max_res:
                res += 0.1
            else:
                del ad.obs[new_grp_key]
                break
        if subclustered:
            print("find_marker", end=" ")
            new_groups = ad.obs[new_grp_key].cat.categories[
                ~ad.obs[new_grp_key].cat.categories.isin(
                    ad.obs[prev_grp_key].cat.categories
                )
            ]
            mkst = calc_marker_stats(ad, groupby=new_grp_key)
            mks = filter_marker_stats(mkst[2], **marker_kw)
            if (
                new_groups.isin(mks.top_frac_group.unique()).sum()
                == new_groups.size
            ):
                prev_grp_key = new_grp_key
                i += 1
                print("done", end="")
            else:
                del ad.obs[new_grp_key]
                print("failed", end="")
        print("")
    if new_grp_key in ad.obs.columns:
        print(new_grp_key)
    else:
        print(f"leiden_split{i-1}")


def custom_pipeline(
    adata,
    qc_only=False,
    plot=True,
    batch=None,
    filter_params={
        "min_genes": 200,
        "min_cells": 3,
        "max_counts": 25000,
        "max_mito": 20,
        "min_mito": 0,
    },
    norm_params={"target_sum": 1e4, "fraction": 0.9},
    combat_args={"key": None},
    hvg_params={"flavor": "seurat", "by_batch": None},
    scale_params={"max_value": 10},
    pca_params={
        "n_comps": 50,
        "svd_solver": "arpack",
        "use_highly_variable": True,
    },
    harmony_params={"batch": None, "theta": 2.0},
    nb_params={"n_neighbors": 15, "n_pcs": 20},
    umap_params={},
    tsne_params={},
    diffmap_params={"n_comps": 15},
    leiden_params={
        "resolution": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    fdg_params={"layout": "fa"},
):
    """
    Scanpy pipeline
    """
    if qc_only:
        calculate_qc(adata)
        if plot:
            plot_qc_violin(
                adata,
                groupby=batch,
                one_per_line=adata.obs[batch].cat.categories.size > 10,
            )
            plot_qc_scatter(adata, use_hexbin=True)
    else:
        if filter_params is not None and isinstance(filter_params, dict):
            if "min_genes" in filter_params:
                sc.pp.filter_cells(adata, min_genes=filter_params["min_genes"])
            if "min_cells" in filter_params:
                sc.pp.filter_genes(adata, min_cells=filter_params["min_cells"])
            if "min_counts" in filter_params:
                k = adata.obs["n_counts"] >= filter_params["min_counts"]
                adata._inplace_subset_obs(k)
            if "max_counts" in filter_params:
                k = adata.obs["n_counts"] <= filter_params["max_counts"]
                adata._inplace_subset_obs(k)
            if "min_mito" in filter_params:
                k = adata.obs["percent_mito"] >= filter_params["min_mito"]
                adata._inplace_subset_obs(k)
            if "max_mito" in filter_params:
                k = adata.obs["percent_mito"] <= filter_params["max_mito"]
                adata._inplace_subset_obs(k)
            if "min_ribo" in filter_params:
                k = adata.obs["percent_ribo"] >= filter_params["min_ribo"]
                adata._inplace_subset_obs(k)
            if "max_ribo" in filter_params:
                k = adata.obs["percent_ribo"] <= filter_params["max_ribo"]
                adata._inplace_subset_obs(k)
            if "min_hb" in filter_params:
                k = adata.obs["percent_hb"] >= filter_params["min_hb"]
                adata._inplace_subset_obs(k)
            if "max_hb" in filter_params:
                k = adata.obs["percent_hb"] <= filter_params["max_hb"]
                adata._inplace_subset_obs(k)
            if "counts" not in adata.layers.keys():
                adata.layers["counts"] = adata.X
        if norm_params is not None and isinstance(norm_params, dict):
            if "counts" in adata.layers.keys():
                adata.X = adata.layers["counts"]
            sc.pp.normalize_total(adata, **norm_params)
            sc.pp.log1p(adata)
            adata.raw = adata
        if combat_args is not None and (
            isinstance(combat_args, dict)
            and combat_args.get("key", None)
            and combat_args["key"] in adata.obs.keys()
        ):
            adata.layers["X"] = adata.X
            adata.X = adata.raw.X
            sc.pp.combat(adata, **combat_args)
        if hvg_params is not None and isinstance(hvg_params, dict):
            hvg(adata, **hvg_params)
        if scale_params is not None and isinstance(scale_params, dict):
            sc.pp.scale(adata, **scale_params)
        if pca_params is not None and isinstance(pca_params, dict):
            pca(adata, **pca_params)
        if harmony_params is not None and (
            isinstance(harmony_params, dict)
            and harmony_params.get("batch", None)
        ):
            from ._utils import run_harmony

            run_harmony(adata, **harmony_params)
        if nb_params is not None and isinstance(nb_params, dict):
            neighbors(adata, **nb_params)
        if umap_params is not None and isinstance(umap_params, dict):
            umap(adata, **umap_params)
        if tsne_params is not None and isinstance(tsne_params, dict):
            tsne(adata, **tsne_params)
        if diffmap_params is not None and isinstance(diffmap_params, dict):
            diffmap(adata, **diffmap_params)
        if leiden_params is not None and isinstance(leiden_params, dict):
            leiden(adata, **leiden_params)
        if fdg_params is not None and isinstance(fdg_params, dict):
            fdg(adata, **fdg_params)
    return adata


def save_pipeline_object(
    ad,
    out_prefix=None,
    batch_method=None,
    obs_keys=[],
    obsm_keys=[],
    uns_keys=[],
):
    if batch_method is None:
        obs_keys = [
            "leiden_r0_1",
            "leiden_r0_3",
            "leiden_r0_5",
            "leiden_r0_7",
            "leiden_r0_9",
        ]
        obsm_keys = ["X_pca"]
    elif batch_method == "harmony":
        obs_keys = [
            "leiden_hm_r0_1",
            "leiden_hm_r0_3",
            "leiden_hm_r0_5",
            "leiden_hm_r0_7",
            "leiden_hm_r0_9",
        ]
        obsm_keys = ["X_pca", "X_pca_hm"]
        uns_keys = ["neighbors"]
    elif batch_method == "bbknn":
        obs_keys = [
            "leiden_bk_r0_1",
            "leiden_bk_r0_3",
            "leiden_bk_r0_5",
            "leiden_bk_r0_7",
            "leiden_bk_r0_9",
        ]
        obsm_keys = ["X_pca", "X_pca_bk"]
        uns_keys = ["neighbors"]

    ad1 = ad.copy()
    for k in obs_keys:
        if k in ad1.obs.keys():
            del ad1.obs[k]
    for k in obsm_keys:
        if k in ad1.obsm.keys():
            del ad1.obsm[k]
    for k in uns_keys:
        if k in ad1.uns.keys():
            del ad1.uns[k]
    clear_colors(ad1)
    if ad1.raw:
        ad1.X = ad1.raw.X
        ad1.raw = None
    if out_prefix:
        ad1.write(f"{out_prefix}.processed.h5ad", compression="lzf")
    return ad1


def integrate(
    ads,
    ad_prefices=None,
    ad_types=None,
    annotations=None,
    batches=None,
    join="inner",
    n_hvg=4000,
    pool_only=False,
    normalize=True,
):
    n_ad = len(ads)
    if ad_prefices is None:
        ad_prefices = list(map(str, range(n_ad)))
    if ad_types is not None:
        if isinstance(ad_types, str):
            ad_types = [ad_types] * n_ad
        elif not isinstance(ad_types, (tuple, list)) or len(ad_types) != n_ad:
            raise ValueError("invalid `ad_types` provided")
    else:
        ad_types = ["auto"] * n_ad
    if annotations is not None:
        if isinstance(annotations, str):
            annotations = [annotations] * n_ad
        elif (
            not isinstance(annotations, (tuple, list))
            or len(annotations) != n_ad
        ):
            raise ValueError("invalid `annotations` provided")
    if batches is not None:
        if isinstance(batches, str):
            batches = [batches] * n_ad
        elif not isinstance(batches, (tuple, list)) or len(batches) != n_ad:
            raise ValueError("invalid `batches` provided")

    norm_ads = []
    for i, ad in enumerate(ads):
        ad_type = ad_types[i]
        if ad_type not in ("raw_norm", "counts", "norm"):
            if ad.raw and sp.issparse(ad.raw.X):
                ad_type = "raw_norm"
            elif (
                sp.issparse(ad.X)
                and np.abs(ad.X.data - ad.X.data.astype(int)).sum()
                < 1e-6 * ad.X.data.size
            ):
                ad_type = "counts"
            elif (
                sp.issparse(ad.X)
                and np.abs(
                    np.expm1(ad.X[0:10, :]).sum(axis=1).A1
                    - np.expm1(ad.X[0:10, :]).sum(axis=1).A1.mean()
                ).sum()
                < 1e-2
            ):
                ad_type = "norm"
            else:
                raise ValueError(
                    f"Cannot determine the type of anndata at position {i}"
                )
            print(ad_type)

        if ad_type == "raw_norm":
            norm_ad = anndata.AnnData(
                X=ad.raw.X, obs=ad.obs.copy(), var=ad.raw.var.copy()
            )
        elif ad_type == "norm":
            if not sp.issparse(ad.X):
                norm_ad = anndata.AnnData(
                    X=sp.csr_matrix(ad.X), obs=ad.obs.copy(), var=ad.var.copy()
                )
            else:
                norm_ad = anndata.AnnData(
                    X=ad.X, obs=ad.obs.copy(), var=ad.var.copy()
                )
        else:
            norm_ad = anndata.AnnData(
                X=ad.X, obs=ad.obs.copy(), var=ad.var.copy()
            )
            if normalize:
                sc.pp.normalize_total(norm_ad, target_sum=1e4)
                sc.pp.log1p(norm_ad)

        if normalize:
            post_norm_count = (
                np.expm1(norm_ad.X[0:10, :]).sum(axis=1).A1.mean().astype(int)
            )
            if post_norm_count != 10000:
                norm_ad.X = norm_ad.X / (post_norm_count / 1e4)

        prefix = ad_prefices[i]
        if (
            batches
            and batches[i] in norm_ad.obs.columns
            and batches[i] != "batch"
        ):
            if "batch" in norm_ad.obs.columns:
                del norm_ad.obs["batch"]
            norm_ad.obs.rename(columns={batches[i]: "batch"}, inplace=True)
        if (
            annotations
            and annotations[i] in norm_ad.obs.columns
            and annotations[i] != "annot"
        ):
            if "annot" in norm_ad.obs.columns:
                del norm_ad.obs["annot"]
            norm_ad.obs.rename(columns={annotations[i]: "annot"}, inplace=True)
            norm_ad.obs["annot"] = f"{prefix}_" + norm_ad.obs["annot"].astype(
                str
            )
        norm_ads.append(norm_ad)
        del norm_ad

    pooled = anndata.AnnData.concatenate(
        *norm_ads, batch_key="dataset", batch_categories=ad_prefices, join=join
    )

    calculate_qc(pooled, flag_only=True)
    if pool_only:
        return pooled
    pooled1 = simple_default_pipeline(
        pooled,
        post_norm_only=True,
        do_clustering=False,
        batch="dataset" if batches is None else ["dataset", "batch"],
        hvg_kw={"by_batch": ("dataset", 1), "n_hvg": n_hvg},
        pca_kw={"remove_genes": ("mito", "ribo")},
        hm_kw={"max_iter_harmony": 20},
    )
    return pooled1


def crossmap(adata, dataset="dataset", annotation="annot"):
    datasets = adata.obs[dataset].cat.categories
    if datasets.size != 2:
        raise ValueError("Number of pooled datasets != 2")
    ds1, ds2 = datasets.to_list()
    skip_prediction = (
        f"{ds1}_annot" in adata.obs.columns
        and f"{ds2}_annot" in adata.obs.columns
    )
    if not skip_prediction:
        ad1 = adata[adata.obs[dataset] == ds1].copy()
        ad2 = adata[adata.obs[dataset] == ds2].copy()
        ad1.obs[annotation].cat.rename_categories(
            lambda x: x.replace(f"{ds1}_", ""), inplace=True
        )
        ad2.obs[annotation].cat.rename_categories(
            lambda x: x.replace(f"{ds2}_", ""), inplace=True
        )
        lr1 = LR_train(ad1, annotation)
        lr2 = LR_train(ad2, annotation)
        LR_predict(adata, lr2, key_added=f"{ds2}_annot")
        LR_predict(adata, lr1, key_added=f"{ds1}_annot")
    df = pd.DataFrame(
        np.dot(
            cross_table(
                adata, f"{ds1}_annot", annotation, normalise="yx"
            ).values,
            cross_table(
                adata, annotation, f"{ds2}_annot", normalise="xy"
            ).values,
        ),
        index=adata.obs[f"{ds1}_annot"].cat.categories,
        columns=adata.obs[f"{ds2}_annot"].cat.categories,
    )
    if skip_prediction:
        return df
    else:
        return df, {ds1: ad1, ds2: ad2}, {ds1: lr1, ds2: lr2}


def auto_filter_cells(
    ad,
    min_count=250,
    min_gene=50,
    subset=True,
    filter_kw={
        "metrics": ["n_counts", "n_genes", "percent_mito"],
        "min_count": 1000,
        "min_gene": 200,
        "max_mito": 20,
        "min_pass_rate": 0.5,
        "onesided": True,
    },
):
    filter_kw = filter_kw.copy()
    while True:
        try:
            ad1 = simple_default_pipeline(
                ad,
                filter_only=True,
                filter_kw=filter_kw,
            )
        except QcLowPassError as e:
            error_metric = str(e)
            sc_warn(f"{error_metric} passing rate too low")
            if error_metric == "n_counts":
                if not filter_kw.get("onesided", False):
                    filter_kw["onesided"] = True
                elif filter_kw["min_count"] > min_count:
                    filter_kw["min_count"] //= 2
                    sc_warn("decrease `min_count` by half")
                else:
                    ad = ad[ad.obs["n_counts"] >= min_count].copy()
                    filter_kw["metrics"].remove(error_metric)
            elif error_metric == "n_genes":
                if not filter_kw.get("onesided", False):
                    filter_kw["onesided"] = True
                elif filter_kw["min_gene"] > min_gene:
                    filter_kw["min_gene"] //= 2
                    sc_warn("decrease `min_gene` by half")
                else:
                    ad = ad[ad.obs["n_genes"] >= min_gene].copy()
                    filter_kw["metrics"].remove(error_metric)
            elif error_metric == "percent_mito" and filter_kw["max_mito"] < 50:
                filter_kw["max_mito"] += 10
                sc_warn("increase `max_mito` by 10")
            else:
                filter_kw["metrics"].remove(error_metric)
                sc_warn(f"remove {error_metric} from filters")
            continue
        break
    if subset:
        ad1.uns["auto_filter_cells"] = filter_kw
        return ad1
    else:
        ad.obs["pass_auto_filter"] = ad.obs_names.isin(ad1.obs_names)
        ad.uns["auto_filter_cells"] = filter_kw
        return ad
