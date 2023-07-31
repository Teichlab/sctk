"""
High-level pipeline functions
"""

from typing import Union, Optional

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
    """
    Error raised when a QC minimum requirements are not met for a dataset.
    """

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
            :py:func:`scanpy.pp.calculate_qc_metrics`.

    Returns:
        None.

    Raises:
        None.

    Examples:
        >>> import scanpy as sc
        >>> import sctk
        >>> adata = sc.datasets.pbmc3k()
        >>> sctk.calculate_qc(adata)

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
) -> anndata.AnnData:
    """
    Generate quality control (QC) clusters for an AnnData object. The object is
    modified in-place.

    This function generates QC clusters for an AnnData object using the
    specified QC metrics. If an auxiliary AnnData object is not provided, this
    function will create one by performing PCA, nearest neighbor graph
    construction, and UMAP embedding on the specified QC metrics.

    Args:
        ad: AnnData object to generate QC clusters for.
        metrics: List of QC metrics to use for generating QC clusters. Must be
            present as obs columns.
        aux_ad: Optional auxiliary AnnData object to use for generating QC
            clusters, created by an earlier call of this function and returned if
            return_aux is set to True. Its neighbour graph will be used for
            clustering and its UMAP will be transferred to the input object.
        n_pcs: Number of principal components to use for PCA. If not provided,
            this will be set to max(2, len(metrics) - 2).
        n_neighbors: Number of nearest neighbors to use for constructing the
            nearest neighbor graph. If not provided, this will be set to min(max(5,
            int(ad.n_obs / 500)), 10).
        res: Resolution parameter to use for the Leiden clustering algorithm.
        return_aux: If True, return the auxiliary AnnData object used for
            generating QC clusters.

    Returns:
        If `return_aux` is False, returns None. Otherwise, returns the auxiliary
        AnnData object used for generating QC clusters.

    Raises:
        None.

    Examples:
        >>> import scanpy as sc
        >>> import sctk
        >>> adata = sc.datasets.pbmc3k()
        >>> sctk.calculate_qc(adata)
        >>> metrics_list = ["n_counts", "n_genes", "percent_mito", "percent_ribo", "percent_hb"]
        >>> sctk.generate_qc_clusters(adata, metrics=metrics_list)

    """
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
    """
    Calculate the scaling factor for a given array.

    This function calculates the scaling factor for a given array by computing
    the difference between the maximum and minimum values of the array and
    scaling it to a fixed value of 5.0.

    TODO why?

    Args:
        x: Numpy array to calculate the scaling factor for.

    Returns:
        Scaling factor as a float.

    Raises:
        None.

    Examples:
        >>> import numpy as np
        >>> from sctk._pipeline import _scale_factor
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> _scale_factor(x)
        1.25

    """
    xmin = np.min(x)
    xmax = np.max(x)
    return 5.0 / (xmax - xmin)


def fit_gaussian(
    x,
    n_components=np.arange(10) + 1,
    threshold=0.05,
    xmin=None,
    xmax=None,
    plot=False,
    nbins=500,
    hist_bins=100,
) -> tuple:
    """
    Fit a Gaussian mixture model to a 1D numpy array.

    This function fits a Gaussian mixture model to a 1D numpy array using the
    specified number of components and threshold. It returns the lower and upper
    bounds of the fitted Gaussian distribution, as well as the fitted Gaussian
    mixture model.

    Args:
        x: 1D numpy array to fit a Gaussian mixture model to.
        n_components: Number of components to use for the Gaussian mixture model.
            The best GMM will be selected based on BIC.
        threshold: Threshold value for determining the lower and upper bounds of
            the fitted Gaussian distribution.
        xmin: Minimum value to use for the fitted Gaussian distribution. If not
            provided, this will be set to the minimum value of `x`.
        xmax: Maximum value to use for the fitted Gaussian distribution. If not
            provided, this will be set to the maximum value of `x`.
        plot: If True, plot the fitted Gaussian distribution.
        nbins: Number of bins to use for the distribution of `x`.
        hist_bins: Number of bins to use for the histogram of `x` in the plot.

    Returns:
        Tuple containing the lower bound, upper bound, and Gaussian mixture
        model.

    Raises:
        None.

    Examples:
        >>> import numpy as np
        >>> from sctk import fit_gaussian
        >>> x = np.random.normal(loc=0, scale=1, size=1000)
        >>> x_left, x_right, gmm = fit_gaussian(x, n_components=[2], threshold=0.1)

    """
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    x_fit = x[(x >= xmin) & (x <= xmax)]
    f = _scale_factor(x_fit)
    x_fit = (x_fit * f).reshape(-1, 1)
    # try a bunch of different component counts for the GMM
    gmms = []
    bics = []
    for n in n_components:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(x_fit)
        while not gmm.converged_:
            gmm.fit(x_fit, warm_start=True)
        gmms.append(gmm)
        bics.append(gmm.bic(x_fit))
    # pick best one based on BIC (the lower the better)
    # making this plot is useless if there's a single component count
    if plot and len(n_components) > 1:
        plt.plot(n_components, bics)
        plt.xlabel("GMM components")
        plt.ylabel("BIC")
        plt.show()
    # the minimum bic's index is the position in n_components
    # as well as the gmm list
    n = n_components[np.argmin(bics)]
    gmm = gmms[np.argmin(bics)]
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


def cellwise_qc(adata, metrics=None, cell_qc_key="cell_passed_qc", **kwargs):
    """
    Filter cells in an AnnData object based on quality control metrics. The
    object is modified in-place.

    This function filters cells in an AnnData object based on quality control
    metrics. The metrics used for filtering can be specified using the `metrics`
    argument. By default, the function uses a set of default metrics, but these
    can be overridden by passing a list/tuple of metric names or a dictionary of
    metric names and their corresponding parameters.

    Args:
        adata: AnnData object to filter cells from.
        metrics: Optional list/tuple of metric names or dictionary of metric
            names and their corresponding parameters. If not provided, the function
            uses a set of default metrics. For defaults and an explanation, please
            refer to the QC workflow demo notebook.
        cell_qc_key: Obs column in the object to store the per-cell QC calls in.
        **kwargs: Additional keyword arguments to pass to the
            :py:func:`fit_gaussian` function.

    Returns:
        None.

    Raises:
        ValueError: If `metrics` is not a list/tuple of metric names or a
            dictionary of metric names and their corresponding parameters.

    Examples:
        >>> import scanpy as sc
        >>> import sctk
        >>> adata = sc.datasets.pbmc3k()
        >>> sctk.calculate_qc(adata)
        >>> sctk.cellwise_qc(adata)
    """
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
            x_low, x_high, _ = fit_gaussian(x, xmin=min_x, xmax=max_x, **kwargs)
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
        if m_pass.sum() < n_obs * min_pass_rate:
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
    adata.obs[cell_qc_key] = all_passed
    if adata.obs[cell_qc_key].sum() == 0:
        print(
            "No cells passed. Performing simple filtering on counts, genes and mito%"
        )
        adata.obs[cell_qc_key] = (
            (adata.obs.n_counts >= metrics["n_counts"][0])
            & (adata.obs.n_genes >= metrics["n_genes"][0])
            & (adata.obs.percent_mito < metrics["percent_mito"][1])
        )


def filter_qc_outlier_legacy(
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
) -> np.ndarray:
    """
    Filter out cells with outlier QC metrics.

    This function filters out cells with outlier QC metrics based on a set of
    predefined thresholds. The function takes an AnnData object as input and
    returns a boolean array indicating which cells pass the QC filter.

    Args:
        adata: AnnData object to filter.
        metrics: List of QC metrics to use for filtering.
        min_count: Minimum number of counts per cell.
        min_gene: Minimum number of genes per cell.
        min_mito: Minimum percentage of mitochondrial genes per cell.
        max_mito: Maximum percentage of mitochondrial genes per cell.
        min_ribo: Minimum percentage of ribosomal genes per cell.
        max_ribo: Maximum percentage of ribosomal genes per cell.
        max_hb: Maximum percentage of hemoglobin genes per cell.
        min_top50: Minimum percentage of reads mapping to the top 50 genes.
        max_top50: Maximum percentage of reads mapping to the top 50 genes.
        min_pass_rate: Minimum pass rate for the QC filter.
        onesided: Whether to use a one-sided or two-sided threshold for count
            and gene metrics.
        force: Whether to force the function to pass even if the pass rate is
            below the minimum threshold.

    Returns:
        Boolean array indicating which cells pass the QC filter.

    Raises:
        QcLowPassError: If the pass rate for a given metric is below the minimum
            threshold and force is False.

    Examples: TODO check
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> adata = adata[:, adata.var["highly_variable"]]
        >>> adata = sc.pp.normalize_total(adata, target_sum=1e4)
        >>> adata = sc.pp.log1p(adata)
        >>> adata = sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        >>> adata = adata[:, adata.var["highly_variable"]]
        >>> sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
        >>> sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
        >>> sc.tl.umap(adata)
        >>> k_pass = filter_qc_outlier_legacy(adata)
    """
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
            n_components=[10],
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


def clusterwise_qc(
    ad,
    threshold=0.5,
    cell_qc_key="cell_passed_qc",
    key_added="cluster_passed_qc",
) -> None:
    """
    Find good quality control (QC) clusters in an AnnData object.

    This function finds good quality control (QC) clusters in an AnnData object
    by identifying clusters that have a high proportion of cells that pass the
    QC filter.

    Args:
        ad: AnnData object to find good QC clusters in. Needs qc_cluster
            present in obs.
        threshold: Clusters featuring at least this fraction of good QC cells
            will be deemed good QC clusters.
        cell_qc_key: Key to use to retrieve per-cell QC calls from obs in the
            AnnData.
        key_added: Key to use for storing the results in the AnnData obs object.

    Returns:
        None.

    Raises:
        None.

    Examples:
        >>> import scanpy as sc
        >>> import sctk
        >>> adata = sc.datasets.pbmc3k()
        >>> sctk.calculate_qc(adata)
        >>> metrics_list = ["n_counts", "n_genes", "percent_mito", "percent_ribo", "percent_hb"]
        >>> sctk.generate_qc_clusters(adata, metrics=metrics_list)
        >>> sctk.cellwise_qc(adata)
        >>> sctk.clusterwise_wc(adata)

    """
    if ad.obs[cell_qc_key].astype(bool).sum() == 0:
        good_qc_clusters = []
    else:
        good_qc_clusters = (
            pd.crosstab(
                ad.obs.qc_cluster,
                ad.obs[cell_qc_key].astype("category"),
                normalize="index",
            )
            .where(lambda x: x[True] >= threshold)
            .dropna()
            .index.tolist()
        )

    ad.obs[key_added] = ad.obs["qc_cluster"].isin(good_qc_clusters)


def get_good_sized_batch(batches, min_size=10) -> list:
    """
    Get a list of batches with a minimum size.

    This function takes a pandas Series of batch IDs and their corresponding
    sizes and returns a list of batch IDs that have a size greater than or equal
    to the specified minimum size.

    Args:
        batches: Pandas Series of batch IDs and their corresponding sizes.
        min_size: Minimum size of batches to include in the output list. Default
            is 10.

    Returns:
        List of batch IDs with a size greater than or equal to `min_size`.

    Raises:
        None.
    """
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
) -> anndata.AnnData:
    """
    Applies a default preprocessing pipeline to the input AnnData object.

    Args:
        adata: AnnData object to preprocess.
        filter_only: If True, only performs filtering steps and returns the
            filtered AnnData object.
        post_filter_only: If True, only performs post-filtering steps and
            returns the processed AnnData object.
        norm_only: If True, only performs normalization steps and returns the
            normalized AnnData object.
        post_norm_only: If True, only performs post-normalization steps and
            returns the processed AnnData object.
        hvg_only: If True, only performs highly variable gene selection and
            returns the selected AnnData object.
        post_hvg_only: If True, only performs post-highly variable gene
            selection steps and returns the processed AnnData object.
        pca_only: If True, only performs PCA and returns the PCA-transformed
            AnnData object.
        post_pca_only: If True, only performs post-PCA steps and returns the
            processed AnnData object.
        do_clustering: If True, performs clustering and updates the AnnData
            object with the clustering results.
        zero_center: If True, centers the data to zero mean before scaling. If
            None, uses a heuristic based on the number of cells.
        do_combat: If True and `batch` is not None, performs batch correction
            using ComBat.
        batch: The batch variable(s) to use for batch correction. If None, no
            batch correction is performed.
        batch_method: The batch correction method to use. Can be "harmony" or
            "bbknn".
        random_state: The random seed to use for reproducibility.
        clustering_resolution: The clustering resolution(s) to use for Leiden
            clustering.
        use_gpu: If True, uses the GPU for some computations. If "all", uses the
            GPU for all computations.
        use_hvg: The list of highly variable genes to use for normalization and
            PCA. If None, uses the default Seurat v3 HVG selection.
        filter_kw: Additional keyword arguments to pass to the
            `filter_qc_outlier_legacy` function.
        hvg_kw: Additional keyword arguments to pass to the `hvg` function.
        rgs_kw: Additional keyword arguments to pass to the `sc.pp.regress_out`
            function.
        pca_kw: Additional keyword arguments to pass to the `pca` function.
        nb_kw: Additional keyword arguments to pass to the `sc.pp.neighbors`
            function.
        umap_kw: Additional keyword arguments to pass to the `umap` function.
        hm_kw: Additional keyword arguments to pass to the `run_harmony`
            function.
        bk_kw: Additional keyword arguments to pass to the `run_bbknn` function.

    Returns:
        The preprocessed AnnData object.
    """
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
        k_cell = filter_qc_outlier_legacy(adata, **filter_kw)
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
    """
    Reclusters a subset of cells in an AnnData object.

    This function reclusters a subset of cells in an AnnData object by first
    filtering, normalizing, and performing PCA on the subset, and then
    performing clustering using the Leiden algorithm. The resulting clusters are
    added to the original AnnData object as a new categorical variable.

    Args:
        adata: AnnData object to recluster.
        groupby: Categorical variable to group cells by.
        groups: List of group names to recluster.
        res: Clustering resolution to use for Leiden clustering.
        new_key: Name of the new categorical variable to add to the AnnData
            object.
        ad_aux: Optional preprocessed AnnData object to use for clustering.
        **kwargs: Additional keyword arguments to pass to the
            :py:func:`simple_default_pipeline` function.

    Returns:
        If `ad_aux` is not provided, the preprocessed AnnData object used for
        clustering. Otherwise, None.
    """

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
) -> None:
    """
    Automatically zooms in on subclusters in an AnnData object.

    This function performs iterative clustering on an AnnData object using the
    Leiden algorithm, starting from a specified categorical variable and
    splitting each group into subclusters until no further subclustering is
    possible. The resulting subclusters are added to the original AnnData object
    as new categorical variables.

    Args:
        ad: AnnData object to cluster.
        use_graph: Name of the graph to use for clustering.
        groupby: Name of the categorical variable to start clustering from.
        restrict_to: Optional list of group names to restrict clustering to.
        start_index: Index to start numbering new categorical variables from.
        min_res: Minimum clustering resolution to use.
        max_res: Maximum clustering resolution to use.
        leiden_kw: Additional keyword arguments to pass to the `leiden`
            function.
        marker_kw: Additional keyword arguments to pass to the
            :py:func:`filter_marker_stats` function.

    Returns:
        None.
    """
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
) -> anndata.AnnData:
    """
    Applies a custom Scanpy pipeline to an AnnData object.

    This function applies a custom Scanpy pipeline to an AnnData object, which
    includes filtering, normalization, batch correction, highly variable gene
    selection, dimensionality reduction, clustering, and visualization. The
    pipeline can be customized by passing various keyword arguments to the
    function.

    Args:
        adata: AnnData object to apply the pipeline to.
        qc_only: If True, only perform quality control and skip the rest of the
            pipeline.
        plot: If True, generate plots of the pipeline results.
        batch: Categorical variable to correct for batch effects.
        filter_params: Dictionary of filtering parameters to pass to the
            `sc.pp.filter_cells` and `sc.pp.filter_genes` functions.
        norm_params: Dictionary of normalization parameters to pass to the
            `sc.pp.normalize_total` function.
        combat_args: Dictionary of batch correction parameters to pass to the
            `sc.pp.combat` function.
        hvg_params: Dictionary of highly variable gene selection parameters to
            pass to the `hvg` function.
        scale_params: Dictionary of scaling parameters to pass to the
            `sc.pp.scale` function.
        pca_params: Dictionary of PCA parameters to pass to the `pca` function.
        harmony_params: Dictionary of Harmony parameters to pass to the
            `run_harmony` function.
        nb_params: Dictionary of nearest neighbor parameters to pass to the
            `neighbors` function.
        umap_params: Dictionary of UMAP parameters to pass to the `umap`
            function.
        tsne_params: Dictionary of t-SNE parameters to pass to the `tsne`
            function.
        diffmap_params: Dictionary of diffusion map parameters to pass to the
            `diffmap` function.
        leiden_params: Dictionary of Leiden clustering parameters to pass to the
            `leiden` function.
        fdg_params: Dictionary of force-directed graph parameters to pass to the
            `fdg` function.

    Returns:
        The modified AnnData object.

    TODO some functions return a modified anndata object, some modify in place
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
) -> anndata.AnnData:
    """
    Saves a processed AnnData object to a file.

    This function saves a processed AnnData object to a file, after removing
    certain keys from the `obs`, `obsm`, and `uns` attributes of the object. The
    specific keys to remove depend on the `batch_method` argument, which can be
    set to "harmony", "bbknn", or None. If `batch_method` is None, the default
    keys to remove are ["leiden_r0_1", "leiden_r0_3", "leiden_r0_5",
    "leiden_r0_7", "leiden_r0_9"] and ["X_pca"]. If `batch_method` is "harmony",
    the keys to remove are ["leiden_hm_r0_1", "leiden_hm_r0_3",
    "leiden_hm_r0_5", "leiden_hm_r0_7", "leiden_hm_r0_9"], ["X_pca"], and
    ["X_pca_hm", "neighbors"]. If `batch_method` is "bbknn", the keys to remove
    are ["leiden_bk_r0_1", "leiden_bk_r0_3", "leiden_bk_r0_5", "leiden_bk_r0_7",
    "leiden_bk_r0_9"], ["X_pca"], and ["X_pca_bk", "neighbors"].

    TODO is this very custom with respect to the keys, or widely used?

    Args:
        ad: AnnData object to save.
        out_prefix: Prefix for the output file name. If None, no file is saved.
        batch_method: Batch correction method used to process the AnnData
            object. Can be "harmony", "bbknn", or None.
        obs_keys: List of keys to remove from the `obs` attribute of the AnnData
            object.
        obsm_keys: List of keys to remove from the `obsm` attribute of the
            AnnData object.
        uns_keys: List of keys to remove from the `uns` attribute of the AnnData
            object.

    Returns:
        The processed AnnData object.
    """
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
    """
    Integrates multiple AnnData objects into a single object.

    This function integrates multiple AnnData objects into a single object by
    first normalizing and preprocessing each object, and then concatenating them
    using the `concatenate` function. The resulting object can be used for
    downstream analysis, such as clustering and visualization.

    Args:
        ads: List of AnnData objects to integrate.
        ad_prefices: List of prefixes to use for the batch categories in the
            concatenated object. If not provided, uses integers starting from 0.
        ad_types: List of types to use for each AnnData object. If not provided,
            attempts to automatically determine the type of each object.
        annotations: List of annotation columns to use for each AnnData object.
            If not provided, uses the default "annot" column.
        batches: List of batch columns to use for each AnnData object. If not
            provided, uses the default "batch" column.
        join: Method to use for joining the AnnData objects. Can be "inner"
            (default) or "outer".
        n_hvg: Number of highly variable genes to select for each batch.
        pool_only: If True, returns the concatenated object without performing
            downstream analysis.
        normalize: If True, normalizes the expression data in each AnnData
            object before concatenating.

    Returns:
        If `pool_only` is True, the concatenated AnnData object. Otherwise, the
        preprocessed and analysed concatenated AnnData object.
    """
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
    """
    Cross-maps annotations between two datasets in an AnnData object.

    This function cross-maps annotations between two datasets in an AnnData
    object by first training a logistic regression model on each dataset, and
    then using the models to predict the annotations in the other dataset. The
    resulting cross-mapped annotations are added to the original AnnData object
    as new categorical variables.

    Args:
        adata: AnnData object to cross-map annotations in.
        dataset: Name of the categorical variable to group cells by. Default is
            "dataset".
        annotation: Name of the categorical variable to cross-map. Default is
            "annot".

    Returns:
        If the annotations have already been cross-mapped, a pandas DataFrame
        containing the cross-tabulated annotations. Otherwise, a tuple
        containing the cross-tabulated annotations, the preprocessed AnnData
        objects used for training the logistic regression models, and the
        trained models themselves.
    """
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
) -> anndata.AnnData:
    """
    Automatically filter cells in an AnnData object based on quality control
    metrics.

    This function automatically filters cells in an AnnData object based on
    quality control metrics using a simple iterative algorithm. The algorithm
    first applies a set of default filters to the data using the
    `simple_default_pipeline` function, and then iteratively adjusts the filter
    parameters until a minimum passing rate is achieved for each metric. The
    resulting filtered AnnData object is returned with a new categorical
    variable indicating which cells passed the filters.

    Args:
        ad (anndata.AnnData): AnnData object to filter cells in.
        min_count (int): Minimum number of counts per cell to retain. Default is 250.
        min_gene (int): Minimum number of genes per cell to retain. Default is 50.
        subset (bool): Whether to return a subset of the original AnnData object
            containing only the passing cells. Default is True.
        filter_kw (dict): Dictionary of keyword arguments to pass to the
            :py:func:`simple_default_pipeline` function. Default is a set of
            default filter parameters.

    Returns:
        If `subset` is True, a new AnnData object containing only the passing
        cells. Otherwise, the original AnnData object with a new categorical
        variable indicating which cells passed the filters.

    Raises:
        QcLowPassError: If dataset does not pass minimum specifications.

    """
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
