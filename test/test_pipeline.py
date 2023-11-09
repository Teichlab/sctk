import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sctk import (
    auto_filter_cells,
    auto_zoom_in,
    calculate_qc,
    cellwise_qc,
    clusterwise_qc,
    crossmap,
    custom_pipeline,
    generate_qc_clusters,
    get_good_sized_batch,
    integrate,
    recluster_subset,
    simple_default_pipeline,
)
from sctk._pipeline import _scale_factor, fit_gaussian
from sklearn.mixture import GaussianMixture


def test_calculate_qc():
    # create test data
    adata = sc.datasets.pbmc3k()

    # test that calculate_qc adds expected columns to adata.obs and adata.var
    calculate_qc(adata)
    assert "n_counts" in adata.obs.columns
    assert "log1p_n_counts" in adata.obs.columns
    assert "n_genes" in adata.obs.columns
    assert "log1p_n_genes" in adata.obs.columns
    assert "mito" in adata.var.columns
    assert "ribo" in adata.var.columns
    assert "hb" in adata.var.columns
    assert "n_counts" in adata.var.columns
    assert "n_cells" in adata.var.columns


def test_generate_qc_clusters():
    # create test data
    adata = sc.datasets.pbmc3k()

    # test that generate_qc_clusters adds expected columns to adata.obs and adata.obsm
    calculate_qc(adata)
    generate_qc_clusters(
        adata,
        [
            "n_counts",
            "n_genes",
            "percent_mito",
            "percent_ribo",
            "percent_hb",
        ],
    )
    assert "qc_cluster" in adata.obs.columns
    assert "X_umap_qc" in adata.obsm.keys()


def test_scale_factor():
    # test that _scale_factor returns expected values
    assert _scale_factor(np.array([0, 1, 2, 3, 4, 5])) == 1.0
    assert _scale_factor(np.array([0, 1, 2, 3, 10])) == 0.5


def test_fit_gaussian():
    # create test data (normal distribution centered on 5)
    x = np.random.normal(5, 2, 1000)

    # test that fit_gaussian returns expected values
    x_peak, x_left, x_right = fit_gaussian(x, n_components=[2], threshold=0.05, plot=False)
    assert -1 < x_peak < 1
    assert 9 < x_left < 11
    assert isinstance(x_right, GaussianMixture)


def test_cellwise_qc():
    # Load example dataset
    adata = sc.datasets.pbmc3k()

    # Test default metrics
    calculate_qc(adata)
    cellwise_qc(adata)
    assert "cell_passed_qc" in adata.obs.columns

    # Test custom metrics and custom storage key
    cellwise_qc(
        adata,
        metrics={
            "n_counts": [500, None, "log", "min_only", 0.5],
            "percent_mito": [0.05, None, "linear", "max_only", 0.5],
        },
        cell_qc_key = "CQK"
    )
    assert "CQK" in adata.obs.columns


def test_clusterwise_qc():
    # Load example dataset
    adata = sc.datasets.pbmc3k()
    calculate_qc(adata)
    generate_qc_clusters(
        adata,
        [
            "n_counts",
            "n_genes",
            "percent_mito",
            "percent_ribo",
            "percent_hb",
        ],
    )
    cellwise_qc(adata)

    # Test default
    clusterwise_qc(adata)
    assert "cluster_passed_qc" in adata.obs

    # Test custom cell QC key
    # Set all but one cells to fail
    adata.obs['all_cells_fail'] = [True]+[False]*(adata.shape[0]-1)
    clusterwise_qc(
        adata,
        cell_qc_key = "all_cells_fail",
        key_added = "all_clusters_fail"
    )
    assert "all_clusters_fail" in adata.obs
    assert adata.obs["all_clusters_fail"].sum() == 0


def test_get_good_sized_batch():
    # Test default min_size
    batches = pd.Series(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],
    )
    good_batches = get_good_sized_batch(batches)
    assert isinstance(good_batches, list)
    assert len(good_batches) == 1

    # Test custom min_size
    good_batches = get_good_sized_batch(batches, min_size=5)
    assert isinstance(good_batches, list)
    assert len(good_batches) == 2


def test_simple_default_pipeline():
    pass


def test_recluster_subset():
    # Test with a small dataset
    adata = sc.datasets.pbmc68k_reduced()
    groupby = "louvain"
    groups = ["0", "1"]
    res = 0.5
    new_key = "reclustered"
    ad_aux = None
    result = recluster_subset(adata, groupby, groups, res, new_key, ad_aux)
    assert isinstance(result, sc.AnnData)
    assert new_key in adata.obs.columns
    assert result.obs["leiden_aux"].nunique() == len(groups)

    # Test with missing values
    adata = sc.datasets.pbmc68k_reduced()
    adata.X[0, 0] = np.nan
    adata.X[0, 1] = np.nan
    groupby = "louvain"
    groups = ["0", "1"]
    res = 0.5
    new_key = "reclustered"
    ad_aux = None
    result = recluster_subset(adata, groupby, groups, res, new_key, ad_aux)
    assert isinstance(result, sc.AnnData)
    assert new_key in adata.obs.columns
    assert not np.isnan(result.X[0, 0])


def test_integrate():
    # Load example datasets (same dataset twice for now)
    adata1 = sc.datasets.pbmc68k_reduced()
    adata2 = sc.datasets.pbmc68k_reduced()

    # Integrate the datasets
    ads = [adata1, adata2]
    ad_prefices = ["pbmc68k", "pbmc3k"]
    ad_types = ["counts", "counts"]
    annotations = ["cell_type", "cell_type"]
    batches = None
    integrated = integrate(
        ads,
        ad_prefices=ad_prefices,
        ad_types=ad_types,
        annotations=annotations,
        batches=batches,
        join="outer",
        n_hvg=2000,
        pool_only=False,
        normalize=False,
    )

    # Check that the integrated object has the correct shape and annotations
    assert integrated.shape[0] == adata1.shape[0] + adata2.shape[0]
    assert "n_genes" in integrated.obs.columns
    assert "dataset" in integrated.obs.columns
    assert len(integrated.obs["dataset"].unique()) == 2
