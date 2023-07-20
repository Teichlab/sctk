import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sctk import (
    auto_filter_cells,
    auto_zoom_in,
    calculate_qc,
    crossmap,
    custom_pipeline,
    find_good_qc_cluster,
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
    adata = anndata.AnnData(
        X=np.random.rand(100, 100),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(100)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(100)]),
    )

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
    adata = anndata.AnnData(
        X=np.random.rand(100, 100),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(100)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(100)]),
    )

    # test that generate_qc_clusters adds expected columns to adata.obs and adata.obsm
    calculate_qc(adata)
    generate_qc_clusters(adata, ["n_counts", "n_genes"])
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
    x_peak, x_left, x_right = fit_gaussian(x, n=2, threshold=0.05, plot=False)
    assert -1 < x_peak < 1
    assert 9 < x_left < 11
    assert isinstance(x_right, GaussianMixture)


def test_integrate():
    # Load example datasets
    adata1 = sc.datasets.pbmc68k_reduced()
    adata2 = sc.datasets.pbmc3k()

    # Integrate the datasets
    ads = [adata1, adata2]
    ad_prefices = ["pbmc68k", "pbmc3k"]
    ad_types = ["counts", "counts"]
    annotations = ["cell_type", "cell_type"]
    batches = ["batch", "batch"]
    integrated = integrate(
        ads,
        ad_prefices=ad_prefices,
        ad_types=ad_types,
        annotations=annotations,
        batches=batches,
        join="outer",
        n_hvg=2000,
        pool_only=False,
        normalize=True,
    )

    # Check that the integrated object has the correct shape and annotations
    assert integrated.shape == (
        adata1.shape[0] + adata2.shape[0],
        adata1.shape[1],
    )
    assert "batch" in integrated.obs.columns
    assert "cell_type" in integrated.obs.columns
    assert len(integrated.obs["batch"].unique()) == 2
    assert len(integrated.obs["cell_type"].unique()) == 14
