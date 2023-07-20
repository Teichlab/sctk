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
    assert np.isclose(_scale_factor(np.array([0, 1, 2, 3, 4])), 1.0)
    assert np.isclose(_scale_factor(np.array([0, 1, 2, 3, 10])), 0.5)
    assert np.isclose(_scale_factor(np.array([0, 1, 2, 3, -10])), -0.5)


def test_fit_gaussian():
    # create test data
    x = np.random.normal(loc=5, scale=2, size=1000)

    # test that fit_gaussian returns expected values
    x_peak, x_left, x_right = fit_gaussian(x, n=2, threshold=0.05, plot=False)
    assert np.isclose(x_peak, 0.0, rtol=1.0)
    assert np.isclose(x_left, 10.0, rtol=1.0)
    assert np.isclose(x_right, 9.0, rtol=0.1)
