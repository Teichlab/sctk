import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sctk import calculate_qc, generate_qc_clusters


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
