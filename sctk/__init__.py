"""
Provides exported functions
"""

from ._read import read_10x, read_10x_atac, read_h5ad, read_cellbender, read_velocyto
from ._filter import filter_anndata
from ._norm import normalize
from ._hvg import hvg
from ._highly_deviant_genes import highly_deviant_genes
from ._pca import pca
from ._neighbors import neighbors
from ._umap import umap
from ._fdg import fdg
from ._tsne import tsne
from ._louvain import louvain
from ._leiden import leiden, leiden_shredding
from ._diffexp import diffexp, diffexp_paired, extract_de_table
from ._diffmap import diffmap
from ._dpt import dpt
from ._paga import paga, plot_paga
from ._doublets import run_scrublet
from ._markers import (
    volcano_plot,
    calc_marker_stats,
    filter_marker_stats,
    top_markers,
    test_markers,
)
from ._utils import (
    sc_warn,
    read_list,
    lognorm_to_counts,
    restore_adata,
    find_top_expressed_genes,
    remove_genes,
    project_into_PC,
    cross_table,
    run_cNMF,
    run_harmony,
    run_bbknn,
    run_phate,
    split_by_group,
    regroup,
    subsample,
    pseudo_bulk,
    show_obs_categories,
    write_10x_h5,
    write_mtx,
    write_table,
    write_cellxgene_object,
)
from ._plot import (
    expression_colormap,
    clear_colors,
    set_figsize,
    abline,
    heatmap,
    plot_df_heatmap,
    plot_qc_violin,
    plot_qc_scatter,
    plot_metric_by_rank,
    plot_embedding,
    plot_markers,
    plot_diffexp,
    highlight,
    dotplot2,
    plot_genes,
    plot_scatter,
)
from ._annot import (
    LR_train,
    LR_predict,
    annotate,
)
from ._velocity import run_scvelo, plot_scvelo
from ._pipeline import (
    calculate_qc,
    generate_qc_clusters,
    get_good_sized_batch,
    simple_default_pipeline,
    recluster_subset,
    auto_zoom_in,
    custom_pipeline,
    integrate,
    crossmap,
    QcLowPassError,
    auto_filter_cells,
)
