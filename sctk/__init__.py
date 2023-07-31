"""
Provides exported functions
"""

from ._read import (
    read_10x,
    read_10x_atac,
    read_h5ad,
    read_cellbender,
    read_velocyto,
)
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
from ._leiden import leiden, leiden_shredding, leiden_shredding2
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
    cross_table,
    dummy_to_categorical,
    expand_feature_space,
    find_top_expressed_genes,
    lognorm_to_counts,
    project_into_PC,
    pseudo_bulk,
    random_partition,
    read_list,
    regroup,
    remove_genes,
    restore_adata,
    run_bbknn,
    run_cNMF,
    run_cellphonedb,
    run_celltype_composition_analysis,
    run_celltypist,
    run_diffQuant,
    run_harmony,
    run_phate,
    sc_warn,
    score_msigdb_genesets,
    show_obs_categories,
    split_by_group,
    subsample,
    summarise_expression_by_group,
    write_10x_h5,
    write_cellxgene_object,
    write_mtx,
    write_table,
)
from ._plot import (
    abline,
    clear_colors,
    dotplot2,
    dotplot3,
    dotplot_combined_coexpression,
    expression_colormap,
    heatmap,
    highlight,
    plot_cellbender_qc,
    plot_composition,
    plot_df_heatmap,
    plot_diffexp,
    plot_embedding,
    plot_genes,
    plot_markers,
    plot_metric_by_rank,
    plot_qc_scatter,
    plot_qc_violin,
    plot_scatter,
    set_figsize,
)
from ._annot import (
    LR_predict,
    LR_train,
    annotate,
    get_top_LR_features,
)
from ._velocity import run_scvelo, plot_scvelo
from ._pipeline import (
    QcLowPassError,
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
