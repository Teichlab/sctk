import scanpy as sc
from ._pca import pca
from ._plot import clear_colors, plot_embedding
from ._utils import remove_genes, run_harmony

try:
    import scvelo as scv
    def run_scvelo(adata, groupby, groups=None, batch=None, mode='dynamical', use_rep='X_pca', pca_kw={}):
        ad = remove_genes(adata, ['mito', 'ribo', 'hb'])
        scv.pp.filter_and_normalize(ad, min_shared_counts=20, n_top_genes=2000)
        pca(ad, **pca_kw)
        if batch:
            run_harmony(ad, batch=batch, key_added='hm')
        scv.pp.moments(ad, use_rep=use_rep, n_neighbors=30, n_pcs=30)
        if mode == 'dynamical':
            scv.tl.recover_dynamics(ad)
        scv.tl.velocity(ad, mode=mode, diff_kinetics=(mode == 'dynamical'))
        scv.tl.velocity_graph(ad)
        scv.tl.latent_time(ad)
        scv.tl.velocity_pseudotime(ad)
        return ad


    def plot_scvelo(adata, basis, groupby=None, groups=None, smooth=1.5, figsize=(4, 4), alpha=0.3, palette=None):
        if groups:
            ad = adata[adata.obs[groupby].isin(groups)].copy()
        else:
            ad = adata
        if palette is None:
            clear_colors(ad)
        scv_plot_settings = {
            'figsize': figsize, 'size': 100, 'palette': palette, 'alpha': alpha, 'legend_loc': 'right margin', 'legend_fontsize': 15
        }
        scv.pl.proportions(ad, groupby=groupby)
        plot_embedding(ad, basis=basis, groupby=groupby, annot='full', figsize=figsize, palette=palette)
        scv.pl.velocity_embedding_stream(ad, basis=basis, color=groupby)
        scv.pl.scatter(ad, basis=basis, c=('root_cells', 'end_points', 'latent_time'), size=30)

except ImportError:
    def run_scvelo(*args, **kwargs):
        raise NotImplementedError

    def plot_scvelo(*args, **kwargs):
        raise NotImplementedError
