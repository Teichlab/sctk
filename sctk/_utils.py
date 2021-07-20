"""
Utility functions
"""

import numpy as np
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

sc_warn = sc.logging.warn if sc.__version__.startswith('1.4') else sc.logging.warning

def read_list(fn, **kwargs):
    return pd.read_csv(fn, header=None, names=['x'], **kwargs).x.to_list()


def lognorm_to_counts(X, n_counts=None, force=False, rounding=True):
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_expm1 = np.expm1(X)
    if n_counts is not None:
        size_factor = n_counts / 1e4
        X_counts = (X_expm1.T * sp.csr_matrix(sp.diags(size_factor))).T
        res = np.abs(X_counts.data - np.round(X_counts.data)).sum() / X_counts.data.sum()
        if res < 1e-6 or force:
            if rounding:
                X_counts.data = np.round(X_counts.data).astype(np.int32)
            return X_counts
        else:
            sc_warn('Non-integer residuals too large, try inferring size_factor')
    x_min = np.array([X_expm1.getrow(i).data.min() for i in range(X_expm1.shape[0])])
    size_factor = 1 / x_min
    X_counts = (X_expm1.T * sp.csr_matrix(sp.diags(size_factor))).T
    res = np.abs(X_counts.data - np.round(X_counts.data)).sum() / X_counts.data.sum()
    if res < 1e-6 or force:
        if rounding:
            X_counts.data = np.round(X_counts.data).astype(np.int32)
        return X_counts
    else:
        raise ValueError('Non-integer residuals too large, failed to recover counts')


def restore_adata(adata, restore_type=['norm', 'count'], use_raw=True, obs_cols=None, var_cols=None, obsm_keys=None, use_n_counts=False, **kwargs):
    if use_raw and not adata.raw:
        raise ValueError('adata.raw not found')

    if isinstance(restore_type, (list, tuple)):
        restore_type = restore_type[0]

    if restore_type == 'norm':
        if use_raw:
            X = adata.raw.X
        else:
            X = adata.X
    elif restore_type == 'count':
        n_counts = None
        if use_n_counts is True:
            if use_n_counts in adata.obs_keys():
                n_counts = adata.obs[use_n_counts].values
            elif 'n_counts' in adata.obs_keys():
                n_counts = adata.obs['n_counts'].values
        if use_raw:
            X0 = adata.raw.X
        else:
            X0 = adata.X
        X = lognorm_to_counts(X0, n_counts=n_counts, **kwargs)
    else:
        raise ValueError(f'{restore_type}: invalid <restore_type>, choose between "norm" and "count"')

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


def find_top_expressed_genes(ad, use_rep='X', groupby=None, n_genes=50, inplace=True):
    if use_rep == 'raw' and ad.raw is not None:
        X = ad.raw.X
        var_names = ad.raw.var_names
    elif use_rep == 'X':
        X = ad.X
        var_names = ad.var_names
    else:
        raise ValueError('invalid `use_rep`')
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
        ad.var[f'top{n_genes}'] = ad.var_names.isin(top_genes)
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


def run_harmony(
        adata,
        batch,
        theta=2.0,
        use_rep='X_pca',
        key_added='hm',
        random_state=0,
        **kwargs
):
    if not isinstance(batch, (tuple, list)):
        batch = [batch]
    if not isinstance(theta, (tuple, list)):
        theta = [theta] * len(batch)
    for b in batch:
        if b not in adata.obs.columns:
            raise KeyError(f'{b} is not a valid obs annotation.')
    if use_rep not in adata.obsm.keys():
        raise KeyError(f'{use_rep} is not a valid embedding.')
    meta = adata.obs[batch].reset_index()
    embed = adata.obsm[use_rep]

    if sc.__version__.startswith('1.4'):
        # ===========
        import rpy2.robjects
        from rpy2.robjects.packages import importr
        harmony = importr('harmony')
        from rpy2.robjects import numpy2ri, pandas2ri
        numpy2ri.activate()
        pandas2ri.activate()
        set_seed = rpy2.robjects.r("set.seed")
        set_seed(random_state)
        hm_embed = harmony.HarmonyMatrix(
                embed, meta, batch, theta, do_pca=False, verbose=False, **kwargs)
        pandas2ri.deactivate()
        numpy2ri.deactivate()
        hm_embed = numpy2ri.ri2py(hm_embed)
        # ===========
    else:
        import harmonypy as hm
        if 'max_iter_harmony' not in kwargs:
            kwargs['max_iter_harmony'] = 20
        hobj = hm.run_harmony(embed, meta, batch, theta=theta, **kwargs)
        hm_embed = hobj.Z_corr.T

    if key_added:
        adata.obsm[f'{use_rep}_{key_added}'] = hm_embed
    else:
        adata.obsm[use_rep] = hm_embed.T


def run_bbknn(adata, batch, use_rep='X_pca', key_added='bk', **kwargs):
    import bbknn
    _set_obsm_key(adata, 'X_pca', use_rep)
    try:
        _backup_default_key(adata.uns, 'neighbors')
        bbknn.bbknn(adata, batch_key=batch, **kwargs)
        if key_added:
            _rename_default_key(adata.uns, 'neighbors', f'neighbors_{key_added}')
            if not sc.__version__.startswith('1.4'):
                _rename_default_key(adata.obsp, 'distances', f'neighbors_{key_added}_distances')
                _rename_default_key(adata.obsp, 'connectivities', f'neighbors_{key_added}_connectivities')
                adata.uns[f'neighbors_{key_added}']['distances_key'] = f'neighbors_{key_added}_distances'
                adata.uns[f'neighbors_{key_added}']['connectivities_key'] = f'neighbors_{key_added}_connectivities'
        else:
            _delete_backup_key(adata.uns, 'neighbors')
    finally:
        _restore_obsm_key(adata, 'X_pca', use_rep)


def run_phate(adata, use_rep='X', key_added=None, knn=5, decay=40, t='auto', n_pca=100, random_state=0, verbose=False, **kwargs):
    import phate
    if use_rep == 'X':
        data = adata.X
    elif use_rep == 'raw':
        data = adata.raw.X
    elif use_rep in adata.obsm.keys():
        data = adata.obsm[use_rep]
    elif use_rep in adata.uns.keys():
        data = adata.uns[use_rep]['distances']
        kwargs['knn_dist'] = 'precomputed'
    else:
        raise KeyError(f'{use_rep} not found.')
    phate_operator = phate.PHATE(
        knn=knn, decay=decay, t=t, n_pca=n_pca, random_state=random_state, verbose=verbose, **kwargs)
    kc_phate = phate_operator.fit_transform(data)
    slot_name = f'X_phate_{key_added}' if key_added else 'X_phate'
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
        X = lognorm_to_counts(X, adata.obs['n_counts'].values, rounding=True, force=(recover_counts=='force'))
    else:
        X.data = np.round(np.expm1(X.data))

    if groups is not None and isinstance(groups,(list, tuple)):
        k_obs = adata.obs[groupby].isin(groups)
        ad = anndata.AnnData(X=X[k_obs, :], obs=adata.obs.loc[k_obs, selected_variables].copy())
    else:
        ad = anndata.AnnData(X=X, obs=adata.obs[selected_variables].copy())
    ad.var_names = adata.raw.var_names if adata.raw else adata.var_names
    ad.obs[groupby].cat.remove_unused_categories(inplace=True)

    count_matrix = pseudo_bulk(ad, groupby=groupby, FUN=np.sum).astype(np.int32)
    metadata = ad.obs.groupby(groupby).first()
    metadata.index.name = 'sample'

    import tempfile
    x_fh = tempfile.NamedTemporaryFile(suffix='.tsv', delete=False)
    meta_fh = tempfile.NamedTemporaryFile(suffix='.tsv', delete=False)

    count_matrix.to_csv(x_fh.name, sep='\t')
    metadata.to_csv(meta_fh.name, sep='\t')

    import subprocess as sbp
    trmt_opt = f'--trmt {trmt}' if trmt else ''
    covar_opt = f'--covar {",".join(cofactor)}' if cofactor else ''
    cmd = f'diffQuant -o {outprfx} --minfrac 0.99 --nCPU {n_cpu} --factor {condition} --ctrl {ctrl} {trmt_opt} {covar_opt} {x_fh.name} {meta_fh.name}'
    if dry:
        print(cmd)
        return
    try:
        job = sbp.run(cmd, shell=True, capture_output=True)
    except sbp.CalledProcessError as e:
        print(e)
    finally:
        if debug:
            print(job.stdout.decode('utf-8'))
            print(job.stderr.decode('utf-8'))
        import os
        os.remove(x_fh.name)
        os.remove(meta_fh.name)
    de_tbl = pd.read_csv(f'{outprfx}.{trmt}-vs-{ctrl}.txt', sep='\t', index_col=0)
    return de_tbl


def run_cellphonedb(adata, groupby, outpath='./cpdb', group_size=-1, thread=8, dry=False, **subsample_kwargs):
    import os
    outdir, outname = os.path.split(outpath)
    if not outdir:
        outdir = '.'
    os.makedirs(outdir, exist_ok=True)

    import tempfile
    tmp_h5ad = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    if group_size > 0:
        adata = subsample(adata, fraction=1, groupby=groupby, max_n=group_size, min_n=group_size, **subsample_kwargs)
    adata.write(tmp_h5ad.name, compression='lzf')

    tmp_obs = f'{outpath}.obs.tsv'
    adata.obs[[groupby]].to_csv(tmp_obs, sep='\t')

    import subprocess as sbp
    cmd = (f'cellphonedb method statistical_analysis --counts-data gene_name --output-format tsv'
           f'--project-name {outname} --output-path {outdir} --threads {thread}'
           f'{tmp_obs} {tmp_h5ad.name}')
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
        use_rep='raw',
        organism='human',
):

    if use_rep == 'raw':
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif use_rep == 'X':
        X = adata.X
        var_names = adata.var_names
    elif use_rep in adata.layers.keys():
        X = adata.layers[X]
        var_names = adata.var_names
    else:
        raise NotImplementedError

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rBase = importr('base')
    rCC = importr('CellChat')
    rMatrix = importr('Matrix')
    rPbapply = importr('pbapply')
    rPbapply.pboptions(type='none')

    cellchat_db = ro.r(f'CellChatDB.{organism}')

    if sp.issparse(X):
        X = X.tocoo()
        r_X = rMatrix.sparseMatrix(
            i=numpy2ri.py2rpy(X.col+1),
            j=numpy2ri.py2rpy(X.row+1),
            x=rBase.as_numeric(numpy2ri.py2rpy(X.data)),
            dims=rBase.as_integer(list(X.shape[::-1])),
            dimnames=rBase.list(
                rBase.as_character(numpy2ri.py2rpy(var_names.values)),
                rBase.as_character(numpy2ri.py2rpy(adata.obs_names.values))
            )
        )
    else:
        r_X = rBase.matrix(
            numpy2ri.py2rpy(X),
            dimnames=rBase.list(
                rBase.as_character(numpy2ri.py2rpy(var_names.values)),
                rBase.as_character(numpy2ri.py2rpy(adata.obs_names.values))
            )
        )

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_obs = ro.conversion.py2rpy(adata.obs[[groupby]])

    cellchat = rCC.createCellChat(object=r_X, meta=r_obs, group_by=groupby)

    cellchat_db_secret = rCC.subsetDB(cellchat_db, search='Secreted Signaling')
    cellchat.slots['DB'] = cellchat_db_secret
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
        use_rep='raw',
        cat_covar=[],
        num_covar=[],
        extra_term=None,
        outprfx=None,
):
    if use_rep == 'raw':
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif use_rep == 'X':
        X = adata.X
        var_names = adata.var_names
    elif use_rep in adata.layers.keys():
        X = adata.layers[X]
        var_names = adata.var_names
    else:
        raise NotImplementedError

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rBase = importr('base')
    rUtils = importr('utils')
    rMatrix = importr('Matrix')
    rTibble = importr('tibble')
    rSCTKR = importr('sctkr')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_obs = ro.conversion.py2rpy(adata.obs)

    ctca_result = rSCTKR.CellTypeCompositionAnalysis(
        rTibble.as_tibble(r_obs),
        sample, celltype,
        cat_covar, num_covar,
        extra_term=extra_term,
        save=outprfx
    )

    return ctca_result


def write_mtx(adata, fname_prefix='', var=None, obs=None, use_raw=False):
    """Export AnnData object to mtx formt
    * Parameters
        + adata : AnnData
        An AnnData object
        + fname_prefix : str
        Prefix of the exported files. If not empty and not ending with '/' or '_',
        a '_' will be appended. Full names will be <fname_prefix>matrix.mtx,
        <fname_prefix>genes.tsv, <fname_prefix>barcodes.tsv
        + var : list
        A list of column names to be exported to gene table
        + obs : list
        A list of column names to be exported to barcode/cell table
    """
    if fname_prefix and not (fname_prefix.endswith('/') or fname_prefix.endswith('_')):
        fname_prefix = fname_prefix + '_'
    if var is None:
        var = []
    if obs is None:
        obs = []
    adata_obs = adata.obs.copy()
    if use_raw:
        adata = adata.raw
    obs = list(set(obs) & set(adata_obs.columns))
    var = list(set(var) & set(adata.var.columns))

    import scipy.sparse as sp
    mat = sp.coo_matrix(adata.X)
    n_obs, n_var = mat.shape
    n_entry = len(mat.data)
    header = '%%MatrixMarket matrix coordinate real general\n%\n{} {} {}\n'.format(
        n_var, n_obs, n_entry)
    df = pd.DataFrame({'col': mat.col + 1, 'row': mat.row + 1, 'data': mat.data})
    mtx_fname = fname_prefix + 'matrix.mtx'
    gene_fname = fname_prefix + 'genes.tsv'
    barcode_fname = fname_prefix + 'barcodes.tsv'
    with open(mtx_fname, 'a') as fh:
        fh.write(header)
        df.to_csv(fh, sep=' ', header=False, index=False)

    obs_df = adata_obs[obs].reset_index(level=0)
    obs_df.to_csv(barcode_fname, sep='\t', header=False, index=False)
    var_df = adata.var[var].reset_index(level=0)
    if not var:
        var_df['gene'] = var_df['index']
    var_df.to_csv(gene_fname, sep='\t', header=False, index=False)


def write_table(
        adata, outdir='.',
        slots=('X', 'obs', 'var', 'obsm', 'varm', 'raw.X', 'raw.var'),
        fmt='tsv', transpose=False, compression=False,
        X_dtype=None, raw_X_dtype=None, obs_columns=None, var_columns=None, obsm_keys=None, varm_keys=None, raw_var_columns=None,
):
    import os
    if not outdir:
        raise ValueError('`outdir` cannot be empty')
    outdir = outdir.rstrip('/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sep = ',' if fmt == 'csv' else '\t'
    suffix = '.gz' if compression else ''

    for slot in slots:
        if slot == 'X' or slot == 'raw.X':
            ad = adata.raw if slot == 'raw.X' else adata
            dtype = raw_X_dtype if slot == 'raw.X' else X_dtype
            if dtype is None:
                dtype = np.float32
            X = ad.X.T if transpose else ad.X
            if sp.issparse(X):
                X = X.toarray()
            if dtype in (int, np.int16, np.int32, np.int64):
                X = np.round(X)
            if transpose:
                df = pd.DataFrame(X, index=ad.var_names.values, columns=adata.obs_names.values, dtype=dtype)
                df.index.name = 'Gene'
            else:
                df = pd.DataFrame(X, index=adata.obs_names.values, columns=ad.var_names.values, dtype=dtype)
                df.index.name = 'Cell'
            df.to_csv(f'{outdir}/{slot}.{fmt}{suffix}', sep=sep)
        elif (slot == 'obs' or slot == 'var') and getattr(adata, slot) is not None:
            df_columns = obs_columns if slot == 'obs' else var_columns
            if df_columns is None:
                df_columns = df.columns.to_list()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            df = getattr(adata, slot)[df_columns]
            df.index.name = 'Cell' if slot == 'obs' else 'Gene'
            df.to_csv(f'{outdir}/{slot}.{fmt}{suffix}', sep=sep)
        elif slot == 'raw.var' and adata.raw is not None and adata.raw.var is not None:
            df_columns = raw_var_columns
            if df_columns is None:
                df_columns = df.columns.to_list()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            df = adata.raw.var[df_columns]
            df.index.name = 'Gene'
            df.to_csv(f'{outdir}/{slot}.{fmt}{suffix}', sep=sep)
        elif (slot == 'obsm' or slot == 'varm') and getattr(adata, slot) is not None:
            obj_keys = obsm_keys if slot == 'obsm' else varm_keys
            if obj_keys is None:
                obj_keys = obj.keys()
            elif isinstance(df_columns, str):
                df_columns = [df_columns]
            obj = getattr(adata, slot)
            for k in obj_keys:
                X = obj[k]
                ind = adata.obs_names.values if slot == 'obsm' else adata.var_names.values
                col = [f'{k}_{i}' for i in range(X.shape[1])]
                df =  pd.DataFrame(X, index=ind, columns=col, dtype=np.float32)
                df.index.name = 'Cell' if slot == 'obsm' else 'Gene'
                df.to_csv(f'{outdir}/{slot}.{k}.{fmt}{suffix}', sep=sep)


def split_by_group(adata, groupby, out_type='dict'):
    if groupby not in adata.obs.columns:
        raise KeyError(f'{groupby} is not a valid obs annotation.')
    groups = sorted(list(adata.obs[groupby].unique()))
    if out_type == 'dict':
        out_adatas = {}
        for grp in groups:
            out_adatas[grp] = adata[adata.obs[groupby] == grp, :].copy()
    elif out_type == 'list':
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
        columns = [k for k in ad.obs.columns if ad.obs[k].dtype.name == 'category']
    for k in columns:
        summary = ad.obs[k].value_counts()
        print(f'{k}, {len(summary.to_dict())}')
        print(summary)
        print('\n')


def regroup(adata, groupby, regroups):
    if groupby not in adata.obs.columns:
        raise KeyError(f'{groupby} is not a valid obs annotation.')
    groups = adata.obs[groupby].astype(str)
    new_groups = groups.copy()
    for new_grp, old_grps in regroups.items():
        if isinstance(old_grps, (list, tuple)):
            for grp in old_grps:
                new_groups[groups == grp] = new_grp
        else:
            new_groups[groups == old_grps] = new_grp
    regroup_keys = [g for g in regroups.keys() if g in set(new_groups.unique())]
    new_groups = new_groups.astype('category')
    if len(regroup_keys) == len(new_groups.cat.categories):
        new_groups = new_groups.cat.reorder_categories(regroup_keys)
    return new_groups


def subsample(adata, fraction, groupby=None, min_n=0, max_n=10000, method='random', index_only=False, random_state=0):
    if method not in ('random', 'top'):
        raise NotImplementedError(f'method={method} unsupported')
    if groupby:
        if groupby not in adata.obs.columns:
            raise KeyError(f'{groupby} is not a valid obs annotation.')
        groups = adata.obs[groupby].unique()
        n_obs_per_group = {}
        sampled_obs_names = []
        for grp in groups:
            k = adata.obs[groupby] == grp
            grp_size = sum(k)
            ds_grp_size = int(min(
                max_n, max(np.ceil(grp_size * fraction), min(min_n, grp_size))))
            if method == 'top':
                idx = np.argsort(-adata.obs.loc[k, 'n_counts'].values)[0:ds_grp_size]
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
    else:
        return adata[adata.obs_names.isin(sampled_obs_names)].copy()


def pseudo_bulk(
        adata, groupby, use_rep='X', highly_variable=False, FUN=np.mean):
    """Make pseudo bulk data from grouped sc data
    """
    if adata.obs[groupby].dtype.name == 'category':
        group_attr = adata.obs[groupby].values
        groups = adata.obs[groupby].cat.categories.values
    else:
        group_attr = adata.obs[groupby].astype(str).values
        groups = np.unique(group_attr)
    n_level = len(groups)
    if highly_variable:
        if isinstance(highly_variable, (list, tuple)):
            if use_rep == 'raw':
                k_hv = adata.raw.var_names.isin(highly_variable)
            else:
                k_hv = adata.var_names.isin(highly_variable)
        else:
            k_hv = adata.var['highly_variable'].values
    if use_rep == 'X':
        x = adata.X
        features = adata.var_names.values
        if highly_variable:
            x = x[:, k_hv]
            features = features[k_hv]
    elif use_rep == 'raw':
        x = adata.raw.X
        features = adata.raw.var_names.values
        if highly_variable:
            x = x[:, k_hv]
            features = features[k_hv]
    elif use_rep in adata.layers.keys():
        x = adata.layers[use_rep]
        features = adata.var_names.values
        if highly_variable:
            x = x[:, k_hv]
            features = features[k_hv]
    elif use_rep in adata.obsm.keys():
        x = adata.obsm[use_rep]
        features = np.arange(x.shape[1])
    elif (isinstance(use_rep, np.ndarray) and
            use_rep.shape[0] == adata.shape[0]):
        x = use_rep
        features = np.arange(x.shape[1])
    else:
        raise KeyError(f'{use_rep} invalid.')
    summarised = np.zeros((n_level, x.shape[1]))
    for i, grp in enumerate(groups):
        k_grp = group_attr == grp
        if sp.issparse(x):
            summarised[i] = FUN(x[k_grp, :], axis=0)
        else:
            summarised[i] = FUN(x[k_grp, :], axis=0, keepdims=True)
    return pd.DataFrame(summarised.T, columns=groups, index=features)


def score_msigdb_genesets(adata, groupby, msigdb, use_rep='raw', pattern=r'^HALLMARK_', use_pseudobulk=False, pb_kw={}):
    import re
    def read_msigdb(msigdb, pattern):
        genesets = {}
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        with open(msigdb, 'r') as fh:
            for line in fh:
                fields = line.rstrip().split('\t')
                geneset_name = fields[0]
                geneset = fields[2:]
                if re.search(pattern, geneset_name):
                    genesets[geneset_name] = geneset
        print(f'read {len(genesets)} genesets')
        return genesets

    genesets = read_msigdb(msigdb, pattern)
    if use_pseudobulk:
        X = pseudo_bulk(adata, groupby, use_rep=use_rep, **pb_kw).T
        ad = anndata.AnnData(X=X)
    else:
        if use_rep == raw:
            X = adata.raw.X
        elif use_rep == 'X':
            X = adata.X
        elif use_rep in adata.layers.keys():
            X = adata.layers[use_rep]
        else:
            raise NotImplementedError
        ad = anndata.AnnData(X, obs=adata.obs[[groupby]])

    for geneset_name, geneset in genesets.items():
        sc.tl.score_genes(ad, geneset, ctrl_size=max(50, len(geneset)), use_raw=(use_rep == 'raw'), score_name=geneset_name)

    del genesets

    return ad


def write_cellxgene_object(adata, output, **kwargs):
    ad = restore_adata(adata, restore_type='norm', use_raw=adata.raw, **kwargs)
    ad.write(output, compression='lzf')
    return ad
