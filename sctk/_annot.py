"""
Transfer annotation functions
"""

import os
import sys
import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp
import pandas as pd
import joblib
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import ClassifierMixin

try:
    from sklearn.utils.testing import ignore_warnings
except Exception:
    from sklearn.utils._testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def LR_train(
    adata,
    groupby,
    use_rep="raw",
    use_hvg=False,
    use_pseudobulk=False,
    max_pass=20,
    downsample_to=None,
    save=None,
    model=None,
    **kwargs,
):
    if downsample_to is not None:
        if (
            isinstance(downsample_to, (tuple, list))
            and isinstance(downsample_to[0], int)
            and isinstance(downsample_to[1], str)
        ):
            from ._utils import subsample

            ad = subsample(adata, 1, groupby=groupby, max_n=downsample_to[0])
            ad.write(downsample_to[1], compression="lzf")
        else:
            raise ValueError("`downsample_to` needs to be (int[N], str[h5ad])")
    else:
        ad = adata

    groupby_var = ad.obs[groupby].cat.remove_unused_categories()
    Y = groupby_var.astype(str)

    if use_rep == "raw":
        X = ad.raw.X
        features = ad.raw.var_names.values
    elif use_rep == "X":
        X = ad.X
        features = ad.var_names.values
        if use_hvg and "highly_variable" in ad.var.keys():
            k_hvg = ad.var["highly_variable"].values
            X = X[:, k_hvg]
            features = features[k_hvg]
    elif use_rep in ad.obsm.keys():
        X = ad.obsm[use_rep]
        features = np.array([f"V{i+1}" for i in range(X.shape[1])])
    else:
        raise KeyError(f"{use_rep}: invalid <use_rep>")

    if use_pseudobulk:
        if sp.issparse(X):
            summarised = np.zeros((Y.unique().size, X.shape[1]))
            for i, grp in enumerate(groupby_var.cat.categories):
                k_grp = np.where(Y == grp)[0]
                summarised[i] = np.mean(X[k_grp, :], axis=0)
            X = summarised
        else:
            X = npg.aggregate(groupby_var.cat.codes, X, axis=0, func="mean")
        Y = groupby_var.cat.categories.values

    lr = (
        model
        if model is not None
        else LogisticRegression(
            penalty="l2", C=0.1, solver="saga", warm_start=True, n_jobs=-1, **kwargs
        )
    )
    n_pass = 0
    while n_pass < max_pass:
        lr.fit(X, Y)
        n_pass += 1
        if lr.n_iter_ < 100:
            break
    if lr.n_iter_ >= 100:
        print("training of LR model failed to converge", file=sys.stderr)
    lr.features = features
    if save:
        outdir = os.path.dirname(save)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        joblib.dump(lr, save)
    return lr


def LR_predict(
    adata,
    model,
    use_rep="raw",
    use_pseudobulk=False,
    groupby=None,
    feature=None,
    key_added=None,
    truth=None,
    return_predict=False,
    min_prob=None,
):
    if use_rep == "raw":
        X = adata.raw.X
        features = adata.raw.var_names if feature is None else adata.raw.var[feature]
    elif use_rep == "X":
        X = adata.X
        features = adata.var_names if feature is None else adata.var[feature]
    elif use_rep in adata.obsm.keys():
        X = adata.obsm[use_rep]
        features = np.array([f"V{i+1}" for i in range(X.shape[1])])
    else:
        raise KeyError(f"{use_rep}: invalid <use_rep>")
    features = pd.Series(features)

    if isinstance(model, str) and os.path.exists(model):
        lr = joblib.load(model)
    elif isinstance(model, ClassifierMixin):
        lr = deepcopy(model)
    else:
        raise ValueError(f"{model}: invalid LR model")
    if getattr(lr, "features", None) is None:
        if lr.n_features_in_ == features.size:
            lr.features = features.values
        else:
            raise ValueError(f"{model}: LR model has no feature names and unmatched size")

    k_x = features.isin(list(lr.features))
    print(f"{k_x.sum()} features used for prediction", file=sys.stderr)
    k_x_idx = np.where(k_x)[0]
    X = X[:, k_x_idx]
    features = features[k_x]

    ad_ft = (
        pd.DataFrame(features.values, columns=["ad_features"])
        .reset_index()
        .rename(columns={"index": "ad_idx"})
    )
    lr_ft = (
        pd.DataFrame(lr.features, columns=["lr_features"])
        .reset_index()
        .rename(columns={"index": "lr_idx"})
    )
    lr_idx = (
        lr_ft.merge(ad_ft, left_on="lr_features", right_on="ad_features")
        .sort_values(by="ad_idx")
        .lr_idx.values
    )

    lr.n_features_in_ = lr_idx.size
    lr.features = lr.features[lr_idx]
    lr.coef_ = lr.coef_[:, lr_idx]

    if use_pseudobulk:
        if not groupby or groupby not in adata.obs.columns:
            raise ValueError("missing or invalid `groupby`")
        groupby_var = adata.obs[groupby].cat.remove_unused_categories()
        summarised = np.zeros((groupby_var.cat.categories.size, X.shape[1]))
        for i, grp in enumerate(groupby_var.cat.categories):
            k_grp = np.where(groupby_var == grp)[0]
            if sp.issparse(X):
                summarised[i] = np.mean(X[k_grp, :], axis=0)
            else:
                summarised[i] = np.mean(X[k_grp, :], axis=0, keepdims=True)
        X = summarised

    ret = {}
    Y_predict = lr.predict(X)
    Y_prob = lr.predict_proba(X)
    max_Y_prob = Y_prob.max(axis=1)
    if use_pseudobulk:
        tmp_groupby = adata.obs[groupby].astype(str)
        tmp_prob = np.zeros(tmp_groupby.size)
        tmp_predict = tmp_prob.astype(str)
        for i, ct in enumerate(adata.obs[groupby].cat.categories):
            tmp_prob[tmp_groupby == ct] = max_Y_prob[i]
            tmp_predict[tmp_groupby == ct] = Y_predict[i]
        max_Y_prob = tmp_prob
        Y_predict = tmp_predict
    if min_prob is not None:
        Y_predict[max_Y_prob < min_prob] = "Uncertain"
    if key_added:
        adata.obs[key_added] = Y_predict
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        adata.obs[f"{key_added}_prob"] = max_Y_prob
    ret["label"] = Y_predict
    ret["prob"] = pd.DataFrame(
        Y_prob,
        index=adata.obs[groupby].cat.categories if use_pseudobulk else adata.obs_names,
        columns=lr.classes_,
    )

    if truth:
        Y_truth = adata.obs[truth].astype(str)
        ret["accuracy"] = (Y_predict == Y_truth).sum() / Y_predict.size
        ret["adjusted_rand_score"] = adjusted_rand_score(Y_truth, Y_predict)

    if return_predict:
        return ret


def annotate(adata, groupby, label, normalise_label=True, threshold=0.85, max_entry=None):
    """Annotate a clustering based on a matrix of values

    + groupby: the clustering to be annotated
    + label : cell-level annotation
    """
    annot_mat = pd.crosstab(adata.obs[groupby], adata.obs[label])
    group_size = annot_mat.sum(axis=1).values
    group_prop = annot_mat / group_size[:, np.newaxis]
    if normalise_label:
        label_size = group_prop.sum(axis=0).values
        label_prop = group_prop / label_size[np.newaxis, :]
    else:
        label_prop = group_prop
    v_max = label_prop.values.max(axis=1)

    annot_dict = {}
    for i, row in label_prop.iterrows():
        idx = np.where(row.values > row.values.max() * threshold)[0]
        od = np.argsort(row.values[idx])
        if max_entry is not None:
            max_entry = min(max_entry, len(idx))
            entries = row[idx[od]][0:max_entry]
        else:
            entries = row[idx[od]]
        gl = ";".join(entries.index.values)
        if gl not in annot_dict:
            annot_dict[gl] = []
        annot_dict[gl].append(i)
    return annot_dict


def get_top_LR_features(model, group, n_top=10):
    return (
        pd.DataFrame(
            {"gene": model.features, "coef": model.coef_[model.classes_ == group, :].flatten()}
        )
        .sort_values("coef", ascending=False)
        .head(n_top)
    )
