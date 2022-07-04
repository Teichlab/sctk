"""
scanpy filter
"""

import logging
import re
import numpy as np
import scanpy as sc


def filter_anndata(
    adata,
    gene_name="index",
    list_attr=False,
    param=None,
    category=None,
    subset=None,
    force_recalc=False,
):
    """
    Wrapper function for sc.pp.filter_cells() and sc.pp.filter_genes(), mainly
    for supporting arbitrary filtering
    """
    param = [] if param is None else param
    category = [] if category is None else category
    subset = [] if subset is None else subset

    logging.debug("--gene-name=%s", gene_name)
    logging.debug("--param=%s", param)
    logging.debug("--category=%s", category)
    logging.debug("--subset=%s", subset)

    if "mito" not in adata.var.keys() and gene_name:
        try:
            gene_names = getattr(adata.var, gene_name)
            k_mito = gene_names.str.startswith("MT-")
            if k_mito.sum() > 0:
                adata.var["mito"] = k_mito
                adata.var["mito"] = adata.var["mito"].astype("category")
            else:
                logging.warning(
                    "No MT genes found, skip calculating " "expression of mitochondria genes"
                )
        except AttributeError:
            logging.warning(
                "Specified gene column [%s] not found, skip calculating "
                "expression of mitochondria genes",
                gene_name,
            )

    attributes = _get_attributes(adata)
    if list_attr:
        print(_repr_obj(attributes))
        return 0

    conditions, qc_vars, pct_top = _get_filter_conditions(attributes, param, category, subset)

    layer = "counts" if "counts" in adata.layers.keys() else None
    obs_columns = adata.obs.columns
    for qv in qc_vars:
        if f"pct_counts_{qv}" in obs_columns and not force_recalc:
            logging.warning("`pct_counts_%s` exists, not overwriting " "without --force-recalc", qv)
            qc_vars.remove(qv)
    for pt in pct_top:
        if f"pct_counts_in_top_{pt}_genes" in obs_columns and not force_recalc:
            logging.warning("`pct_counts_%s` exists, not overwriting " "without --force-recalc", pt)
            pct_top.remove(pt)
    sc.pp.calculate_qc_metrics(
        adata, layer=layer, qc_vars=qc_vars, percent_top=pct_top, inplace=True
    )
    adata.obs["n_counts"] = adata.obs["total_counts"]
    adata.obs["n_genes"] = adata.obs["n_genes_by_counts"]
    adata.var["n_counts"] = adata.var["total_counts"]
    adata.var["n_cells"] = adata.var["n_cells_by_counts"]

    k_cell = np.ones(len(adata.obs)).astype(bool)
    for cond in conditions["c"]["numerical"]:
        name, vmin, vmax = cond
        attr = adata.obs[name]
        k_cell = k_cell & (attr >= vmin) & (attr <= vmax)

    for cond in conditions["c"]["categorical"]:
        name, values = cond
        attr = getattr(adata.obs, name).astype(str)
        if values[0].startswith("!"):
            values[0] = values[0][1:]
            k_cell = k_cell & (~attr.isin(values))
        else:
            k_cell = k_cell & attr.isin(values)

    k_gene = np.ones(len(adata.var)).astype(bool)
    for cond in conditions["g"]["numerical"]:
        name, vmin, vmax = cond
        attr = adata.var[name]
        k_gene = k_gene & (attr >= vmin) & (attr <= vmax)

    for cond in conditions["g"]["categorical"]:
        name, values = cond
        attr = getattr(adata.var, name).astype(str)
        if values[0].startswith("!"):
            values[0] = values[0][1:]
            k_gene = k_gene & ~(attr.isin(values))
        else:
            k_gene = k_gene & attr.isin(values)

    adata._inplace_subset_obs(k_cell)
    adata._inplace_subset_var(k_gene)


def _get_attributes(adata):
    attributes = {
        "c": {
            "numerical": [],
            "categorical": ["index"],
            "bool": [],
        },
        "g": {
            "numerical": [],
            "categorical": ["index"],
            "bool": [],
        },
    }

    for attr, dtype in adata.obs.dtypes.to_dict().items():
        typ = dtype.kind
        if typ == "O":
            if dtype.name == "category" and dtype.categories.is_boolean():
                attributes["c"]["bool"].append(attr)
            attributes["c"]["categorical"].append(attr)
        elif typ in ("i", "f", "u"):
            attributes["c"]["numerical"].append(attr)
        elif typ == "b":
            attributes["c"]["bool"].append(attr)
            attributes["c"]["categorical"].append(attr)

    for attr, dtype in adata.var.dtypes.to_dict().items():
        typ = dtype.kind
        if typ == "O":
            if dtype.name == "category" and dtype.categories.is_boolean():
                attributes["g"]["bool"].append(attr)
            attributes["g"]["categorical"].append(attr)
        elif typ in ("i", "f", "u"):
            attributes["g"]["numerical"].append(attr)
        elif typ == "b":
            attributes["g"]["bool"].append(attr)
            attributes["g"]["categorical"].append(attr)

    attributes["c"]["numerical"].extend(
        [
            "n_genes",
            "n_counts",
        ]
    )

    for attr in attributes["g"]["bool"]:
        attr2 = "pct_counts_" + attr
        if attr2 not in adata.obs.columns:
            attr2 += "*"
        attributes["c"]["numerical"].append(attr2)

    attributes["g"]["numerical"].extend(
        [
            "n_cells",
            "n_counts",
            "mean_counts",
            "pct_dropout_by_counts",
        ]
    )
    logging.debug(attributes)
    return attributes


def _attributes_exists(name, attributes, dtype):
    cond_cat = ""
    if name.startswith("c:") or name.startswith("g:"):
        cond_cat, _, cond_name = name.partition(":")
        found = int(cond_name in attributes[cond_cat][dtype])
    else:
        cond_name = name
        if cond_name in attributes["c"][dtype]:
            cond_cat += "c"
        if cond_name in attributes["g"][dtype]:
            cond_cat += "g"
        found = len(cond_cat)
    return found, cond_cat, cond_name


def _get_filter_conditions(attributes, param, category, subset):
    conditions = {
        "c": {
            "numerical": [],
            "categorical": [],
            "bool": [],
        },
        "g": {
            "numerical": [],
            "categorical": [],
            "bool": [],
        },
    }
    percent_top_pattern = re.compile(r"^pct_counts_in_top_(?P<n>\d+)_genes$")
    pct_top = []
    qc_vars_pattern = re.compile(r"^pct_counts_(?P<qc_var>\S+)$")
    qc_vars = []

    for name, vmin, vmax in param:
        found, cond_cat, cond_name = _attributes_exists(name, attributes, "numerical")
        pt_match = percent_top_pattern.match(cond_name)
        qv_match = qc_vars_pattern.match(cond_name)
        if found > 1:
            raise ValueError(f'Ambiguous parameter "{name}" found in ' "both cell and gene table")
        if found < 1:
            if pt_match:
                pct_top.append(int(pt_match["n"]))
                cond_cat = "c"
            elif qv_match and qv_match["qc_var"] in attributes["g"]["bool"]:
                qc_vars.append(qv_match["qc_var"])
                cond_cat = "c"
            else:
                raise ValueError(f'Parameter "{name}" unavailable')
        if pt_match or qv_match:
            vmin *= 100
            vmax *= 100
        conditions[cond_cat]["numerical"].append([cond_name, vmin, vmax])

    for name, values in category + subset:
        found, cond_cat, cond_name = _attributes_exists(name, attributes, "categorical")
        if found > 1:
            raise ValueError(f'Ambiguous attribute "{name}" found in ' "both cell and gene table")
        if found < 1:
            raise ValueError(f'Attribute "{name}" unavailable')
        if not isinstance(values, (list, tuple)):
            fh = values
            values = fh.read().rstrip().split("\n")
            fh.close()
        conditions[cond_cat]["categorical"].append((cond_name, values))

    logging.debug((conditions, qc_vars, pct_top))
    return conditions, qc_vars, sorted(pct_top)


def _repr_obj(obj, padding="  ", level=0):
    if isinstance(obj, dict):
        obj_str = "\n".join(
            [
                "\n".join([padding * level + k + ":", _repr_obj(v, level=level + 1)])
                for k, v in obj.items()
            ]
        )
    elif isinstance(obj, (tuple, list, set)):
        obj_str = "\n".join([_repr_obj(elm, level=level) for elm in obj])
    else:
        obj_str = padding * level + repr(obj)
    return obj_str
