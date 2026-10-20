#!/usr/bin/env python3
"""Report on a finished build (steps 01-07): job outcomes, cells per dataset, label transfer,
pseudobulk samples and ages, batch correction, developmental binning, key genes.

Read-only, and never loads a whole matrix (a few GB of memory at most). From the repository root:
    python scripts/check_build.py | tee build_report.txt
"""
import glob, json, os, re, sys
import h5py, numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
ROOT = os.environ.get("SFARI_ROOT", "/sc/arion/projects/ad-omics/raphael/SFARI")
os.environ["SFARI_ROOT"] = ROOT
PIPE, DATA = f"{ROOT}/pipeline_output", f"{ROOT}/data"
EX = os.environ.get("SFARI_EXCHANGE_DIR", f"{DATA}/r_exchange")
CONCORD = f"{DATA}/concord"
KEY_GENES = ["SHANK3", "SYNGAP1", "PTEN", "CHD8", "SCN2A", "ARID1B", "FOXP1", "MECP2", "FMR1", "NRXN1"]
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 40)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def guarded(fn):
    def run():
        try: fn()
        except FileNotFoundError as e: print(f"  missing: {e.filename or e}")
        except Exception as e: print(f"  could not check: {type(e).__name__}: {e}")
    return run


def lsf_summary(path):
    t = open(path, errors="replace").read()
    code = re.search(r"Exited with exit code (\d+)", t)
    status = ("ok" if "Successfully completed" in t else "MEMLIMIT" if "TERM_MEMLIMIT" in t
              else f"EXIT {code.group(1)}" if code else "running/unknown")
    mem = re.search(r"Max Memory :\s+([\d.]+) (\w+)", t)
    req = re.search(r"Total Requested Memory :\s+([\d.]+) (\w+)", t)
    rt = re.search(r"Run time :\s+(\d+) sec", t)
    return status, f"{mem.group(1)} {mem.group(2)}" if mem else "?", f"{req.group(1)} {req.group(2)}" if req else "?", \
        f"{int(rt.group(1)) / 3600:.1f} h" if rt else "?"


@guarded
def jobs():
    section("1. Jobs (latest LSF log per step)")
    logs = {f"step0{i}": f"{PIPE}/logs/step0{i}_*.out" for i in range(1, 6)}
    logs.update({"06 concord": f"{REPO}/sfari_concord_*.log", "07 postprocess": f"{REPO}/sfari_postprocess_*.log"})
    print(f"  {'step':<16}{'status':<18}{'max memory':<16}{'requested':<16}{'run time'}")
    for step, pat in logs.items():
        files = sorted(glob.glob(pat), key=os.path.getmtime)
        if not files: print(f"  {step:<16}no log found"); continue
        s, m, r, t = lsf_summary(files[-1])
        print(f"  {step:<16}{s:<18}{m:<16}{r:<16}{t}   {os.path.basename(files[-1])}")
    errs = sorted(glob.glob(f"{REPO}/sfari_postprocess_*.err"), key=os.path.getmtime)
    if errs:
        bad = [l.rstrip() for l in open(errs[-1], errors="replace") if re.search(r"Error|Traceback|Killed", l)]
        if bad: print(f"  {os.path.basename(errs[-1])}: {len(bad)} error lines, last: {bad[-1][:150]}")


@guarded
def datasets():
    section("2. Datasets in steps 01-05 (expected from config vs. present in sfari_final.h5ad)")
    import config
    expected = {k: config.DATASET_META.get(k, {}).get("dataset", k)
                for k in [*config.SYMBOL_DATASETS, *config.ENSEMBL_DATASETS, *config.MTX_DATASETS]}
    print(f"  configured: {len(expected)} ({config.N_REGISTERED_DATASETS} from the CELLxGENE registry)")
    insp = json.load(open(f"{PIPE}/dataset_inspection.json"))
    for k in expected:
        r = insp.get(k, {})
        if "error" in r or not r.get("recommended_source"):
            print(f"  SKIPPED in step 01: {k}: {r.get('error', 'no raw-count source found')}")
    from h5io import read_elem
    with h5py.File(f"{PIPE}/sfari_final.h5ad", "r") as f:
        cols = [c for c in ["dataset", "organism", "cell_type_supercategory", "cell_type_merged"] if f"obs/{c}" in f]
        obs = pd.DataFrame({c: read_elem(f[f"obs/{c}"]) for c in cols})
    n = obs.groupby("dataset", observed=True).size()
    sc_ = obs.cell_type_supercategory.astype(str)
    share = lambda m: (100 * obs.assign(u=m).groupby("dataset", observed=True)["u"].mean()).round(1).astype(str) + " %"
    t = pd.DataFrame({"organism": obs.groupby("dataset", observed=True)["organism"].first(), "cells": n,
                      "no author label": share(sc_.isin(["Unknown", ""])),
                      "'Other'": share(sc_ == "Other")}).sort_values(["organism", "cells"])
    missing = sorted(set(expected.values()) - set(n.index))
    mixed = obs.groupby("dataset", observed=True)["organism"].nunique()
    for ds in mixed[mixed > 1].index:
        print(f"  SEVERAL SPECIES under one name: {ds} -> run fetch_cellxgene.py register, then scripts/relabel_datasets.py")
    print(t.to_string()); print(f"  total: {n.sum():,} cells in {len(n)} datasets")
    if missing: print(f"  MISSING from the final object: {missing}")
    if "cell_type_merged" in obs:
        other = obs[sc_ == "Other"]
        big = other.groupby("dataset", observed=True).size() / n
        for ds in big[big > 0.02].index:
            top = other.loc[other.dataset == ds, "cell_type_merged"].astype(str).value_counts().head(10)
            print(f"  author labels mapped to 'Other' in {ds}: " + ", ".join(f"{k} ({v:,})" for k, v in top.items()))


@guarded
def concord():
    section("3. CONCORD integration and label transfer (data/concord)")
    bi = json.load(open(f"{CONCORD}/build_info.json"))
    i = bi.get("integration", {})
    print(f"  {bi.get('n_cells', 0):,} cells, {i.get('n_features')} features, latent {i.get('latent_dim')}, "
          f"{i.get('n_epochs')} epochs on {i.get('device')}; held-out accuracy {i.get('holdout_accuracy')}")
    qc = pd.read_csv(f"{CONCORD}/concord_label_transfer_qc.csv", index_col=0)
    per_ds = qc[qc.index.str.startswith("dataset: ")][["accuracy", "n_holdout"]].sort_values("accuracy")
    per_ds.index = per_ds.index.str.replace("dataset: ", "")
    print("  held-out accuracy per dataset (author label hidden, CONCORD prediction compared):")
    print(per_ds.round(3).to_string())
    ct = qc[~qc.index.str.startswith("dataset: ") & ~qc.index.isin(["accuracy", "macro avg", "weighted avg"])]
    print("  cell types, lowest F1 first:")
    print(ct[["precision", "recall", "f1-score", "support"]].sort_values("f1-score").round(3).head(12).to_string())
    a = pd.read_parquet(f"{CONCORD}/concord_cell_annotations.parquet", columns=["dataset", "label_source"])
    src = pd.crosstab(a["dataset"], a["label_source"], normalize="index").mul(100).round(1)
    print("  label source per dataset (%):"); print(src.to_string())
    pf = f"{CONCORD}/concord_precision_filter.csv"
    if os.path.exists(pf):
        print(f"  precision filter ({bi.get('integration', {}).get('label_filter', '')}):")
        print(pd.read_csv(pf).round(3).to_string(index=False))


@guarded
def pseudobulk():
    section("4. Pseudobulk samples and numeric ages (r_exchange)")
    m = pd.read_csv(f"{EX}/pseudobulk_meta_numeric_time.csv")
    g = m.groupby(["organism", "dataset", "sample_type"], dropna=False)
    t = pd.DataFrame({"samples": g.size(), "cell types": g["cell_type"].nunique(), "time points": g["timepoint"].nunique(),
                      "no numeric age": g["numeric_time"].apply(lambda s: int(s.isna().sum())),
                      "age min": g["numeric_time"].min(), "age max": g["numeric_time"].max()})
    print(t.to_string()); print(f"  total: {len(m):,} pseudobulk samples")
    bad = m[m["numeric_time"].isna()].groupby("dataset")["timepoint"].unique()
    for ds, tps in bad.items(): print(f"  unparsed ages in {ds}: {list(map(str, tps))[:8]}")
    with open(f"{EX}/pseudobulk_counts.csv") as fh:
        n_genes = len(fh.readline().split(",")) - 1
        n_rows = sum(1 for _ in fh)
    print(f"  pseudobulk_counts.csv: {n_rows:,} samples x {n_genes:,} genes")


@guarded
def correction():
    section("5. Within-species correction (corrected_expression.csv)")
    path = f"{EX}/corrected_expression.csv"
    with open(path) as fh: genes = fh.readline().rstrip("\n").split(",")[1:]
    ids = pd.read_csv(path, usecols=[0]).iloc[:, 0]
    m = pd.read_csv(f"{EX}/pseudobulk_meta_numeric_time.csv")
    kept = m["sample_id"].isin(set(ids))
    print(f"  {len(ids):,} samples x {len(genes):,} genes; {kept.mean():.1%} of pseudobulk samples retained")
    lost = m[~kept].groupby("dataset").size()
    if len(lost): print("  samples dropped by the correction, per dataset:"); print(lost.to_string())
    genes = {g.strip('"') for g in genes}
    print(f"  key genes absent: {[g for g in KEY_GENES if g not in genes] or 'none'}")
    vp = f"{EX}/variance_partition_comparison.csv"
    if os.path.exists(vp): print("  variance partition before/after:"); print(pd.read_csv(vp).round(3).to_string(index=False))


@guarded
def binning():
    section("6. App data: developmental time (data/app/temporal_meta.parquet)")
    t = pd.read_parquet(f"{DATA}/app/temporal_meta.parquet")
    print(f"  {len(t):,} temporal groups; files in data/app: {len(os.listdir(f'{DATA}/app'))}")
    r = t.groupby(["species", "sample_type"]).agg(datasets=("tissue", "nunique"), groups=("row_id", "size"),
                                                   rel_min=("relative_dev_time", "min"), rel_max=("relative_dev_time", "max"))
    print(r.round(3).to_string())
    print(t.groupby(["species", "sample_type", "time_order", "time_bin"]).size().rename("groups").to_string())
    print("  unified stages:", t.groupby("unified_stage").size().to_dict())


if __name__ == "__main__":
    print(f"SFARI_ROOT = {ROOT}\nexchange dir = {EX}")
    for check in (jobs, datasets, concord, pseudobulk, correction, binning): check()
