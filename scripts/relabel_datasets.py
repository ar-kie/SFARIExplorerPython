#!/usr/bin/env python3
"""Apply changed dataset names to finished outputs of steps 02-05 without rerunning them.

Step 03 names each cell <id>-<dataset key> and labels obs['dataset'] with the key's display name
(pipeline/config.py, including the CELLxGENE registry). After names change (e.g. `fetch_cellxgene.py
register` now gives every dataset its own name), this rewrites obs/dataset of the step 03-05 h5ad
files in place, from the key in each cell id, and the dataset column of gene_manifest.parquet.
Nothing else in the files changes. Dry run unless --apply.

    python scripts/relabel_datasets.py            # show what would change
    python scripts/relabel_datasets.py --apply
"""
import argparse, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import config
from h5io import read_elem, write_elem

FILES = [config.CONCATENATED_PATH, config.ANNOTATED_PATH, config.FINAL_PATH]


def relabel(ids, labels, names):
    """New label per cell; cells whose label is not a current name get it from the key in their id."""
    new = labels.astype(object).copy()
    stale = ~pd.Series(labels).isin(set(names.values())).to_numpy()
    if not stale.any():
        return new, {}
    ids_stale = pd.Series(ids[stale])
    in_use = set(labels[~stale])  # names already in use cannot be the stale ones
    found = np.full(len(ids_stale), None, dtype=object)
    for key, name in names.items():
        if name in in_use:
            continue
        hit = ids_stale.str.endswith(f"-{key}").to_numpy()
        if (hit & ~pd.isnull(found)).any():
            raise SystemExit(f"cells match more than one dataset key ({key})")
        found[hit] = name
    if pd.isnull(found).any():
        bad = pd.Series(labels[stale][pd.isnull(found)]).value_counts()
        raise SystemExit(f"no dataset key found for cells labelled:\n{bad.to_string()}")
    new[stale] = found
    changes = pd.DataFrame({"old": labels[stale], "new": found}).value_counts().to_dict()
    return new, changes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = p.parse_args()
    names = {k: m["dataset"] for k, m in config.DATASET_META.items()}
    print(f"{len(names)} dataset names in the configuration ({config.N_REGISTERED_DATASETS} from the registry)")

    for path in FILES:
        if not os.path.exists(path):
            print(f"\n{os.path.basename(path)}: not found, skipped"); continue
        with h5py.File(path, "r") as f:
            obs = f["obs"]
            ids = np.asarray(read_elem(obs[obs.attrs["_index"]]), dtype=object)
            labels = np.asarray(read_elem(obs["dataset"]), dtype=object)
        new, changes = relabel(ids, labels, names)
        print(f"\n{os.path.basename(path)}: {len(ids):,} cells")
        if not changes:
            print("  all dataset names current"); continue
        for (old, nw), n in sorted(changes.items()):
            print(f"  {old!r:30s} -> {nw!r:30s} {n:>10,} cells")
        if args.apply:
            with h5py.File(path, "r+") as f:
                write_elem(f, "obs/dataset", pd.Categorical(new))
            print("  written")

    gm = config.GENE_MANIFEST_PATH
    if os.path.exists(gm):
        m = pd.read_parquet(gm)
        target = m["dataset_internal"].map(names).fillna(m["dataset"])
        n = int((target != m["dataset"]).sum())
        print(f"\n{os.path.basename(gm)}: {n:,} rows renamed")
        if n and args.apply:
            m.assign(dataset=target).to_parquet(gm, index=False); print("  written")
    if not args.apply:
        print("\nDry run; rerun with --apply to write.")


if __name__ == "__main__":
    main()
