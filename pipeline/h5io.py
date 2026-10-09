"""Read and write parts of .h5ad files without loading whole datasets (used by steps 01-05).

anndata.read_h5ad(path) loads every matrix, and even backed mode ('r') loads layers, obsm, obsp and
uns into memory. These helpers read only the element a step needs (gene names, one count matrix,
obs), sample count values from disk, and write large matrices to disk one block at a time.
"""
import os, zipfile
import h5py, numpy as np, pandas as pd, anndata as ad
from scipy import sparse

try:  # anndata >= 0.11
    from anndata.io import read_elem, write_elem
except ImportError:  # anndata 0.10
    from anndata.experimental import read_elem, write_elem

SCAN_CHUNK = 100_000_000  # entries per block when a matrix has to be scanned


def is_legacy(f):
    """True for files written by anndata < 0.7 (no element encodings); read those with anndata."""
    obs = f.get("obs")
    return not (isinstance(obs, h5py.Group) and "encoding-type" in obs.attrs)


def _index_len(g):
    return g[g.attrs["_index"]].shape[0]


def n_obs(f):
    return _index_len(f["obs"])


def has_raw(f):
    return "raw" in f and "X" in f["raw"]


def var_names(f, raw=False):
    """Gene names of .var (or .raw.var) as strings."""
    return read_elem(f["raw/var" if raw else "var"]).index.astype(str)


def layer_names(f):
    return list(f["layers"].keys()) if "layers" in f else []


def read_matrix(path, key):
    """One matrix ('X', 'raw/X' or 'layers/<name>'), nothing else."""
    with h5py.File(path, "r") as f:
        if not is_legacy(f):
            return read_elem(f[key])
    print(f"    (old h5ad format: loading {os.path.basename(path)} with anndata)")
    adata = ad.read_h5ad(path)
    if key == "X": return adata.X
    if key == "raw/X": return adata.raw.X
    return adata.layers[key.split("/", 1)[1]]


def read_obs(path):
    with h5py.File(path, "r") as f:
        if not is_legacy(f):
            return read_elem(f["obs"])
    return ad.read_h5ad(path, backed="r").obs


def sample_rows(el, n):
    """Nonzero values of up to n random rows of a matrix stored in an h5ad, and its dtype.

    Draws the rows with np.random.choice exactly like an in-memory X[idx, :], so a seeded run picks
    the same rows. CSR and dense matrices are read row by row; CSC matrices are scanned in blocks.
    """
    if isinstance(el, h5py.Dataset):
        n_rows = el.shape[0]
        idx = np.random.choice(n_rows, min(n, n_rows), replace=False)
        vals = el[np.sort(idx), ...].ravel()
        return vals[vals != 0], str(el.dtype)
    fmt = el.attrs.get("encoding-type", el.attrs.get("h5sparse_format", ""))
    fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
    shape = el.attrs.get("shape", el.attrs.get("h5sparse_shape"))
    n_rows = int(shape[0])
    idx = np.random.choice(n_rows, min(n, n_rows), replace=False)
    data, indptr = el["data"], el["indptr"][...]
    if fmt.startswith("csr"):
        vals = [data[indptr[i]:indptr[i + 1]] for i in np.sort(idx)]
    else:  # csc: rows are spread over the whole matrix
        wanted = np.zeros(n_rows, dtype=bool); wanted[idx] = True
        vals = []
        for s in range(0, len(data), SCAN_CHUNK):
            m = wanted[el["indices"][s:s + SCAN_CHUNK]]
            if m.any(): vals.append(data[s:s + SCAN_CHUNK][m])
    vals = np.concatenate(vals) if vals else np.array([], dtype=data.dtype)
    return vals[vals != 0], str(data.dtype)


def remap_columns(X, oi, ni, n_cols):
    """CSR matrix (n_obs x n_cols) holding column oi[k] of X in column ni[k]; other columns dropped.

    Same result as building it from X[:, oi] via COO, with fewer copies of the matrix in memory.
    """
    X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
    colmap = np.full(X.shape[1], -1, dtype=np.int32 if n_cols < 2**31 else np.int64)
    colmap[np.asarray(oi, dtype=np.int64)] = ni
    new = colmap[X.indices]
    keep = new >= 0
    kept_before = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
    indptr = kept_before[X.indptr]
    del kept_before
    out = sparse.csr_matrix((X.data[keep], new[keep], indptr), shape=(X.shape[0], n_cols))
    out.sort_indices()
    return out


def _npz_info(path):
    """(shape, nnz, dtype) of a scipy.sparse .npz without loading its arrays."""
    with np.load(path) as z:
        shape = tuple(int(s) for s in z["shape"])
    with zipfile.ZipFile(path) as z, z.open("data.npy") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        (nnz,), _, dtype = read_header(fh)
    return shape, nnz, dtype


def write_stacked(path, parts, obs, var):
    """Write AnnData(X=vstack(parts), obs, var) to path with one part in memory at a time.

    parts: .npz files (CSR, saved with scipy.sparse.save_npz) in the row order of obs. The file is
    written as <path>.tmp and renamed when complete.
    """
    info = [_npz_info(p) for p in parts]
    n_rows, n_cols = sum(s[0] for s, _, _ in info), info[0][0][1]
    nnz = sum(n for _, n, _ in info)
    if n_rows != len(obs): raise ValueError(f"{n_rows} matrix rows but {len(obs)} obs rows")
    if any(s[1] != n_cols for s, _, _ in info): raise ValueError("parts differ in their number of columns")
    dtype = np.result_type(*[d for _, _, d in info])
    idx_dtype = np.int64 if max(nnz, n_cols) > np.iinfo(np.int32).max else np.int32
    print(f"  Writing {n_rows:,} x {n_cols:,}, {nnz:,} non-zeros ({dtype}), "
          f"{nnz * (dtype.itemsize + np.dtype(idx_dtype).itemsize) / 1e9:,.1f} GB")

    tmp = f"{path}.tmp"
    ad.AnnData(obs=obs, var=var).write(tmp)  # obs, var, uns; X is added below
    with h5py.File(tmp, "a") as f:
        g = f.create_group("X")
        g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": (n_rows, n_cols)})
        data = g.create_dataset("data", (nnz,), dtype=dtype)
        indices = g.create_dataset("indices", (nnz,), dtype=idx_dtype)
        indptr = g.create_dataset("indptr", (n_rows + 1,), dtype=idx_dtype)
        r = e = 0
        for p in parts:
            X = sparse.load_npz(p).tocsr()
            data[e:e + X.nnz] = X.data.astype(dtype, copy=False)
            indices[e:e + X.nnz] = X.indices
            indptr[r:r + X.shape[0]] = X.indptr[:-1].astype(np.int64) + e
            r, e = r + X.shape[0], e + X.nnz
            del X
        indptr[r] = e
    os.replace(tmp, path)


def write_with_obs(src, dst, obs):
    """Copy the h5ad src to dst with a new obs. HDF5 copies the matrices block by block, so they are
    never loaded. The file is written as <dst>.tmp and renamed when complete."""
    tmp = f"{dst}.tmp"
    with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
        for k, v in fs.attrs.items(): fd.attrs[k] = v
        for k in fs:
            if k != "obs": fs.copy(fs[k], fd, name=k)
        write_elem(fd, "obs", obs)
    os.replace(tmp, dst)
