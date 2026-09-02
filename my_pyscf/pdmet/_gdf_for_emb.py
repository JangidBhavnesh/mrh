"""Stream and compress Gamma-point DF factors for periodic embedding."""

import os

import h5py
import numpy as np

from pyscf import ao2mo
from pyscf.pbc import gto, tools
from pyscf.pbc.df import incore

MAX_NAUX = 256  # maximum number of auxiliary functions to process at once

def _open_h5(h5file, mode="a"):
    """Return an HDF5 handle and whether this call must close it."""
    if isinstance(h5file, (str, bytes, os.PathLike)):
        return h5py.File(h5file, mode), True
    if isinstance(h5file, (h5py.File, h5py.Group)):
        return h5file, False
    raise TypeError("h5file must be an HDF5 handle or filename")


def _on_the_fly_aux_shell_slices(auxcell, max_naux):
    """Yield auxiliary shell ranges bounded by ``max_naux`` functions."""
    if not isinstance(max_naux, (int, np.integer)) or max_naux < 1:
        raise ValueError("max_naux must be a positive integer")
    ao_loc = auxcell.ao_loc_nr()
    shell_start = 0
    while shell_start < auxcell.nbas:
        shell_stop = shell_start + 1
        while shell_stop < auxcell.nbas:
            if ao_loc[shell_stop + 1] - ao_loc[shell_start] > max_naux:
                break
            shell_stop += 1
        yield shell_start, shell_stop
        shell_start = shell_stop


def _on_the_fly_ao2lo_transform(ao_3c, ao2lo):
    """Transform one real AO 3C block with PySCF's AO2MO kernel."""
    if ao_3c.dtype != np.double or ao2lo.dtype != np.double:
        raise ValueError("only real Gamma-point transformations are supported")
    nlo = ao2lo.shape[1]
    _, _, moij, ijslice = ao2mo.incore._conc_mos(ao2lo, ao2lo, compact=False)
    lo_3c = ao2mo._ao2mo.nr_e2(
        np.asarray(ao_3c, order="C"), moij, ijslice,
        aosym="s1", mosym="s1",
    )
    return lo_3c.reshape(-1, nlo, nlo)


def _on_the_fly_gdf_in_ao2lo(
        scell, ao2lo, h5file, auxbasis=None, max_naux=64,
        dataset="j3c/0/0", overwrite=False):
    """Stream bare AO-to-LO 3C blocks into an intermediate HDF5 dataset."""
    ao2lo = np.asarray(ao2lo)
    nao = scell.nao_nr()
    if ao2lo.ndim != 2 or ao2lo.shape[0] != nao or ao2lo.shape[1] == 0:
        raise ValueError(f"ao2lo must have shape ({nao}, nlo); got {ao2lo.shape}")
    if ao2lo.dtype != np.double:
        raise ValueError("ao2lo must be real for this Gamma-point implementation")

    output = h5file
    h5file, close_file = _open_h5(h5file)
    try:
        if dataset in h5file:
            if not overwrite:
                raise ValueError(f"dataset {dataset!r} already exists")
            del h5file[dataset]

        auxcell = incore.make_auxcell(scell, auxbasis)
        naux = auxcell.nao_nr()
        nlo = ao2lo.shape[1]
        bare_3c = h5file.create_dataset(
            dataset, shape=(naux, nlo, nlo), dtype=np.double,
            chunks=(min(max_naux, naux), nlo, nlo),
        )
        bare_3c.attrs["representation"] = "bare_3center_ao2lo"

        int3c = incore.wrap_int3c(scell, auxcell, aosym="s1")
        ao_loc = auxcell.ao_loc_nr()
        row_start = 0
        for shell_start, shell_stop in _on_the_fly_aux_shell_slices(
                auxcell, max_naux):
            row_stop = row_start + ao_loc[shell_stop] - ao_loc[shell_start]
            raw_3c = int3c((
                0, scell.nbas, 0, scell.nbas, shell_start, shell_stop,
            ))
            bare_3c[row_start:row_stop] = _on_the_fly_ao2lo_transform(
                np.asarray(raw_3c.T, order="C"), ao2lo,
            )
            row_start = row_stop
        h5file.flush()
    finally:
        if close_file:
            h5file.close()
    return os.fspath(output) if close_file else output


def _on_the_fly_aux_coulomb_metric(
        scell, h5file, auxbasis=None, metric_group="aux_metric",
        linear_dep_tol=1e-10, overwrite=False):
    """Factor the auxiliary Coulomb metric and save its retained modes."""
    if not np.isscalar(linear_dep_tol) or linear_dep_tol < 0:
        raise ValueError("linear_dep_tol must be a non-negative scalar")

    output = h5file
    h5file, close_file = _open_h5(h5file)
    try:
        if metric_group in h5file:
            if not overwrite:
                raise ValueError(f"group {metric_group!r} already exists")
            del h5file[metric_group]

        auxcell = incore.make_auxcell(scell, auxbasis)
        coulomb = auxcell.pbc_intor("int2c2e", hermi=1)
        coulomb = (coulomb + coulomb.T) * 0.5
        values, vectors = np.linalg.eigh(coulomb)
        keep = values > linear_dep_tol * values[-1]
        if not np.any(keep):
            raise ValueError("auxiliary metric has no retained modes")

        group = h5file.create_group(metric_group)
        group.attrs["rank"] = int(np.count_nonzero(keep))
        group.create_dataset("coulomb", data=coulomb)
        group.create_dataset("eigenvalues", data=values[keep])
        group.create_dataset("eigenvectors", data=vectors[:, keep])
        h5file.flush()
    finally:
        if close_file:
            h5file.close()
    return os.fspath(output) if close_file else output


def _compression_coefficients(
        h5file, input_dataset, metric_group, max_naux,
        compression_tol, max_rank):
    """Return retained generalized-metric compression coefficients."""
    bare_3c = h5file[input_dataset]
    if bare_3c.ndim != 3 or bare_3c.shape[1] != bare_3c.shape[2]:
        raise ValueError("bare 3C dataset must have shape (naux, nlo, nlo)")
    naux = bare_3c.shape[0]
    metric = h5file[metric_group]
    metric_values = metric["eigenvalues"][()]
    metric_vectors = metric["eigenvectors"][()]
    if metric_vectors.shape != (naux, metric_values.size):
        raise ValueError("bare 3C and auxiliary metric dimensions differ")

    pair_metric = np.zeros((naux, naux), dtype=np.double)
    for p0 in range(0, naux, max_naux):
        p1 = min(p0 + max_naux, naux)
        b_p = bare_3c[p0:p1].reshape(p1 - p0, -1)
        for q0 in range(0, p1, max_naux):
            q1 = min(q0 + max_naux, naux)
            b_q = bare_3c[q0:q1].reshape(q1 - q0, -1)
            block = b_p @ b_q.T
            pair_metric[p0:p1, q0:q1] = block
            if p0 != q0:
                pair_metric[q0:q1, p0:p1] = block.T

    inverse_metric_sqrt = 1.0 / np.sqrt(metric_values)
    compressed_metric = metric_vectors.T @ pair_metric @ metric_vectors
    compressed_metric *= inverse_metric_sqrt[:, None]
    compressed_metric *= inverse_metric_sqrt[None, :]
    values, vectors = np.linalg.eigh(compressed_metric)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = values > compression_tol * values[0]
    if max_rank is not None:
        keep[np.flatnonzero(keep)[max_rank:]] = False
    if not np.any(keep):
        raise ValueError("compression retained no DF factors")
    coefficients = (
        metric_vectors * inverse_metric_sqrt
    ) @ vectors[:, keep]
    return coefficients, values[keep]


def _on_the_fly_metric_compress_3c(
        intermediate_file, cderi_file, input_dataset="j3c/0/0",
        metric_group="aux_metric", max_naux=MAX_NAUX, compression_tol=1e-10,
        max_rank=None, overwrite=False, delete_intermediate=False):
    """Write compressed cderi to a separate file from staged bare 3C data."""
    if not isinstance(max_naux, (int, np.integer)) or max_naux < 1:
        raise ValueError("max_naux must be a positive integer")
    if not np.isscalar(compression_tol) or compression_tol < 0:
        raise ValueError("compression_tol must be a non-negative scalar")
    if max_rank is not None and (
            not isinstance(max_rank, (int, np.integer)) or max_rank < 1):
        raise ValueError("max_rank must be a positive integer or None")
    if not isinstance(intermediate_file, (str, bytes, os.PathLike)):
        raise TypeError("intermediate_file must be a filename")
    if not isinstance(cderi_file, (str, bytes, os.PathLike)):
        raise TypeError("cderi_file must be a filename")
    if os.path.abspath(intermediate_file) == os.path.abspath(cderi_file):
        raise ValueError("intermediate and compressed cderi files must differ")

    output_mode = "w" if overwrite else "x"
    with h5py.File(intermediate_file, "r") as intermediate:
        coefficients, eigenvalues = _compression_coefficients(
            intermediate, input_dataset, metric_group, max_naux,
            compression_tol, max_rank,
        )
        bare_3c = intermediate[input_dataset]
        naux, nlo, _ = bare_3c.shape
        ncompressed = coefficients.shape[1]

        with h5py.File(cderi_file, output_mode) as output:
            # PySCF v2 layout: j3c/<k-point-pair>/<segment>.
            factors = output.create_dataset(
                "j3c/0/0", shape=(ncompressed, nlo * nlo),
                chunks=(min(max_naux, ncompressed), nlo * nlo), dtype=np.double,
            )
            for a0 in range(0, ncompressed, max_naux):
                a1 = min(a0 + max_naux, ncompressed)
                factor_block = np.zeros((a1 - a0, nlo * nlo))
                for p0 in range(0, naux, max_naux):
                    p1 = min(p0 + max_naux, naux)
                    b_p = bare_3c[p0:p1].reshape(p1 - p0, -1)
                    factor_block += coefficients[p0:p1, a0:a1].T @ b_p
                factors[a0:a1] = factor_block
            output.create_dataset("kpts", data=np.zeros((1, 3)))
            output.create_dataset("aosym", data=np.bytes_("s1"))
            output.attrs["nao"] = nlo
            output.attrs["naux_original"] = naux
            output.attrs["naux_compressed"] = ncompressed
            output.attrs["compression_tol"] = compression_tol
            output.attrs["largest_eigenvalue"] = eigenvalues[0]
            output.flush()

    if delete_intermediate:
        os.remove(intermediate_file)
    return os.fspath(cderi_file)


def _show_h5_contents(h5file):
    """Print HDF5 dataset names, shapes, and dtypes."""
    with h5py.File(h5file, "r") as handle:
        print(f"HDF5 contents: {h5file}")
        handle.visititems(
            lambda name, item: print(
                f"  {name}: shape={item.shape}, dtype={item.dtype}"
            ) if isinstance(item, h5py.Dataset) else None
        )


if __name__ == "__main__":
    cell = gto.Cell()
    cell.a = np.eye(3) * 3.5668
    cell.atom = """
        C 0.0    0.0    0.0
        C 0.8917 0.8917 0.8917
        C 1.7834 1.7834 0.0
        C 2.6751 2.6751 0.8917
        C 1.7834 0.0    1.7834
        C 2.6751 0.8917 2.6751
        C 0.0    1.7834 1.7834
        C 0.8917 2.6751 2.6751
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()

    supercell = tools.super_cell(cell, (2, 2, 2))
    from pyscf import lo
    ao2lo = lo.orth_ao(supercell, method="meta_lowdin")[:, :50]

    intermediate_file = "diamond_bare_df.h5"
    compressed_file = "diamond_compressed_df.h5"
    _on_the_fly_gdf_in_ao2lo(
        supercell, ao2lo, intermediate_file, overwrite=True,
    )
    _on_the_fly_aux_coulomb_metric(
        supercell, intermediate_file, overwrite=True,
    )
    _show_h5_contents(intermediate_file)

    _on_the_fly_metric_compress_3c(
        intermediate_file, compressed_file, overwrite=True,
        delete_intermediate=True,
    )
    _show_h5_contents(compressed_file)
    os.remove(compressed_file)
    # embedding_mf.with_df._cderi = compressed_file
