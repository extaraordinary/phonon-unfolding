"""Core phonon band-unfolding implementation (Allen et al., PRB 87, 085322 (2013)
formalism, generalized to a multi-atom primitive basis).

Given supercell phonon eigenvectors/frequencies (from a phonopy Phonopy object
whose `unitcell` IS the supercell/SQS structure, i.e. primitive_matrix = identity),
this computes the spectral weight of each supercell mode onto a target primitive-
cell wavevector q, resolved by primitive sublattice (e.g. cation site / anion site).
"""
import numpy as np


def q_frac_primitive_to_supercell(q_frac_prim, M):
    """Convert primitive-cell fractional q-point to supercell fractional q-point.

    A_S = M @ A_P  (supercell lattice = integer transform of primitive lattice)
    => q_frac_S = q_frac_P @ M.T
    """
    return np.asarray(q_frac_prim) @ np.asarray(M).T


def spectral_weights(phonon, q_frac_prim, M, sublattice_of_atom, n_sublattice_sites=None):
    """Compute spectral weight P(q, nu) for every supercell band at primitive q.

    Parameters
    ----------
    phonon : phonopy.Phonopy
        Phonopy object whose unitcell is the supercell (SQS) structure, with
        force constants already produced.
    q_frac_prim : array-like, shape (3,)
        Target q-point in fractional coordinates of the PRIMITIVE reciprocal lattice.
    M : array-like, shape (3,3)
        Integer transformation A_supercell = M @ A_primitive.
    sublattice_of_atom : array-like, shape (n_atoms_supercell,)
        Index j (0..n_sublattices-1) identifying which primitive-cell site each
        supercell atom nominally occupies (e.g. 0=cation site, 1=anion site),
        REGARDLESS of which chemical species sits there. This is what lets the
        method handle chemical disorder.
    n_sublattice_sites : int, optional
        Number of primitive-cell sites (defaults to max(sublattice_of_atom)+1).

    Returns
    -------
    freqs : ndarray, shape (n_bands,)   frequencies in THz
    weights : ndarray, shape (n_bands,) spectral weight P(q, nu), 0 <= P <= n_sites

    Notes
    -----
    IMPORTANT: q_frac_prim is deliberately passed to phonopy WITHOUT reducing it
    modulo 1 (i.e. we do NOT fold it into the supercell's own first Brillouin
    zone first). Phonopy's dynamical matrix uses the full atomic-position phase
    convention, so evaluating it directly at the *unreduced* q_frac_sc already
    returns eigenvectors with the correct primitive-q Bloch phase baked in. No
    additional phase factor should be applied in the projection step below -
    doing so would double-count the phase and silently collapse every q back
    onto the zone-center pattern (this was verified empirically).
    """
    q_frac_prim = np.asarray(q_frac_prim, dtype=float)
    M = np.asarray(M, dtype=float)

    q_frac_sc = q_frac_primitive_to_supercell(q_frac_prim, M)  # NOT reduced mod 1

    sc = phonon.supercell  # == unitcell here, since supercell_matrix = identity
    freqs, eigvecs = phonon.get_frequencies_with_eigenvectors(q_frac_sc)
    # eigvecs shape: (3*n_atoms, n_bands); reshape to (n_atoms, 3, n_bands)
    n_atoms = len(sc.symbols)
    eigvecs = eigvecs.reshape(n_atoms, 3, -1)

    sublattice_of_atom = np.asarray(sublattice_of_atom)
    if n_sublattice_sites is None:
        n_sublattice_sites = int(sublattice_of_atom.max()) + 1

    n_bands = eigvecs.shape[-1]
    weights = np.zeros(n_bands)

    for j in range(n_sublattice_sites):
        mask = sublattice_of_atom == j
        n_p = mask.sum()  # number of primitive-cell copies contributing to this sublattice
        # c_j has shape (3, n_bands) - coherent sum over the images of site j
        c_j = (1.0 / np.sqrt(n_p)) * np.sum(eigvecs[mask], axis=0)
        weights += np.sum(np.abs(c_j) ** 2, axis=0)

    return freqs, weights
