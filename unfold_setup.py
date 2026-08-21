"""Setup helpers for unfolding a real SQS structure onto a reference 
primitive rocksalt cell.

These handle the two "bookkeeping" steps that are specific to your real
structures (as opposed to the toy validation case): finding the integer
transformation matrix M between your reference primitive cell and the SQS
supercell, and figuring out which SQS atom sits on which ideal primitive site
(cation sublattice vs anion sublattice), independent of chemical species.
"""
import numpy as np


def find_transformation_matrix(A_prim, A_sc, tol=1e-3):
    """Find integer M such that A_sc = M @ A_prim.

    Parameters
    ----------
    A_prim : (3,3) array, rows = primitive lattice vectors (Angstrom)
    A_sc   : (3,3) array, rows = SQS supercell lattice vectors (Angstrom)

    Returns
    -------
    M : (3,3) ndarray of ints
    """
    A_prim = np.asarray(A_prim, dtype=float)
    A_sc = np.asarray(A_sc, dtype=float)
    M_float = A_sc @ np.linalg.inv(A_prim)
    M = np.round(M_float).astype(int)
    max_err = np.max(np.abs(M_float - M))
    if max_err > tol:
        raise ValueError(
            f"A_sc is not an integer multiple of A_prim (max deviation from "
            f"integer = {max_err:.4f}, tol = {tol}). Check that A_prim is "
            f"really the primitive cell your SQS was built from, and that "
            f"both cells are given in the same lattice-vector convention "
            f"(e.g. both relaxed, both in Angstrom)."
        )
    print(f"Transformation matrix M (A_sc = M @ A_prim), integer to within "
          f"{max_err:.2e}:\n{M}")
    print(f"det(M) = {np.linalg.det(M):.4f}  "
          f"(should equal N_sc_atoms / N_prim_atoms)")
    return M


def assign_sublattices(sc_scaled_positions, sc_cell, prim_positions_cart,
                        prim_sublattice_labels, M, tol=0.6):
    """Assign each SQS supercell atom to a primitive sublattice (site index),
    independent of which chemical species occupies it.

    This works by generating all N_p ideal images of each primitive-cell site
    within the supercell (via the integer translations implied by M) and
    matching each real SQS atom to its nearest ideal image. This is robust to
    the relaxation displacements of an SQS structure (atoms move somewhat off
    their ideal lattice sites but should stay closer to "their" site than to
    any other site).

    Parameters
    ----------
    sc_scaled_positions : (N_sc, 3) fractional coords of the SQS structure
    sc_cell : (3,3) SQS lattice vectors (Angstrom)
    prim_positions_cart : (n_sites, 3) Cartesian positions of the primitive
        cell's sites (e.g. [[0,0,0], [a/2,0,0]] for rocksalt cation, anion)
    prim_sublattice_labels : (n_sites,) int label for each primitive site
        (e.g. [0, 1] meaning site 0 = cation, site 1 = anion)
    M : (3,3) int transformation matrix, A_sc = M @ A_prim

    Returns
    -------
    sublattice_of_atom : (N_sc,) ndarray of ints, one label per SQS atom
    """
    sc_cell = np.asarray(sc_cell, dtype=float)
    sc_positions_cart = np.asarray(sc_scaled_positions) @ sc_cell
    prim_positions_cart = np.asarray(prim_positions_cart, dtype=float)

    # Build every ideal image of every primitive site that lies inside the
    # supercell, by tiling over a generous integer range and keeping those
    # that land inside the supercell (in fractional coords, [0,1)).
    inv_M = np.linalg.inv(M)
    max_n = int(np.max(np.abs(M))) + 2
    n_range = range(-max_n, max_n + 1)

    ideal_images_cart = []
    ideal_images_label = []
    # A_sc = M @ A_prim  =>  A_prim = inv(M) @ A_sc  (matrix multiplication
    # is not commutative - this must be left-multiplication by inv(M).
    # Using A_sc @ inv_M here instead is a real bug that stays silently
    # invisible whenever M is a scalar multiple of the identity, since
    # scalar matrices commute with anything - it only surfaces for a
    # genuinely non-uniform M, e.g. [[4,0,0],[0,4,0],[0,0,2]]. Caught via
    # an end-to-end validation test against an independent ground-truth
    # calculation before this fix was made.)
    A_prim = inv_M @ sc_cell
    for i in n_range:
        for j in n_range:
            for k in n_range:
                R = np.array([i, j, k]) @ A_prim  # primitive lattice translation, Cartesian
                for site_pos, label in zip(prim_positions_cart, prim_sublattice_labels):
                    cart = R + site_pos
                    frac_sc = cart @ np.linalg.inv(sc_cell)
                    if np.all(frac_sc > -1e-4) and np.all(frac_sc < 1 - 1e-4):
                        ideal_images_cart.append(cart)
                        ideal_images_label.append(label)

    ideal_images_cart = np.array(ideal_images_cart)
    ideal_images_label = np.array(ideal_images_label)

    n_expected = len(prim_sublattice_labels) * int(round(np.linalg.det(M)))
    if len(ideal_images_cart) != n_expected:
        print(f"WARNING: found {len(ideal_images_cart)} ideal sites, "
              f"expected {n_expected}. Check tolerances / cell consistency.")

    sublattice_of_atom = np.zeros(len(sc_positions_cart), dtype=int)
    for a, pos in enumerate(sc_positions_cart):
        # nearest ideal image, accounting for periodic boundary conditions
        diffs = ideal_images_cart - pos
        frac_diffs = diffs @ np.linalg.inv(sc_cell)
        frac_diffs -= np.round(frac_diffs)  # minimum image
        diffs_pbc = frac_diffs @ sc_cell
        dists = np.linalg.norm(diffs_pbc, axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] > tol:
            print(f"WARNING: atom {a} is {dists[nearest]:.2f} A from its "
                  f"nearest ideal site - relaxation may be large, or the "
                  f"reference primitive cell/lattice parameter may not "
                  f"match this structure well.")
        sublattice_of_atom[a] = ideal_images_label[nearest]

    return sublattice_of_atom