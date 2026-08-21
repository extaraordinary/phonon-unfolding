"""
Build the ideal rocksalt primitive cell to unfold onto.

Two modes, controlled by MODE below:

MODE = "cubic_guess" (default, use this first)
    Backs out a single scalar lattice parameter from your relaxed SQS
    structure's volume, assuming a 1:1 cation:anion rocksalt stoichiometry
    (true for J14/J14+Cr) and a cubic cell shape:

        a_conventional = (4 * V_cell / n_formula_units) ** (1/3)

    (the factor of 4 converts primitive-cell volume per formula unit to the
    conventional cubic cell, which holds 4 formula units - matching what
    ase.build.bulk(..., crystalstructure='rocksalt', a=...) expects).

    This mode AUTOMATICALLY CHECKS whether your relaxed SQS cell is actually
    close enough to this cubic guess to be unfoldable at all (i.e. whether
    A_sc really is close to an integer multiple of the cubic guess). If your
    relaxation used PRESERVE_CELL_SHAPE = True in 00_relax_structure_chgnet.py
    (the default there), this will always pass, since the cell shape never
    had the chance to distort. If you relaxed with the cell shape fully free
    and it drifted away from cubic (angles != 90 deg / unequal axis lengths,
    which CAN happen for a low-symmetry SQS with no constraint holding it
    cubic), this check will fail with a clear error - switch to MODE =
    "known_transformation" below.

MODE = "known_transformation"
    For a relaxed SQS cell that is genuinely NOT close to cubic. Rather than
    guessing a shape, this recovers the primitive cell directly and exactly
    from your ACTUAL relaxed (possibly anisotropic) supercell lattice, using
    the integer supercell matrix M that was used when the SQS was originally
    built (e.g. from your mcsqs/ATAT generation records, or computed from the
    UNRELAXED SQS structure against an ideal cubic reference, since before
    relaxation an SQS is exactly commensurate with its ideal parent lattice
    by construction). Given M:

        A_primitive = inv(M) @ A_supercell_relaxed

    This is exact for ANY cell shape - no cubic assumption at all. The
    downstream unfolding code (unfold_core.py, unfold_setup.py) never assumed
    cubic either; it was really only this "guess a scalar a" shortcut that
    needed it. Set KNOWN_M below (as a 3x3 nested list/array) to use this.

    KNOWN_M may be defined relative to EITHER the 2-atom rhombohedral
    primitive cell (1 formula unit) OR the conventional 4-formula-unit
    cubic cell (8 atoms) - both are common conventions. This is detected
    automatically from the recovered cell's angles (~60 deg vs ~90 deg),
    so you don't need to know or track which one your SQS-generation
    workflow used.
"""
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import bulk

from unfold_setup import find_transformation_matrix

# ---- EDIT THESE ----
RELAXED_STRUCTURE = "J14_SQS_relaxed.vasp"
OUTPUT_PRIMITIVE = "rocksalt_primitive.vasp"
CATION_SPECIES = {"Mg", "Co", "Ni", "Cu", "Zn"}   # add "Cr" for the J14+Cr file
ANION_SPECIES = {"O"}
# use a real cation/anion symbol present in CATION_SPECIES/ANION_SPECIES here -
# it only serves as a label so 02_run_unfolding.py can tell cation and anion
# sites apart; it does not need to be the majority species.
PRIMITIVE_CATION_LABEL = "Mg"
PRIMITIVE_ANION_LABEL = "O"

MODE = "cubic_guess"   # or "known_transformation"
KNOWN_M = None          # e.g. [[4,0,0],[0,4,0],[0,0,4]] - only used if MODE == "known_transformation"
# ---------------------


def report_stoichiometry(sqs):
    symbols = sqs.get_chemical_symbols()
    n_cation = sum(1 for s in symbols if s in CATION_SPECIES)
    n_anion = sum(1 for s in symbols if s in ANION_SPECIES)
    n_other = len(symbols) - n_cation - n_anion
    print(f"{RELAXED_STRUCTURE}: {len(symbols)} atoms total, "
          f"{n_cation} cation-site, {n_anion} anion-site, {n_other} unclassified")
    if n_other:
        print(f"WARNING: {n_other} atoms matched neither CATION_SPECIES nor "
              f"ANION_SPECIES - check the species lists above.")
    if n_cation != n_anion:
        print(f"WARNING: cation/anion counts are not equal ({n_cation} vs "
              f"{n_anion}) - not a 1:1 rocksalt stoichiometry.")
    return n_cation, n_anion


def build_cubic_guess(sqs, n_formula_units):
    V_cell = sqs.get_volume()
    a_conventional = (4 * V_cell / n_formula_units) ** (1 / 3)
    print(f"Cell volume: {V_cell:.3f} A^3  ->  "
          f"effective conventional rocksalt a = {a_conventional:.4f} A")
    prim = bulk(f"{PRIMITIVE_CATION_LABEL}{PRIMITIVE_ANION_LABEL}",
                crystalstructure="rocksalt", a=a_conventional)

    # sanity check: is the relaxed cell actually close to this cubic guess?
    # (this is the same check 02_run_unfolding.py will do anyway - doing it
    # here first gives a clearer, earlier error message)
    try:
        find_transformation_matrix(prim.cell.array, sqs.cell.array, tol=1e-2)
    except ValueError:
        cellpar = sqs.cell.cellpar()
        print("\nERROR: the relaxed SQS cell is not close to this cubic "
              f"guess.\nActual cell parameters (a,b,c,alpha,beta,gamma): "
              f"{np.round(cellpar, 4)}\n"
              "This usually means the cell shape was allowed to relax "
              "freely (PRESERVE_CELL_SHAPE = False) and genuinely drifted "
              "away from cubic. Switch MODE to \"known_transformation\" "
              "above and supply KNOWN_M (the integer supercell matrix used "
              "when the SQS was built) instead of guessing a cubic shape.")
        raise
    return prim


def build_from_known_transformation(sqs, M):
    M = np.asarray(M)
    A_prim = np.linalg.inv(M) @ sqs.cell.array

    # KNOWN_M may have been defined relative to EITHER the 2-atom
    # rhombohedral primitive cell (1 formula unit, ~60 deg angles) OR the
    # conventional 4-formula-unit cubic cell (8 atoms, ~90 deg angles) -
    # both are common conventions depending on how your SQS was built.
    # Getting this wrong silently generates the wrong number of ideal
    # sites downstream in unfold_setup.assign_sublattices (missing 3 out
    # of 4 cation AND anion sites for the conventional case), which shows
    # up as a large fraction of atoms being reported "far from their
    # nearest ideal site" - detect which convention this actually is from
    # the recovered cell's angles rather than assuming one.
    angles = np.array(Atoms(cell=A_prim, pbc=True).cell.cellpar()[3:])
    if np.allclose(angles, 60, atol=5):
        print("Recovered cell angles are ~60 deg: treating A_prim as the "
              "2-atom rhombohedral primitive (1 formula unit).")
        symbols = [PRIMITIVE_CATION_LABEL, PRIMITIVE_ANION_LABEL]
        scaled_positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    elif np.allclose(angles, 90, atol=5):
        print("Recovered cell angles are ~90 deg: treating A_prim as the "
              "conventional cubic cell (4 formula units, 8-atom basis).")
        symbols = [PRIMITIVE_CATION_LABEL] * 4 + [PRIMITIVE_ANION_LABEL] * 4
        scaled_positions = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                             [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]]
    else:
        raise ValueError(
            f"Recovered primitive cell angles {angles} match neither the "
            f"rhombohedral (~60 deg) nor conventional cubic (~90 deg) "
            f"rocksalt convention - double check KNOWN_M.")

    prim = Atoms(symbols=symbols, cell=A_prim, scaled_positions=scaled_positions,
                 pbc=True)
    print(f"Recovered (possibly non-cubic) primitive cell from known M:\n{A_prim}")
    print(f"Cell parameters (a,b,c,alpha,beta,gamma): "
          f"{np.round(prim.cell.cellpar(), 4)}")
    # this should now match M exactly by construction
    find_transformation_matrix(prim.cell.array, sqs.cell.array, tol=1e-6)
    return prim


def main():
    sqs = read(RELAXED_STRUCTURE)
    n_cation, n_anion = report_stoichiometry(sqs)

    if MODE == "cubic_guess":
        prim = build_cubic_guess(sqs, n_formula_units=n_cation)
    elif MODE == "known_transformation":
        if KNOWN_M is None:
            raise ValueError("Set KNOWN_M to the integer supercell matrix "
                              "used to build the SQS when MODE == "
                              "'known_transformation'.")
        prim = build_from_known_transformation(sqs, KNOWN_M)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    write(OUTPUT_PRIMITIVE, prim)
    print(f"\nSaved {len(prim)}-atom ideal primitive cell to {OUTPUT_PRIMITIVE}")
    print("Use this as PRIM_STRUCTURE in 02_run_unfolding.py")


if __name__ == "__main__":
    main()
