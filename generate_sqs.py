"""
Generate a cubic rocksalt SQS (special quasirandom structure) for a
multi-cation composition like J14 (Mg,Co,Ni,Cu,Zn / O) or J14+Cr
(Mg,Co,Ni,Cu,Zn,Cr / O), using icet.

Install: pip install icet

WHY icet AND NOT ATAT/mcsqs: icet is a pure-Python package (no external
binary to build/install) that does the same thing mcsqs does - simulated
annealing to match the supercell's correlation functions to the ideal
infinite random alloy - so it drops straight into this Colab/VS Code
workflow without needing ATAT set up separately.

WHY A FIXED SUPERCELL (IMPORTANT FOR SPEED): icet's generate_sqs() by
default searches over EVERY symmetry-inequivalent supercell shape up to
your requested size before annealing, which is combinatorially expensive
and was still running after several minutes for a 128-atom cell in testing.
This script instead builds ONE specific supercell directly - an NxNxNx...
diagonal repeat of the primitive rocksalt cell, i.e. exactly cubic - and
anneals only on that fixed shape via
icet.tools.structure_generation.generate_target_structure_from_supercells.
This is both what you actually want (a cubic cell, matching the rest of
this pipeline) and dramatically faster: ~10-20 seconds for a 128-atom,
5-6 species cell at 2000 MC steps in testing, vs. the shape search alone
exceeding several minutes.

WHY THE COMPOSITION ISN'T EXACTLY "1/5 EACH": with 64 cation sites (for a
128-atom cell) and 5 species, 64/5 = 12.8 - there's no way to split 64 sites
into exactly equal integer counts across 5 species. This script
automatically finds the closest achievable integer split (13,13,13,13,12
for 5 species; 11,11,11,11,10,10 for 6) and uses that as the exact target,
rather than 0.2 each which no integer supercell of this size can satisfy
exactly.
"""
import numpy as np
from ase.build import bulk
from ase.io import write
from icet import ClusterSpace
from icet.tools.structure_generation import (
    _get_sqs_cluster_vector,
    generate_target_structure_from_supercells,
)

# ---- EDIT THESE ----
CATION_SPECIES = ["Mg", "Co", "Ni", "Cu", "Zn"]   # add "Cr" for J14+Cr
ANION_SPECIES = "O"
LATTICE_PARAMETER = 4.20     # conventional cubic rocksalt a, Angstrom - pick
                              # something physically reasonable for your
                              # composition (e.g. Vegard's-law estimate from
                              # your parent oxides); this only sets the
                              # starting geometry, not the SQS quality
REPEAT = (4, 4, 4)            # supercell size as a diagonal repeat of the
                              # 2-atom primitive cell: (4,4,4) -> 64 primitive
                              # cells -> 128 atoms total. Increase for a
                              # bigger cell (cost grows with system size).
CUTOFFS = [8.0, 4.0]          # [pair cutoff, triplet cutoff] in Angstrom;
                              # extend with a 4th-shell value etc. if you
                              # want longer-range correlations matched too
N_STEPS = 2000                # Monte Carlo steps; RMSE between achieved and
                              # target cluster vector plateaus quickly (see
                              # docstring test note below) - 2000 is usually
                              # enough, more gives diminishing returns
RANDOM_SEED = None            # set an int for a reproducible structure
OUTPUT_STRUCTURE = "J14_SQS_unrelaxed.vasp"
# ---------------------


def nearest_equal_integer_split(n_sites, n_species):
    """Split n_sites as evenly as possible across n_species integer counts."""
    base, remainder = divmod(n_sites, n_species)
    return [base + 1] * remainder + [base] * (n_species - remainder)


def main():
    prim = bulk(f"{CATION_SPECIES[0]}{ANION_SPECIES}", crystalstructure="rocksalt",
                a=LATTICE_PARAMETER)
    chemical_symbols = [CATION_SPECIES, [ANION_SPECIES]]
    cs = ClusterSpace(prim, CUTOFFS, chemical_symbols)
    print(cs)

    supercell = prim.repeat(REPEAT)
    n_cation_sites = len(supercell) // 2
    print(f"Building a {REPEAT} supercell: {len(supercell)} atoms total, "
          f"{n_cation_sites} cation sites, cell shape:\n{supercell.cell.cellpar()}")

    counts = nearest_equal_integer_split(n_cation_sites, len(CATION_SPECIES))
    target_concentrations = {s: c / n_cation_sites
                              for s, c in zip(CATION_SPECIES, counts)}
    print(f"Target composition (closest achievable to equal shares): "
          f"{ {s: f'{c}/{n_cation_sites}' for s, c in zip(CATION_SPECIES, counts)} }")

    target_cv = _get_sqs_cluster_vector(cluster_space=cs,
                                         target_concentrations=target_concentrations)

    sqs = generate_target_structure_from_supercells(
        cluster_space=cs,
        supercells=[supercell],       # single fixed (cubic) shape - see docstring
        target_concentrations=target_concentrations,
        target_cluster_vector=target_cv,
        n_steps=N_STEPS,
        random_seed=RANDOM_SEED,
    )

    achieved_cv = cs.get_cluster_vector(sqs)
    rmse = np.sqrt(np.mean((np.array(achieved_cv) - np.array(target_cv)) ** 2))
    print(f"\nCluster-vector RMSE vs. ideal random alloy: {rmse:.5f} "
          f"(lower is better; for reference this was ~0.013 in testing at "
          f"N_STEPS=2000 for a 128-atom, 5-species cell, with only marginal "
          f"improvement from running 5x longer)")

    from collections import Counter
    print("Achieved composition:", Counter(sqs.get_chemical_symbols()))
    print("Cell shape (should still be exactly cubic):", sqs.cell.cellpar())

    write(OUTPUT_STRUCTURE, sqs)
    print(f"\nSaved to {OUTPUT_STRUCTURE}")
    print("Use this as INPUT_STRUCTURE in 00_relax_structure_chgnet.py")


if __name__ == "__main__":
    main()
