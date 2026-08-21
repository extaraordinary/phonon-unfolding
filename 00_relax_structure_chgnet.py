"""
Step 1: relax an SQS (special quasirandom structure) supercell with CHGNet
before computing phonons.

Why this matters specifically for phonons: any residual forces or stress in
the starting structure show up as spurious/imaginary (negative-frequency)
low-frequency acoustic modes near Gamma, which corrupt both the raw phonon
DOS and the unfolded spectral function. It is worth relaxing tighter than
you would for a quick energy/structure screen.

This performs a FULL relaxation (atomic positions + cell shape + volume, by
default) via ASE's FrechetCellFilter. That is appropriate for an SQS cell,
which typically has no symmetry constraining the lattice angles/shape (unlike
a hand-built high-symmetry cell) - local distortions around differently
sized species need room to relax.

NOTE ON CELL SHAPE: with PRESERVE_CELL_SHAPE = True (default), only the
overall volume is allowed to relax, not the cell's shape/angles. This keeps
the relaxed structure an exact integer supercell of a well-defined primitive
cell, which the unfolding step downstream depends on. If the shape is instead
allowed to relax freely, building the ideal reference primitive cell needs
more care - see make_ideal_primitive.py's docstring for how to handle that
case.
"""
from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from chgnet.model.dynamics import CHGNetCalculator

# ---------------------
INPUT_STRUCTURE = "sqs_unrelaxed.vasp"
OUTPUT_STRUCTURE = "sqs_relaxed.vasp"
FMAX = 0.01          # eV/Angstrom - tight (see note below)
MAX_STEPS = 1000
RELAX_CELL = True    # relax the cell, not just atomic positions
PRESERVE_CELL_SHAPE = True
# If True, only the overall volume is allowed to relax (hydrostatic strain) -
# the cell's shape/angles stay exactly as built (e.g. cubic rocksalt shape).
# If False, the cell shape is also free to relax anisotropically, which a
# genuinely disordered SQS can drift into depending on how differently sized
# its species are - but that breaks the assumption used by
# make_ideal_primitive.py's "back out a scalar lattice parameter" mode. Keep
# this True unless you specifically want anisotropic relaxation and are
# prepared to use the "known transformation matrix" mode in
# make_ideal_primitive.py instead (see its docstring for non-cubic cells).
# ---------------------

def main():
    atoms = read(INPUT_STRUCTURE)
    print(f"Loaded {INPUT_STRUCTURE}: {len(atoms)} atoms, "
          f"species: {sorted(set(atoms.get_chemical_symbols()))}")
    print(f"Starting cell:\n{atoms.cell.array}")

    atoms.calc = CHGNetCalculator()

    # NOTE: chgnet's StructOptimizer.relax() convenience wrapper does not
    # expose hydrostatic_strain (its **kwargs go to the ASE optimizer, not
    # the cell filter), so the filter and optimizer are built directly here -
    # this is what that wrapper does internally, just with control over the
    # filter's own options.
    to_optimize = atoms
    if RELAX_CELL:
        to_optimize = FrechetCellFilter(atoms, hydrostatic_strain=PRESERVE_CELL_SHAPE)

    optimizer = FIRE(to_optimize)
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    converged = optimizer.converged()
    n_steps_taken = optimizer.nsteps
    print(f"\n{'Converged' if converged else 'DID NOT CONVERGE'} "
          f"in {n_steps_taken} steps (max force tolerance {FMAX} eV/A)")
    if not converged:
        print("Consider increasing MAX_STEPS, or loosening FMAX slightly if "
              "this has genuinely plateaued rather than still descending.")

    print(f"Final lattice:\n{atoms.cell.array}")
    print(f"Final cell parameters (a,b,c,alpha,beta,gamma): "
          f"{atoms.cell.cellpar()}")
    print(f"Final volume: {atoms.get_volume():.3f} A^3 "
          f"({atoms.get_volume() / len(atoms):.3f} A^3/atom)")

    write(OUTPUT_STRUCTURE, atoms)
    print(f"\nSaved relaxed structure to {OUTPUT_STRUCTURE}")
    print("Use this as INPUT_STRUCTURE in 01_compute_force_constants_chgnet.py")


if __name__ == "__main__":
    main()