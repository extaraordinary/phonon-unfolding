"""
Step 3: compute force constants for a relaxed SQS structure using CHGNet as
the force calculator, via phonopy's finite-displacement machinery.

KEY POINT FOR UNFOLDING: the SQS cell itself is treated as the phonopy
"unitcell", with supercell_matrix=[1,1,1] and primitive_matrix=identity - the
full force-constant matrix is computed for exactly the atoms already present
in the cell, with no further symmetry reduction (there isn't any, since the
structure is disordered) and no further periodic replication. This differs
from a normal phonopy run on an ordered crystal, where you'd hand phonopy a
small primitive cell and let it build a bigger supercell for you - here the
SQS cell is already the "supercell" from the unfolding point of view, so it
goes in directly.

This mirrors what a tool like atomate2's PhononMaker does under the hood
(phonopy + a force calculator); this script just produces a plain
phonopy_params.yaml file with the force constants and structure, in a form
that's easy to hand off to the unfolding step. If you're already using
atomate2/PhononMaker, you can get an equivalent file from its output
document's force_constants field (a pymatgen ForceConstants object, which
phonopy can also read/reconstruct) - 02_run_unfolding.py only needs a
phonopy.Phonopy object with force_constants set and a matching structure,
however you got there.

This is the CPU/GPU-heavy step in the pipeline (cost scales with number of
atoms x number of displacements); everything else here is fast.
"""
import numpy as np
from ase.io import read
from ase import Atoms
from chgnet.model.dynamics import CHGNetCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# ---------------------
INPUT_STRUCTURE = "sqs_relaxed.vasp"
OUTPUT_YAML = "phonopy_params.yaml"
DISPLACEMENT_DISTANCE = 0.01               # Angstrom
# ---------------------

def main():
    sqs = read(INPUT_STRUCTURE)
    print(f"Loaded {INPUT_STRUCTURE}: {len(sqs)} atoms, "
          f"species: {sorted(set(sqs.get_chemical_symbols()))}")

    punitcell = PhonopyAtoms(symbols=sqs.get_chemical_symbols(),
                              cell=sqs.cell.array,
                              scaled_positions=sqs.get_scaled_positions())

    # supercell_matrix = identity: the SQS cell IS the box the FCs are computed for.
    # primitive_matrix = identity: no symmetry reduction (disordered structure).
    phonon = Phonopy(punitcell, supercell_matrix=[1, 1, 1],
                      primitive_matrix=np.eye(3))
    phonon.generate_displacements(distance=DISPLACEMENT_DISTANCE)
    disp_supercells = phonon.supercells_with_displacements
    print(f"Generated {len(disp_supercells)} displaced structures "
          f"(phonopy uses the SQS cell's own point-group symmetry, if any, "
          f"to minimize this - for a true SQS this is usually close to "
          f"3 * n_atoms unless there's accidental local symmetry)")

    calc = CHGNetCalculator()
    forces = []
    for i, ph_atoms in enumerate(disp_supercells):
        at = Atoms(symbols=ph_atoms.symbols, cell=ph_atoms.cell,
                   scaled_positions=ph_atoms.scaled_positions, pbc=True)
        at.calc = calc
        f = at.get_forces()
        forces.append(f)
        if (i + 1) % 10 == 0 or (i + 1) == len(disp_supercells):
            print(f"  computed forces for displacement {i + 1}/{len(disp_supercells)}")

    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()  # enforces the acoustic sum rule - important
    phonon.save(OUTPUT_YAML, settings={"force_constants": True})
    print(f"\nSaved {OUTPUT_YAML} - use this directly in 02_run_unfolding.py")


if __name__ == "__main__":
    main()