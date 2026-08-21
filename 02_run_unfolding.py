"""
Step 4: unfold an SQS structure's phonon force constants onto the primitive
rocksalt Brillouin zone, and plot the resulting spectral function A(q,omega) 
as a heatmap.

Inputs:
  - PRIM_STRUCTURE: an ideal rocksalt primitive cell (2 atoms if built via
    "cubic_guess" mode, or 8 atoms/conventional cell if built via
    "known_transformation" mode - make_ideal_primitive.py handles both, and
    this script reads whichever it produced directly from the file, no
    changes needed here either way) to unfold onto. Use a lattice parameter
    representative of the actual SQS cell, not necessarily a literature value
    for the parent compound - this keeps q-points defined consistently with
    the supercell being unfolded.
  - SQS_PHONOPY_YAML: output of 01_compute_force_constants_chgnet.py (or an
    equivalent phonopy_params.yaml with force_constants written out from an
    atomate2/PhononMaker run).
  - CATION_SPECIES / ANION_SPECIES: which chemical symbols in the SQS belong
    to the cation sublattice vs. the anion sublattice. Unfolding groups all
    cation-site atoms together as "site 0" regardless of which species sits
    there - the chemical/mass disorder across those species is exactly what
    produces the spectral broadening this analysis is meant to reveal.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from ase.io import read
from ase.cell import Cell
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml

from unfold_core import spectral_weights
from unfold_setup import find_transformation_matrix, assign_sublattices

# ---------------------
PRIM_STRUCTURE = "rocksalt_primitive.vasp"   # 2-atom ideal rocksalt cell (cation, anion)
SQS_PHONOPY_YAML = "phonopy_params.yaml"
CATION_SPECIES = {"Mg", "Co", "Ni", "Cu", "Zn"}
ANION_SPECIES = {"O"}
Q_PATH = "GXWKGL"          # standard FCC/rocksalt path; set to None to
                            # auto-detect the correct path for a non-cubic
                            # primitive cell (see README, non-cubic section).
                            # Falls back to auto-detection automatically if
                            # this path isn't valid for the cell you end up
                            # with (e.g. from "known_transformation" mode).
N_POINTS_PER_SEGMENT = 40
SIGMA_THZ = 0.15           # Gaussian broadening width for the spectral function
FREQ_MAX_THZ = 20.0
OUTPUT_PREFIX = "unfolded"
SPECTRAL_COLOR = "#08306b"  # dark navy; try "#67000d" (dark red) or
                             # "#00441b" (dark green) for a different accent
SPECTRAL_GAMMA = 0.5        # PowerNorm(gamma<1) stretches the low end of the
                             # color scale, making the faint, diffuse spectral
                             # weight - the signal of chemical/mass disorder
                             # broadening - visible instead of washed out next
                             # to the much brighter concentrated peaks near
                             # the ideal bands. gamma=1 recovers a linear scale.
# ---------------------


def load_phonon(yaml_path):
    py = PhonopyYaml()
    py.read(yaml_path)
    phonon = Phonopy(py.unitcell, supercell_matrix=[1, 1, 1],
                      primitive_matrix=np.eye(3))
    phonon.force_constants = py.force_constants
    phonon.symmetrize_force_constants()
    return phonon


def main():
    prim = read(PRIM_STRUCTURE)
    A_prim = prim.cell.array
    prim_positions_cart = prim.get_positions()
    # site label per primitive atom: 0 = cation, 1 = anion (order-independent)
    prim_labels = [0 if s in CATION_SPECIES else 1 for s in prim.get_chemical_symbols()]
    print("Primitive cell species/labels:", list(zip(prim.get_chemical_symbols(), prim_labels)))

    phonon = load_phonon(SQS_PHONOPY_YAML)
    sc = phonon.supercell
    A_sc = sc.cell

    M = find_transformation_matrix(A_prim, A_sc)

    sub = assign_sublattices(sc.scaled_positions, A_sc, prim_positions_cart,
                              prim_labels, M)
    n_cation = np.sum(sub == 0)
    n_anion = np.sum(sub == 1)
    print(f"Assigned {n_cation} atoms to cation sublattice, "
          f"{n_anion} to anion sublattice "
          f"(expect a ratio matching the primitive cell, e.g. 1:1 for rocksalt)")

    # --- build the primitive-cell q-path using ASE (uses the primitive cell
    # supplied above, so labeled points are correct for this specific lattice) ---
    cell = Cell(A_prim)
    n_segments = (len(Q_PATH) - 1) if Q_PATH is not None else 5  # rough default
    try:
        bandpath = cell.bandpath(Q_PATH, npoints=N_POINTS_PER_SEGMENT * n_segments)
    except KeyError as e:
        # Q_PATH's labels (e.g. G/X/W/K/L) don't exist for the Bravais lattice
        # ASE actually detected for this cell - common with
        # "known_transformation" mode in make_ideal_primitive.py, where
        # A_prim comes straight from a real relaxed structure: even a tiny
        # deviation from perfect cubic/FCC symmetry is enough for ASE's
        # strict lattice classifier to call it something lower-symmetry
        # (e.g. triclinic) with an entirely different set of special points.
        # Falling back to the auto-detected path for whatever lattice type
        # this actually is, rather than failing outright.
        print(f"Requested path {Q_PATH!r} isn't valid for this cell's "
              f"detected Bravais lattice ({cell.get_bravais_lattice()}) - "
              f"missing label {e}. Falling back to the auto-detected path "
              f"for this lattice instead.")
        n_segments = 5
        bandpath = cell.bandpath(None, npoints=N_POINTS_PER_SEGMENT * n_segments)
    q_points_frac = bandpath.kpts   # fractional coords, primitive reciprocal lattice
    xcoords, special_x, special_labels = bandpath.get_linear_kpoint_axis()

    print(f"Computing spectral weights along {len(q_points_frac)} q-points...")
    all_freqs = []
    all_weights = []
    for iq, q in enumerate(q_points_frac):
        freqs, weights = spectral_weights(phonon, q, M, sub, n_sublattice_sites=2)
        all_freqs.append(freqs)
        all_weights.append(weights)
        if (iq + 1) % 20 == 0 or iq == len(q_points_frac) - 1:
            print(f"  {iq + 1}/{len(q_points_frac)}")
    all_freqs = np.array(all_freqs)      # (n_q, n_bands)
    all_weights = np.array(all_weights)  # (n_q, n_bands)

    np.savez(f"{OUTPUT_PREFIX}_raw.npz", xcoords=xcoords, freqs=all_freqs,
             weights=all_weights, special_x=special_x, special_labels=special_labels)
    print(f"Saved raw data to {OUTPUT_PREFIX}_raw.npz")

    # --- broaden into a spectral function A(q, omega) on a regular frequency grid ---
    freq_grid = np.linspace(0, FREQ_MAX_THZ, 400)
    A = np.zeros((len(freq_grid), len(q_points_frac)))
    for iq in range(len(q_points_frac)):
        for f, w in zip(all_freqs[iq], all_weights[iq]):
            if w < 1e-4 or f < 0:
                continue
            A[:, iq] += w * np.exp(-0.5 * ((freq_grid - f) / SIGMA_THZ) ** 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    spectral_cmap = LinearSegmentedColormap.from_list(
        "white_to_color", ["white", SPECTRAL_COLOR])
    im = ax.pcolormesh(xcoords, freq_grid, A, shading="auto", cmap=spectral_cmap,
                        norm=PowerNorm(gamma=SPECTRAL_GAMMA, vmin=0, vmax=A.max()))
    for x in special_x:
        ax.axvline(x, color="0.6", lw=0.6, alpha=0.8)
    ax.set_xticks(special_x)
    ax.set_xticklabels([lab if lab != "G" else r"$\Gamma$" for lab in special_labels])
    ax.set_ylabel("Frequency (THz)")
    ax.set_ylim(0, FREQ_MAX_THZ)
    fig.colorbar(im, ax=ax, label="Spectral weight (arb. units)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_spectral_function.png", dpi=200)
    print(f"Saved plot to {OUTPUT_PREFIX}_spectral_function.png")


if __name__ == "__main__":
    main()
