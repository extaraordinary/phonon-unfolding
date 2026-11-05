# phonon-unfolding

Phonon band unfolding for chemically disordered special quasirandom structure (SQS) supercells, applied 
to high-entropy rocksalt oxides.

Multi-cation rocksalt oxides have no periodic primitive cell once disorder is introduced, so a phonon calculation on an SQS supercell produces a dense, hard-to-interpret band structure with 3N branches instead of a clean 6-branch dispersion. Band unfolding recovers a spectral function A(q, ω) on the *parent* primitive Brillouin zone, projecting the supercell modes back onto primitive-cell wavevectors 
so the result can be compared directly to the parent compound's phonons (and, in principle, to inelastic scattering/EELS measurements). The unfolding implementation follows the spectral-weight formalism of 
Allen et al. (Phys. Rev. B 87, 085322 (2013)), generalized to a multi-atom primitive basis and resolved 
by sublattice, so mass/force-constant disorder shared across chemically distinct species on the same site is captured correctly.

## Pipeline

```
generate_sqs.py                         build a cubic rocksalt SQS via icet
        │
        ▼
00_relax_structure_chgnet.py            relax positions + cell with CHGNet
        │
        ▼
make_ideal_primitive.py                 build the ideal primitive cell to unfold onto
        │
        ▼
01_compute_force_constants_chgnet.py    phonopy finite-displacement force constants (CHGNet)
        │
        ▼
02_run_unfolding.py                     unfold + plot the spectral function
```

`unfold_core.py` and `unfold_setup.py` hold the unfolding math and are imported by `02_run_unfolding.py`; they don't need editing. Each other script has a small block of parameters at the top and reads the previous script's output file by default.

`01_compute_force_constants_chgnet.py` is the only CPU-heavy step (finite-displacement forces, scaling with atom count x number of displacements) - worth a GPU for a full-size supercell. Everything else runs in secs to mins on CPU, which makes it cheap to iterate on just the unfolding step: re-plotting with different styling, trying a different primitive-cell convention, or unfolding a new SQS structure without recomputing force constants.

## Usage

0. **`generate_sqs.py`** *(skip if an SQS already exists)* - generates a cubic 
   rocksalt SQS via `icet` for a given cation composition.
   `pip install icet` first.
1. **`00_relax_structure_chgnet.py`** - relaxes atomic positions + cell with
   CHGNet. `PRESERVE_CELL_SHAPE = True` (default) keeps the cell shape fixed
   during relaxation, which is what keeps the unfolding step simple later.
2. **`make_ideal_primitive.py`** - builds the ideal reference primitive cell
   to unfold onto, using the *relaxed* structure's actual volume/lattice.
   Two modes (`MODE` at the top): `"cubic_guess"` (default, works when the
   relaxed cell stayed close to cubic) or `"known_transformation"` (for a
   genuinely distorted cell - see "Key technical notes" below).
3. **`01_compute_force_constants_chgnet.py`** - computes force constants for
   the relaxed SQS cell via CHGNet + phonopy finite displacements.
4. **`02_run_unfolding.py`** - unfolds onto the primitive Brillouin zone and
   plots the spectral function.

## What each file does

| File | Purpose |
|---|---|
| `generate_sqs.py` | build a cubic rocksalt SQS via `icet` for a given cation composition |
| `00_relax_structure_chgnet.py` | relax an SQS structure with CHGNet (positions + optionally cell shape/volume) |
| `make_ideal_primitive.py` | build the ideal primitive cell to unfold onto, from the relaxed structure |
| `unfold_core.py` | core unfolding math (spectral-weightformalism) |
| `unfold_setup.py` | helpers: find the primitive→supercell transformation matrix, assign atoms to sublattices |
| `01_compute_force_constants_chgnet.py` | compute force constants for the relaxed SQS cell via CHGNet |
| `02_run_unfolding.py` | run the unfolding along a q-path and plot the spectral function |
| `validation_plot.png` | ref: the unfolding math correctly reproduces exact primitive-cell phonons for a disorder-free test case (see Validation below) |

The pipeline self-diagnoses several common failure modes by printing diagnostics:

- **Cation/anion counts should match the expected stoichiometry** (e.g.
  roughly equal for 1:1 rocksalt). Printed by `make_ideal_primitive.py`.
- **"far from its nearest ideal site" warnings** - a handful (a small
  fraction of atoms, sub-1 Å) is normal, reflecting real relaxation
  displacement. A large fraction of atoms all off by ~1 bond length (~2 Å
  for these oxides) means the reference primitive cell's basis convention
  doesn't match the structure - see the conventional-vs-rhombohedral note
  below.
- **`det(M)` should equal the number of primitive cells packed into the
  supercell** - i.e. (atoms in the structure) / (atoms in the reference
  primitive cell). Printed alongside the transformation matrix.
- **Total spectral weight should sum to exactly the number of primitive
  branches** (6 for a 2-atom basis, 24 for the 8-atom conventional basis) at
  *every* q-point - this is a hard physical requirement (a sum rule), not
  just a sanity heuristic. Worth spot-checking (`weights.sum()`) if a result
  looks suspicious.

## Key technical notes

A few non-obvious things came up while building this pipeline that are worth noting:

- **`phonon.primitive` vs. `phonon.supercell`.**
  Some phonon workflow tools (e.g. atomate2's `PhononMaker`) can expand an
  SQS cell *further* to satisfy a minimum supercell length, meaning the
  force constants end up living on a bigger box than the original structure.
  Phonopy's band-structure evaluation reads q-points relative to
  `phonon.primitive`, not `phonon.supercell` - these differ exactly when
  this expansion happens, and using the wrong one silently gives wrong (not
  crashing) results. This repo's scripts avoid the issue entirely -
  `01_compute_force_constants_chgnet.py` always builds with
  `supercell_matrix=[1,1,1]`, so primitive and supercell are the same object
  from the start.
- **Matrix multiplication order in `assign_sublattices`.** Recovering the
  primitive cell from `A_sc = M @ A_prim` requires `A_prim = inv(M) @ A_sc`
  (left-multiplication) - using `A_sc @ inv(M)` instead is silently wrong,
  but invisible whenever `M` is a scalar multiple of the identity (which
  commutes either way). It only surfaces for a genuinely non-uniform `M`,
  e.g. `[[4,0,0],[0,4,0],[0,0,2]]`.
- **Two rocksalt primitive-cell conventions.** `"known_transformation"` mode
  can recover either the 2-atom rhombohedral primitive (~60° angles, 1
  formula unit) or the conventional 4-formula-unit cubic cell (8 atoms,
  ~90° angles), depending on what convention `KNOWN_M` was defined against.
  `make_ideal_primitive.py` detects which one automatically from the
  recovered cell's angles and builds the matching basis - getting this wrong
  silently generates only 1/4 of the true ideal sites, which shows up as a
  large fraction of atoms mismatched by ~1 bond length.
- **The conventional 8-atom cell is a *simple cubic* Bravais lattice, not
  FCC.** ASE's lattice classifier sees only the 3 cell vectors, not the
  atomic basis, so it reports `"CUB"` (different high-symmetry point labels
  - Γ,X,M,R - from FCC's Γ,X,W,K,L) for the conventional cell even though
  the underlying physics is the same rocksalt structure. Both scripts try
  the requested `Q_PATH` first and fall back to auto-detecting the correct
  path if the labels don't exist for the detected lattice - this fallback is
  the *normal* path whenever the conventional-cell convention is in use, not
  a rare edge case.
- **Plot colors.** A white background with a `PowerNorm(gamma=0.5)` color
  scale (rather than plain linear) keeps the faint, diffuse spectral
  weight visible instead of washed out next to the much brighter concentrated
  peaks near the ideal bands.

## Validation

The core unfolding math (`unfold_core.py`, `unfold_setup.py`) was checked
against independent ground-truth calculations rather than trusted on first
principles:

1. **Disorder-free supercell test.** Unfolding a perfect (non-disordered)
   rocksalt supercell back onto its own primitive cell must exactly
   reproduce the true primitive-cell dispersion - this is the standard
   correctness test for any unfolding implementation. `validation_plot.png`
   is the result: a clean, correct MgO phonon dispersion, matching an
   independently-computed primitive-cell calculation.
2. **Non-uniform supercell expansion test.** A synthetic case reproducing
   the behavior of tools that can expand a base cell by a non-scalar
   `supercell_matrix` caught the `assign_sublattices` matrix-order bug and
   the `phonon.primitive`-vs-`phonon.supercell` issue above; now fixed and
   reverified against ground truth.
3. **Conventional-cell convention test.** A synthetic case using the 8-atom
   conventional cell caught the missing-3-of-4-sites bug described above;
   fixed and reverified - zero mismatch warnings, exact sum rule at every
   q-point tested, and exact recovery of the true primitive cell from
   `KNOWN_M`.
