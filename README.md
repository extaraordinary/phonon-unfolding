# phonon-unfolding
Spectral weight phonon band unfolding for SQS supercells

# Phonon band unfolding for high-entropy rocksalt oxides

This is the complete workflow for computing and analyzing CHGNet phonons for
J14 and J14+Cr SQS structures, including phonon band unfolding onto the
primitive rocksalt Brillouin zone for meaningful comparison to parent MgO
and EELS data.

## Folder structure

```
workflow/
├── README.md
    ├── generate_sqs.py
    ├── 00_relax_structure_chgnet.py
    ├── make_ideal_primitive.py
    ├── unfold_core.py
    ├── unfold_setup.py
    ├── 01_compute_force_constants_chgnet.py
    ├── 02_run_unfolding.py
    └── validation_plot.png
```

The full pipeline via atomate2's `PhononMaker`: relaxation, force constants, 
branch bandwidths, and unfolding, as standalone scripts to run locally.
Useful when you want to iterate on just the unfolding step — re-plotting
with different styling, testing a different primitive-cell convention,
generating a new SQS or primitive cell. `01_compute_force_constants_chgnet.py`
is the one CPU/GPU-heavy step here (finite-displacement forces); everything
else is fast.

## Step-by-step: VS Code workflow

Run in order, in the same working directory (each script's output feeds
the next one's input):

0. **`generate_sqs.py`** *(skip if you already have an SQS structure)* —
   generates a cubic rocksalt SQS via `icet` for a given composition.
   `pip install icet` first.
1. **`00_relax_structure_chgnet.py`** — relaxes atomic positions + cell with
   CHGNet. `PRESERVE_CELL_SHAPE = True` (default) keeps the cell shape fixed
   during relaxation, which is what keeps the unfolding step simple later.
2. **`make_ideal_primitive.py`** — builds the ideal reference primitive cell
   to unfold onto, using your *relaxed* structure's actual volume/lattice.
   Two modes (`MODE` at the top): `"cubic_guess"` (default, works when your
   relaxed cell stayed close to cubic) or `"known_transformation"` (for a
   genuinely distorted cell — see "Key technical notes" below).
3. **`01_compute_force_constants_chgnet.py`** — computes force constants for
   your relaxed SQS cell via CHGNet + phonopy finite displacements. This is
   the slow step (scales with number of atoms × number of species); fine on
   CPU for small test structures, worth GPU/TPU for a full supercell.
4. **`02_run_unfolding.py`** — unfolds onto the primitive Brillouin zone and
   plots the spectral function. This one is fast (just phonopy + numpy + matplotlib).

## What each file does

| File | Purpose |
|---|---|
| `generate_sqs.py` | Build a cubic rocksalt SQS via `icet` for a given cation composition |
| `00_relax_structure_chgnet.py` | Relax an SQS structure with CHGNet (positions + optionally cell shape/volume) |
| `make_ideal_primitive.py` | Build the ideal primitive cell to unfold onto, from your relaxed structure |
| `unfold_core.py` | Core unfolding math (spectral-weight / Allen et al. formalism) — no need to edit |
| `unfold_setup.py` | Helpers: find the primitive→supercell transformation matrix, assign atoms to sublattices — no need to edit |
| `01_compute_force_constants_chgnet.py` | Compute phonopy force constants for the relaxed SQS cell via CHGNet |
| `02_run_unfolding.py` | Run the unfolding along a q-path and plot the spectral function |
| `validation_plot.png` | Reference: the unfolding math correctly reproduces exact primitive-cell phonons for a disorder-free test case (see Validation below) |

## What to watch for

These are real issues that came up during development and are now handled,
but the code will only self-diagnose if you read what it prints:

- **Cation/anion counts should match your expected stoichiometry** (e.g.
  roughly equal for 1:1 rocksalt). Printed by `make_ideal_primitive.py`.
- **"far from its nearest ideal site" warnings** — a handful (small
  fraction of atoms, sub-1 Å) is normal, reflecting real relaxation
  displacement. A large fraction of atoms all off by ~1 bond length (~2 Å
  for these oxides) means the reference primitive cell's basis convention
  doesn't match your structure — see the conventional-vs-rhombohedral note
  below.
- **`det(M)` should equal the number of primitive cells packed into the
  supercell** — i.e. (atoms in your structure) / (atoms in the reference
  primitive cell). Printed alongside the transformation matrix.
- **Total spectral weight should sum to exactly the number of primitive
  branches** (6 for a 2-atom basis, 24 for the 8-atom conventional basis) at
  *every* q-point — this is a hard physical requirement (a sum rule), not
  just a sanity heuristic. Neither script currently prints this directly, but
  it's straightforward to check: `weights.sum()` should equal 6 or 24
  depending on which basis your primitive cell ended up using — worth
  spot-checking if a result looks suspicious.

## Key technical notes

A few non-obvious things were discovered getting this pipeline right — worth
knowing so you don't reintroduce them if you modify this code:

- **`phonon.primitive` vs. `phonon.supercell`.**
  Atomate2's `PhononMaker` can expand your SQS cell *further* to satisfy
  `MIN_SUPERCELL_LENGTH`, meaning the force constants live on a bigger box
  than your original structure. Phonopy's band-structure evaluation reads
  q-points relative to `phonon.primitive`, not `phonon.supercell` — these
  differ exactly when this expansion happens, and using the wrong one
  silently gives wrong (not crashing) results. The provided python scripts
  don't need this — `01_compute_force_constants_chgnet.py` always builds
  with `supercell_matrix=[1,1,1]`, so primitive and supercell are the same
  object from the start.
- **Matrix multiplication order in `assign_sublattices`.** Recovering the
  primitive cell from `A_sc = M @ A_prim` requires `A_prim = inv(M) @ A_sc`
  (left-multiplication) — using `A_sc @ inv(M)` instead is silently wrong,
  but invisible whenever `M` is a scalar multiple of the identity (which
  commutes either way). Only surfaces for a genuinely non-uniform `M`, e.g.
  `[[4,0,0],[0,4,0],[0,0,2]]`. Fixed in `unfold_setup.py`.
- **Two rocksalt primitive-cell conventions.** `"known_transformation"` mode
  can recover either the 2-atom rhombohedral primitive (~60° angles, 1
  formula unit) or the conventional 4-formula-unit cubic cell (8 atoms,
  ~90° angles), depending on what convention your `KNOWN_M` was defined
  against. `make_ideal_primitive.py` now detects which
  one automatically from the recovered cell's angles and build the matching
  basis — getting this wrong silently generates only 1/4 of the true ideal
  sites, which shows up as a large fraction of atoms mismatched by
  ~1 bond length.
- **The conventional 8-atom cell is a *simple cubic* Bravais lattice, not
  FCC.** ASE's lattice classifier sees only the 3 cell vectors, not the
  atomic basis, so it reports `"CUB"` (different high-symmetry point labels
  — Γ,X,M,R — from FCC's Γ,X,W,K,L) for the conventional cell even though
  the underlying physics is the same rocksalt structure. Both scripts try
  your requested `Q_PATH` first and fall back to auto-detecting the correct
  path if the labels don't exist for the detected lattice — this fallback
  is the *normal* path whenever you're using the conventional-cell
  convention, not a rare edge case.
- **Plot colors.** White background with a `PowerNorm(gamma=0.5)` color
  scale (rather than plain linear) keeps the faint, diffuse spectral
  weight visible instead of washed out next to the much brighter concentrated 
  peaks near the ideal bands.

## Validation

The core unfolding math (`unfold_core.py`, `unfold_setup.py`) was checked against independent ground-truth calculations rather than trusted on first principles:

1. **Disorder-free supercell test.** Unfolding a perfect (non-disordered)
   rocksalt supercell back onto its own primitive cell must exactly
   reproduce the true primitive-cell dispersion — this is the standard
   correctness test for any unfolding implementation. `validation_plot.png`
   is the result: clean, correct MgO phonon dispersion, matching an
   independently-computed primitive-cell calculation.
2. **Non-uniform supercell expansion test.** A synthetic case reproducing
   atomate2's real behavior (a base cell expanded by a non-scalar
   `supercell_matrix`) caught the `assign_sublattices` matrix-order bug and
   the `phonon.primitive`-vs-`phonon.supercell` issue above; now fixed and
   reverified against ground truth.
3. **Conventional-cell convention test.** A synthetic case using the
   8-atom conventional cell (matching what turned out to be your actual
   structure's convention) caught the missing-3-of-4-sites bug; fixed and
   reverified — zero mismatch warnings, exact sum rule at every q-point
   tested, and exact recovery of the true primitive cell from `KNOWN_M`.