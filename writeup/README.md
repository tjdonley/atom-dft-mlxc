# SIMPLE-Xhole-writeup

Publication-ready (PRL-style) writeup of the **SIMPLE** (Scale-Invariant Multipole
Local Expansion) density features and the **self-consistent exchange-hole
functional** built on them. Intro and conclusions are handled separately; this
document contains the **Methods** and **Results** sections plus a pedagogical
**Appendix** with all derivations.

## Build

```
latexmk -pdf main.tex      # or: pdflatex main; bibtex main; pdflatex main x2
```

- `main.tex` — PRL/`revtex4-2` main text (Methods + Results) and `\appendix`.
- `appendix.tex` — full derivations (followable by a first-year graduate student).
- `refs.bib` — all references DOI-resolved via
  `amedford6-admin/scripts/doi2bib.py`.
- `figures/` — feature-validation figures (copied from `NOLE_writeup/`; regenerated
  on the `simple-writeup` branch of `atom-dft-SIMPLE` in Phase C).
- `scripts/` — pointers/copies of the figure + benchmark generators (Phase C/D).

## Status (gated phases)

- **Phase A (done):** scaffold + theory skeleton; feature figures wired; core
  references seeded.
- **Phase B:** curated `simple-writeup` code branch off `master` (rename + docs).
- **Phase C:** regenerate feature results/figures on the clean branch.
- **Phase D:** regenerate GEA + exchange-hole results (`v_x` vs OEP; energies/gaps
  vs HF/PBE/r²SCAN). Items marked `[...]` in red in the PDF are placeholders
  pending these runs.
- **Phase E:** final assembly; figure/table → script map added here.

## Source material distilled

Theory text is distilled from `NOLE_writeup/SIMPLE.tex` (features),
`NOLE_writeup/NOLE.tex` (operator projections), and
`atom-dft-SIMPLE/scripts/learned_hole/hole_derivation/hole_derivation.tex`
(exchange hole). See `../cleanup_plan.md` for the outline.
