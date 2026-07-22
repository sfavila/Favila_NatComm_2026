# Analyses and figures for Favila et al., Nature Communications (2026)

This repository contains the analysis and figure code for the following paper:

Favila S.E. and Aly M (in press). Hippocampal mechanisms underlying the resolution of competition in memory and perception. *Nature Communications*.

There are two datasets associated with this release:
- Raw data + `fmriprep`/`mriqc` derivatives: ds008366
- ROI masks, pRF estimates, preprocessed eye-tracking data, GLM outputs, 
and searchlight maps used by the analyses below: https://osf.io/c948a

Intermediate results (the CSVs each notebook reads/writes) are also included
directly in `dataframes/`, so the figure notebooks can be run standalone
without first downloading the datasets above or re-running `precompute/`.

## Layout

```
mgvs_release/           shared Python package (config, plotting, eye-tracking
                        parsing, ROI loading, stats helpers) used across the
                        notebooks below
precompute/             batch scripts that generate dataframes/ from the 
                        derivatives (GLM residual extraction + per-subject
                        aggregation); see precompute/README.md
figures/
  figure1.ipynb            Fig 1D, 1E
  figure2.ipynb            Fig 2C-F + text-only stats
  supp_figure1.ipynb       Supp Fig 1A-D
  supp_figure2.ipynb       Supp Fig 2A-D
  figure3.ipynb            Fig 3B, 3C + text-only stats
  supp_figure3.ipynb       Supp Fig 3A-D
  supp_figure4.ipynb       Supp Fig 4A-E
  figure4.ipynb            Fig 4B-D
  supp_figure5.ipynb       Supp Fig 5A-B
  figure5.ipynb            Fig 5A-C
  supp_figure6.ipynb       Supp Fig 6A-C
dataframes/             intermediate CSVs read/written by the notebooks above
plots/                  SVGs written by the notebooks above, named after the
                        manuscript panel they reproduce (e.g. fig3b.svg,
                        sfig4d_ppa.svg)
stats/                  R scripts for the manuscript's mixed-effects models
                        (lme4/lmerTest); see stats/README.md for which
                        script covers which figure/panel
```

## Setup

```
conda env create -f environment.yml
conda activate mgvs-release
pip install -e .
```

Then point the code at wherever you've placed the two datasets described
above:
```
export MGVS_BIDS_DIR=/path/to/MGVS_fmri_BIDS
export MGVS_DERIV_DIR=/path/to/merged_derivatives   # see below
```

The two datasets are split for distribution, but every notebook and
`precompute/` script here reads/writes through a single `MGVS_DERIV_DIR`,
expecting `fmriprep/`, `mriqc/`, `rois/`, `prfanalyze-vista/`,
`eyetracking-preproc/`, `glm-scene_nii/`, `glm-loc_resids/`, etc. to live
together under one `derivatives/` directory. Merge the two before pointing
`MGVS_DERIV_DIR` at them.

If you only want to run the figure notebooks against the included
`dataframes/`, none of the above is required — just `pip install -e .` and
open a notebook.

## Run order

`dataframes/` ships pre-populated, so every notebook runs standalone out of
the box. Only relevant if you're regenerating everything from scratch:

- `figure1.ipynb` before `figure2.ipynb` and `figure3.ipynb` (writes
  `learn_subj_firstlast.csv`, read by figure2's session-correlation section).
- `figure2.ipynb` before `supp_figure1.ipynb`/`supp_figure2.ipynb` (writes
  `search_data_all.csv`, which both supplements read). The two supplements
  have no dependency on each other.
- `figure3.ipynb` before `supp_figure3.ipynb`/`supp_figure4.ipynb` (writes
  `pattern_sim_avg.csv`, `pattern_sim_behav_data.csv`,
  `pattern_sim_behav_data_unsorted.csv`). The two supplements have no
  dependency on each other.
- Figures 4 and 5: no order dependency. `precompute/fig4_glm_resids_aggregate.py`
  and `precompute/fig5_glm_resids_aggregate.py` must be run first if
  regenerating their dataframes from raw derivatives (see
  `precompute/README.md`); `figure5.ipynb`'s precompute step additionally
  depends on `figure3.ipynb` having been run for `pattern_sim_behav_data.csv`.

## Statistics

Mixed-effects models reported in the manuscript are in `stats/` (R, not
Python) — run e.g. `Rscript stats/fig5_stats.R` from the repo root. See
`stats/README.md` for which script covers which figure/panel and required
packages (`lme4`, `lmerTest`, `dplyr`, `emmeans`, `parameters`).
