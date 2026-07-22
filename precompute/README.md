# precompute/

Standalone, cluster-only scripts that don't belong in a notebook meant to
be re-run casually -- either because they're slow or because they're
themselves batch jobs over the full cohort.

Run and validated for the full 32-subject cohort. See each script's module
docstring for what it reads/writes and any non-obvious behavior. 

## GLM scripts

`glm_scene_nii.py` and `glm_loc_resids.py` each fit one first-level GLM per
subject over the full search-task session.

```
python precompute/glm_scene_nii.py
python precompute/glm_loc_resids.py
```

## fig4/fig5 aggregation scripts

`fig4_glm_resids_aggregate.py` and `fig5_glm_resids_aggregate.py` are
standalone precompute, not run by any notebook directly.

```
python precompute/fig4_glm_resids_aggregate.py agg_data --sacc first 
python precompute/fig4_glm_resids_aggregate.py curve_averages --sacc first 
python precompute/fig5_glm_resids_aggregate.py --roi hipp --sep-type corr
```
`agg_data` always writes into `<deriv_dir>/glm-loc_resids/` and `curve_averages`
and `fig5_glm_resids_aggregate` write to `dataframes/`

### fig4_glm_resids_aggregate.py

Two steps, run per `--sacc` (`first`/`last`):

1. `agg_data` -- merges each subject's GLM-residual data (produced by `glm_loc_resids.py`') 
   with retinotopy (pRF angle/eccentricity/R2/sigma) and behavioral (saccade/response) 
   data, bins voxels by angular distance from target/competitor, and writes one
   `..._merged_{sacc}sacc.parquet` per subject back into
   `<deriv_dir>/glm-loc_resids/`.
2. `curve_averages` -- reads those merged per-subject Parquet files, z-scores and
   collapses to one row per trial, and writes the concatenated
   `dataframes/evc_prep_curve_{sacc}.csv` that `figure4.ipynb` reads.

### fig5_glm_resids_aggregate.py

**Depends on `figure3.ipynb` having already been run**. Run `figure3.ipynb` first if
`dataframes/pattern_sim_behav_data.csv` isn't already populated.

One step: reads the `..._merged_firstsacc.parquet` files figure4's 
precompute wrote and merges with a per subject pattern similarity.
Writes `dataframes/<roi>_rank_corr.csv` (or `<roi>_rank_univariate.csv` 
for `--sep-type univ`).

Needs one call per ROI actually used by `figure5.ipynb`/`supp_figure6.ipynb`:
`hipp`, `CoS-places`, `rsc`, `phc`, `hipp_anterior`, `hipp_posterior`
(all `--sep-type corr`), and `DAN` (`--sep-type univ`).