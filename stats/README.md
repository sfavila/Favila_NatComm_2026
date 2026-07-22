# stats/

R scripts for the paper's mixed-effects-model results. These cover
every reported statistic that isn't already computed inline in a figure
notebook.

| Script | Manuscript figure | Reads |
|---|---|---|
| `sfig3_stats.R` | Supp Fig 3D, plus one text-only model | `dataframes/pattern_sim_behav_data.csv` |
| `fig4_stats.R` | Fig 4D | `dataframes/evc_prep_curve_last.csv` |
| `sfig5_stats.R` | Supp Fig 5B | `dataframes/evc_prep_curve_first.csv` |
| `fig5_stats.R` | Fig 5C, plus text-only models | `dataframes/hipp_rank_corr.csv` |
| `sfig6_stats.R` | Supp Fig 6B and 6C, plus one text-only model | `dataframes/{hipp_anterior,hipp_posterior,CoS-places,rsc,phc}_rank_corr.csv`, `dataframes/DAN_rank_univariate.csv` |

"Text-only" means the statistic is reported in the manuscript's prose, not
tied to a specific figure panel.

Run from the repo root, e.g. `Rscript stats/fig5_stats.R`. Requires `lme4`,
`lmerTest`, `dplyr`, `emmeans`, and `parameters`.

Each analysis fits two versions of the same model: a commented-out maximal
random-effects structure (kept in the file for robustness/reviewer
transparency) and the reduced structure actually reported in the manuscript. 
Every model is labeled with the figure panel (or "reported in manuscript" for 
text-only statistics) it corresponds to.
