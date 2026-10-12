# figures/

One notebook per manuscript figure. `dataframes/` ships pre-populated, so
every notebook runs standalone out of the box.

## Run order

Only relevant if you're regenerating everything from scratch:

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
