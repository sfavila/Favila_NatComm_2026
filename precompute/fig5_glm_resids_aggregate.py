"""Standalone precompute script for Figure 5.

Aggregates per-vertex GLM residual timeseries (``<deriv_dir>/glm-loc_resids/``)
into the merged per-ROI CSVs that ``figures/figure5.ipynb`` reads.

Needs one call per ROI actually used by
``figure5.ipynb``/``supp_figure6.ipynb``: ``hipp``, ``CoS-places``, ``rsc``,
``phc``, ``hipp_anterior``, ``hipp_posterior`` (all ``--sep-type corr``),
and ``DAN`` (``--sep-type univ``). Depends on
``precompute/fig4_glm_resids_aggregate.py``'s ``agg_data`` step already
having produced the per-subject ``merged_{sacc}sacc.parquet`` files it reads,
and on ``figures/figure3.ipynb`` having already been run to produce 
``dataframes/pattern_sim_behav_data.csv``, which ``prepare_roi_sep`` 
reads on every invocation.

Usage::

    python precompute/fig5_glm_resids_aggregate.py --roi hipp --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi CoS-places --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi rsc --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi phc --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi hipp_anterior --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi hipp_posterior --sep-type corr
    python precompute/fig5_glm_resids_aggregate.py --roi DAN --sep-type univ

``--subjects`` overrides the default full 32-subject list (e.g. for a
1-subject test before committing to a full run).
"""
import argparse
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats

from mgvs_release import config

N_SCENE_PAIRS = 12  # number of scene categories


def prepare_roi_sep(dataframes_dir):
    """Rebuild ``roi_sep`` and ``roi_univ_sep``.

    ``roi_sep`` ranks every subject's trials within each ROI by pattern
    similarity (``corr_z``, from ``dataframes/pattern_sim_behav_data.csv``)
    into a 2-way ``rank_bin``. ``roi_univ_sep`` does the same ranked instead
    by DAN's univariate activity (``--sep-type univ``)
    """
    # corr_z/DAN are identical regardless of sacc_type so filtering to 'first' here is
    # pure deduplication, not a meaningful choice between two analyses.
    roi_sep = pd.read_csv(op.join(dataframes_dir, "pattern_sim_behav_data.csv")).query("sacc_type=='first'")

    roi_sep = roi_sep.groupby(["subj", "roi"]).apply(
        lambda x: x.sort_values(by="corr_z")).reset_index(drop=True)
    roi_sep["rank"] = np.tile(np.arange(1, N_SCENE_PAIRS + 1), int(len(roi_sep) / N_SCENE_PAIRS))
    # Median split: bottom half of ranks (lowest pattern similarity) -> bin 1, top half -> bin 2.
    roi_sep["rank_bin"] = np.where(roi_sep["rank"] <= N_SCENE_PAIRS // 2, 1, 2)

    # DAN is a univariate measure, broadcast identically across every roi row 
    # in pattern_sim_behav_data.csv. 'hipp' here is an arbitrary single row to
    # deduplicate, not a meaningful selection of the hipp ROI specifically.
    roi_univ_sep = roi_sep.query("roi=='hipp'").groupby(["subj", "roi"]).apply(
        lambda x: x.sort_values(by="DAN")).reset_index(drop=True)
    roi_univ_sep["rank"] = np.tile(np.arange(1, N_SCENE_PAIRS + 1), int(len(roi_univ_sep) / N_SCENE_PAIRS))
    roi_univ_sep["rank_bin"] = np.where(roi_univ_sep["rank"] <= N_SCENE_PAIRS // 2, 1, 2)
    roi_univ_sep.loc[:, "roi"] = "DAN"
    roi_univ_sep = roi_univ_sep.rename(columns={"DAN": "univ"})

    return roi_sep, roi_univ_sep


def curve_averages(subjects, roi_sep, roi, deriv_dir, dataframes_dir, sep_type="corr"):
    """Collapse one ROI's merged residuals to one row per trial and write its rank CSV.

    Reads the ``merged_firstsacc.parquet`` files
    ``fig4_glm_resids_aggregate.py``'s ``agg_data`` wrote, merges
    in each trial's pattern-similarity or univariate rank (``roi_sep``),
    z-scores BOLD within each trial-condition combination, takes the median
    across vertices within each trial, and writes
    ``dataframes/<roi>_rank_corr.csv`` (or ``<roi>_rank_univariate.csv`` for
    ``sep_type="univ"``).
    """
    roi_sep = roi_sep.query("roi==@roi")

    avg_df = []

    for subj in subjects:
        print(subj)

        data_resid_df = pd.read_parquet(op.join(
            deriv_dir, "glm-loc_resids", subj, "ses-01",
            f"{subj}_ses-01_glm_loc_resids_merged_firstsacc.parquet"))

        # Attach each trial's pattern-similarity (or univariate) rank bin.
        if sep_type == "corr":
            data_resid_df = data_resid_df.merge(
                roi_sep[["subj", "stim_category", "corr_z", "rank_bin", "rank", "next_targ_comp_dist"]],
                on=["subj", "stim_category", "next_targ_comp_dist"])
        elif sep_type == "univ":
            data_resid_df = data_resid_df.merge(
                roi_sep[["subj", "stim_category", "univ", "rank_bin", "rank", "next_targ_comp_dist"]],
                on=["subj", "stim_category", "next_targ_comp_dist"])

        # Z-score within each specific trial-condition combination. For visualization only.
        data_resid_df["bold_z"] = data_resid_df.groupby(
            ["subj", "roi", "stim_category", "stim_name", "cue_validity", "rank_bin", "next_targ_comp_dist"]
        )["bold"].transform(lambda x: stats.zscore(x))

        # Collapse to one row per trial (median across the ROI's vertices).
        subj_df = data_resid_df.groupby(
            ["subj", "run", "trial", "stim_category", "stim_name", "cue_validity", "rank_bin",
             "next_targ_comp_dist", "ang_dist_bin", "cond"]
        ).median(numeric_only=True).reset_index()
        avg_df.append(subj_df)

    avg_df = pd.concat(avg_df).reset_index(drop=True)
    avg_df = avg_df.query("cue_validity=='valid'")
    if sep_type == "corr":
        (avg_df[["subj", "run", "trial", "stim_category", "stim_name", "cue_validity", "rank", "rank_bin",
                 "next_targ_comp_dist", "ang_dist_bin", "cond", "corr_z", "bold_z", "bold"]]
         .dropna().to_csv(op.join(dataframes_dir, f"{roi}_rank_corr.csv"), index=False))
    elif sep_type == "univ":
        (avg_df[["subj", "run", "trial", "stim_category", "stim_name", "cue_validity", "rank", "rank_bin",
                 "next_targ_comp_dist", "ang_dist_bin", "cond", "univ", "bold_z", "bold"]]
         .dropna().to_csv(op.join(dataframes_dir, f"{roi}_rank_univariate.csv"), index=False))


def main():
    """CLI entry point: build the rank CSV for one ROI (``--roi``/``--sep-type``) across ``--subjects``."""
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi", required=True,
                         help="e.g. hipp, CoS-places, rsc, phc, hipp_anterior, hipp_posterior, DAN")
    parser.add_argument("--sep-type", choices=["corr", "univ"], default="corr")
    parser.add_argument("--subjects", default=None,
                         help="Comma-separated subject list, e.g. sub-01 for a 1-subject test. "
                              "Defaults to the full cohort (mgvs_release.config.SUBJECTS).")
    parser.add_argument("--deriv-dir", default=config.DERIV_DIR)
    parser.add_argument("--dataframes-dir", default="dataframes")
    args = parser.parse_args()

    subjects = args.subjects.split(",") if args.subjects else config.SUBJECTS

    roi_sep, roi_univ_sep = prepare_roi_sep(args.dataframes_dir)
    sep_df = roi_univ_sep if args.sep_type == "univ" else roi_sep
    curve_averages(subjects, sep_df, args.roi, args.deriv_dir, args.dataframes_dir, sep_type=args.sep_type)


if __name__ == "__main__":
    main()
