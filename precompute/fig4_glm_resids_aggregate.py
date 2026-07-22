"""Standalone precompute script for Figure 4.

Aggregates per-vertex GLM residual timeseries (``<deriv_dir>/glm-loc_resids/``)
into the merged per-trial CSVs that ``figures/figure4.ipynb`` reads.

Two independent steps, run in order for a given ``--sacc``:

1. ``agg_data``
   reads  ``<deriv_dir>/glm-loc_resids/sub-NN/ses-01/sub-NN_ses-01_glm_loc_resids_winroizscore.parquet``
   writes ``<deriv_dir>/glm-loc_resids/sub-NN/ses-01/sub-NN_ses-01_glm_loc_resids_merged_{sacc}sacc.parquet``

2. ``curve_averages``
   reads  the ``merged_{sacc}sacc.parquet`` files ``agg_data`` wrote
   writes ``dataframes/evc_prep_curve_{sacc}.csv``

Usage::

    python precompute/fig4_glm_resids_aggregate.py agg_data --sacc first
    python precompute/fig4_glm_resids_aggregate.py agg_data --sacc last
    python precompute/fig4_glm_resids_aggregate.py curve_averages --sacc first
    python precompute/fig4_glm_resids_aggregate.py curve_averages --sacc last

``--subjects`` overrides the default full 32-subject list (e.g. for a
1-subject test before committing to a full run).
"""
import argparse
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from mgvs_release import config
from mgvs_release.behavior import prepare_sacc_both as _prepare_sacc_both

ROIS = ["V1", "V2", "V3"]


def true2brain_ang(x):
    """Convert a true screen polar angle to brain-referenced (visual-field) polar angle."""
    return np.mod((90.0 - 180.0 / np.pi * np.deg2rad(x)) + 180, 360) - 180


def restrict_ang_range(x):
    """Wrap a polar angle (degrees) into (-180, 180]."""
    return np.mod(x + 180, 360) - 180


def load_roi_masks_mgz(deriv_dir, subjects, rois=None):
    """Load V1/V2/V3 hand-drawn masks (``.mgz`` variant) for each subject.

    Returns ``{"sub-NN": {roi_name: bool_array}}``.
    """
    rois = rois if rois is not None else ROIS
    roi_masks = {}
    for subj in subjects:
        roi_masks[subj] = {}
        sub_roi_path = op.join(deriv_dir, "rois", subj, "ses-01")
        for r in rois:
            lh_r = nib.load(op.join(sub_roi_path, f"lh.{r}_hand.mgz")).get_fdata().astype(bool)
            rh_r = nib.load(op.join(sub_roi_path, f"rh.{r}_hand.mgz")).get_fdata().astype(bool)
            bi_r = np.concatenate([np.squeeze(lh_r), np.squeeze(rh_r)])
            roi_masks[subj][r] = bi_r
    return roi_masks


def load_retinotopy_data(deriv_dir, subjects, roi_masks, rois=None):
    """Load per-vertex pRF parameters (angle, eccentricity, R2, sigma) for each ROI, for each subject.

    Returns one long-format dataframe with columns
    ``subj, roi, vertex, ang, ecc, r2, sigma``.
    """
    rois = rois if rois is not None else ROIS
    ret_params = ["ang", "ecc", "r2", "sigma"]
    ret_data = []

    for subj in subjects:
        sub_ret_path = op.join(deriv_dir, "prfanalyze-vista", subj, "ses-01")

        for p in ret_params:
            lh_p = nib.load(op.join(sub_ret_path, f"lh.{p}.mgz")).get_fdata()
            rh_p = nib.load(op.join(sub_ret_path, f"rh.{p}.mgz")).get_fdata()
            bi_p = np.squeeze(np.concatenate([lh_p, rh_p]))

            for r in rois:
                p_roi = bi_p[roi_masks[subj][r]]
                verts = np.squeeze(np.argwhere(roi_masks[subj][r]))
                rdf = pd.DataFrame({"param": p, "value": p_roi, "roi": r, "subj": subj, "vertex": verts})
                ret_data.append(rdf)

    ret_data = pd.concat(ret_data).reset_index(drop=True)
    ret_data = ret_data.pivot(index=["subj", "roi", "vertex"], columns="param", values="value").reset_index()
    return ret_data


def prepare_sacc_both(bids_dir, subjects, dataframes_dir):
    """Rebuild the ``sacc_both`` per-trial dataframe (see `mgvs_release.behavior`).
    """
    return _prepare_sacc_both(bids_dir, dataframes_dir, subjects,
                               roll_cols=["comp_resp", "warp_resp", "warp_ang"])


def agg_data(subjects, sacc, deriv_dir, sacc_both, ret_data):
    """Merge each subject's residual timeseries with retinotopy/behavior and bin by angular distance.

    For each subject, merges ``glm_loc_resids.py``'s per-vertex residuals
    with ``ret_data`` (pRF params) and ``sacc_both`` (behavior), restricts to
    vertices with pRF centers near the target eccentricity and reliable pRF
    fits, bins each vertex by its angular distance from the target and
    competitor ("target"/"competitor"/"control"/"other" conditions), and
    writes one merged Parquet file per subject.
    """
    for subj in subjects:
        print(subj)
        df = pd.read_parquet(op.join(deriv_dir, "glm-loc_resids", subj, "ses-01",
                                      f"{subj}_ses-01_glm_loc_resids_winroizscore.parquet"))

        # Attach this subject's next-saccade behavior to its residual timeseries.
        df = df.merge(sacc_both.query("sacc_type==@sacc & subject==@subj"), on=["run", "trial"], how="outer")

        # On "nopred" trials there's no predicted target location to compare
        # against; fall back to the actual (true) warp location instead.
        isnopred = df["cue_validity"] == "nopred"
        df.loc[isnopred, "next_target_ang"] = df.query("cue_validity=='nopred'")["next_warp_ang"]
        df["next_target_ang_brain"] = df["next_target_ang"].apply(true2brain_ang)
        df["next_comp_ang_brain"] = df["next_comp_ang"].apply(true2brain_ang)

        # Attach this subject's retinotopy (pRF angle/eccentricity/R2/sigma).
        all_data_resid = pd.merge(df, ret_data.query("subj==@subj"), on=["subj", "roi", "vertex"])
        del df

        # Bin each vertex by its angular distance from the upcoming target
        # (9 bins spanning the full circle, wrapped to +-180 deg).
        bins = np.array([-180, -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5, 180])
        bin_labels = np.array([180, -135, -90, -45, 0, 45, 90, 135, 180])
        all_data_resid["ang_dist"] = all_data_resid["ang"] - all_data_resid["next_target_ang_brain"]
        all_data_resid["ang_dist"] = all_data_resid["ang_dist"].apply(restrict_ang_range)
        all_data_resid["ang_dist_bin"] = pd.cut(all_data_resid["ang_dist"], bins=bins, right=False,
                                                 labels=bin_labels, ordered=False)
        all_data_resid["ang_dist_bin"] = all_data_resid["ang_dist_bin"].astype(np.float64)

        # Same, for the competitor.
        all_data_resid["comp_dist"] = all_data_resid["ang"] - all_data_resid["next_comp_ang_brain"]
        all_data_resid["comp_dist"] = all_data_resid["comp_dist"].apply(restrict_ang_range)
        all_data_resid["comp_dist_bin"] = pd.cut(all_data_resid["comp_dist"], bins=bins, right=False,
                                                  labels=bin_labels, ordered=False)
        all_data_resid["comp_dist_bin"] = all_data_resid["comp_dist_bin"].astype(np.float64)

        # Label each vertex's condition: at the target, at the competitor, at
        # the "control" (same distance from target as competitor), or none
        # of the above.
        all_data_resid["targ_comp_dist_dir"] = (all_data_resid["ang_dist_bin"] -
                                                 all_data_resid["comp_dist_bin"])
        all_data_resid["targ_comp_dist_dir"] = all_data_resid["targ_comp_dist_dir"].apply(restrict_ang_range)
        all_data_resid.loc[:, "cond"] = "other"

        all_data_resid.loc[all_data_resid["ang_dist_bin"] == 0, "cond"] = "target"
        is_control = all_data_resid["ang_dist_bin"] == -1 * all_data_resid["targ_comp_dist_dir"]
        all_data_resid.loc[is_control, "cond"] = "control"
        all_data_resid.loc[all_data_resid["comp_dist_bin"] == 0, "cond"] = "competitor"

        # Restrict to vertices with reliable pRF fits near the target eccentricity.
        ecc_range = (all_data_resid["ecc"] > .5) & (all_data_resid["ecc"] < 10)
        close_eccen = abs(all_data_resid["ecc"] - config.TARGET_ECCEN) <= all_data_resid["sigma"] * 2
        high_r2 = all_data_resid["r2"] >= .1

        include_data = ecc_range & close_eccen.values & high_r2.values

        data_resid_df = all_data_resid.loc[include_data, :]
        del all_data_resid
        
        # Every run has exactly 56 trials, so trial 56 is always a run's last
        # trial; prepare_sacc_both() builds "next_*" columns via np.roll(-1)
        # per run, which wraps trial 56's "next" values around to trial 1's
        # -- not a real next-trial outcome -- so it's excluded here.
        data_resid_df = data_resid_df.query("trial!=56").dropna(subset=["bold"]).reset_index(drop=True)

        out_name = op.join(deriv_dir, "glm-loc_resids", subj, "ses-01",
                            f"{subj}_ses-01_glm_loc_resids_merged_{sacc}sacc.parquet")
        data_resid_df.to_parquet(out_name)

        del data_resid_df


def curve_averages(subjects, sacc_type, deriv_dir, dataframes_dir):
    """Collapse each subject's merged residuals to one row per trial and concatenate across subjects.

    Reads the ``merged_{sacc_type}sacc.parquet`` files ``agg_data`` wrote,
    z-scores BOLD within each (subject, saccade type, cue validity) group,
    takes the median across vertices within each trial, and writes the
    concatenated ``dataframes/evc_prep_curve_{sacc_type}.csv`` that
    ``figure4.ipynb`` reads (``bold``) and ``stats/fig4_stats.R``/
    ``sfig5_stats.R`` fit their reported mixed model on (``bold``, raw).
    """
    search_data_all = []

    for subj in subjects:
        print(subj)

        data_resid_df = pd.read_parquet(op.join(
            deriv_dir, "glm-loc_resids", subj, "ses-01",
            f"{subj}_ses-01_glm_loc_resids_merged_{sacc_type}sacc.parquet"))

        # Z-score within each specific trial condition. For visualization only.
        data_resid_df["bold_z_all"] = data_resid_df.groupby(
            ["subj", "sacc_type", "cue_validity"])["bold"].transform(lambda x: stats.zscore(x))

        # Collapse to one row per trial (median across the ROI's vertices).
        all_df = data_resid_df.groupby(
            ["subj", "run", "trial", "stim_category", "stim_name", "next_target_ang", "sacc_type",
             "cue_validity", "next_targ_comp_dist", "ang_dist_bin", "comp_dist_bin", "cond",
             "next_warp_resp", "next_comp_resp"]
        ).median(numeric_only=True)[["bold", "bold_z_all"]].reset_index()

        search_data_all.append(all_df)

    search_data_all = pd.concat(search_data_all).reset_index(drop=True)
    search_data_all.to_csv(op.join(dataframes_dir, f"evc_prep_curve_{sacc_type}.csv"), index=False)


def main():
    """CLI entry point: run either the ``agg_data`` or ``curve_averages`` step for ``--subjects``."""
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["agg_data", "curve_averages"])
    parser.add_argument("--sacc", choices=["first", "last"], required=True)
    parser.add_argument("--subjects", default=None,
                         help="Comma-separated subject list, e.g. sub-01 for a 1-subject test. "
                              "Defaults to the full cohort (mgvs_release.config.SUBJECTS).")
    parser.add_argument("--bids-dir", default=config.BIDS_DIR)
    parser.add_argument("--deriv-dir", default=config.DERIV_DIR)
    parser.add_argument("--dataframes-dir", default="dataframes")
    args = parser.parse_args()

    subjects = args.subjects.split(",") if args.subjects else config.SUBJECTS

    if args.step == "agg_data":
        roi_masks = load_roi_masks_mgz(args.deriv_dir, subjects)
        ret_data = load_retinotopy_data(args.deriv_dir, subjects, roi_masks)
        sacc_both = prepare_sacc_both(args.bids_dir, subjects, args.dataframes_dir)
        agg_data(subjects, args.sacc, args.deriv_dir, sacc_both, ret_data)
    elif args.step == "curve_averages":
        curve_averages(subjects, args.sacc, args.deriv_dir, args.dataframes_dir)


if __name__ == "__main__":
    main()
