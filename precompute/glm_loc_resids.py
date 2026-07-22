"""Standalone precompute script for the localizer-GLM residual timeseries.

For each subject, fits a first-level GLM (regressors: search-task cue
condition) over V1/V2/V3 surface vertices and extracts each trial's
within-ROI-z-scored residual BOLD in the ITI window. Writes
``<deriv_dir>/glm-loc_resids/sub-NN/ses-01/sub-NN_ses-01_glm_loc_resids_winroizscore.parquet``,
exactly the input ``precompute/fig4_glm_resids_aggregate.py`` expects (see
that script's ``agg_data``).

Usage::

    python precompute/glm_loc_resids.py --subjects sub-01
    python precompute/glm_loc_resids.py   # full cohort
"""
import argparse
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces import fmriprep
from scipy import stats

from mgvs_release import config
from mgvs_release.bids import load_events

T_R = 1.5
HRF_DELAY = 4  # TRs
MIN_ITI = 3  # TRs
ROIS = ["V1", "V2", "V3"]


def load_roi_masks_mgz(deriv_dir, subj, rois=None):
    """Load a subject's V1/V2/V3 masks as bilateral boolean vertex arrays.

    Reads the ``.mgz`` (surface-space) mask variant, as opposed to
    ``mgvs_release.rois.load_roi_masks``'s volumetric ``.nii.gz`` files.
    """
    rois = rois if rois is not None else ROIS
    roi_path = op.join(deriv_dir, "rois", subj, "ses-01")
    masks = {}
    for r in rois:
        lh_r = nib.load(op.join(roi_path, f"lh.{r}_hand.mgz")).get_fdata().astype(bool)
        rh_r = nib.load(op.join(roi_path, f"rh.{r}_hand.mgz")).get_fdata().astype(bool)
        masks[r] = np.concatenate([np.squeeze(lh_r), np.squeeze(rh_r)])
    return masks


def run_subject(bids_dir, deriv_dir, subj):
    """Fit the localizer GLM for one subject and write its residual timeseries.

    Writes one long-format Parquet file of per-trial, per-vertex,
    within-ROI-z-scored residual BOLD to
    ``<deriv_dir>/glm-loc_resids/<subj>/ses-01/``.
    """
    print(subj)

    sbehav = load_events(bids_dir, subj, "search")
    rnums = np.unique(sbehav["run"].values)
    if subj == "sub-09":
        # sub-09's run 6 is excluded for excessive motion
        rnums = rnums[rnums != 6]
        sbehav = sbehav.query("run!=6")

    # Restrict the GLM to visual cortex: union of V1/V2/V3 vertices.
    roi_masks = load_roi_masks_mgz(deriv_dir, subj)
    all_vis = roi_masks["V1"] | roi_masks["V2"] | roi_masks["V3"]
    # int32, not int64: nibabel rejects int64 image data outright.
    mask = nib.Nifti2Image(np.expand_dims(all_vis.astype(np.int32), [1, 2]), affine=np.eye(4))

    preproc_dir = op.join(deriv_dir, "fmriprep", subj, "ses-01", "func")

    # Build per-run event dataframe, concatenated L/R surface
    # timeseries, and motion confounds.
    timeseries, events, confounds = [], [], []
    for r in rnums:
        ev = sbehav.query("run==@r")[["cue_cond", "onset", "stim_dur"]]
        ev = ev.rename(columns={"cue_cond": "trial_type", "stim_dur": "duration"})
        events.append(ev)

        tsfile_ref = op.join(preproc_dir,
                              f"{subj}_ses-01_task-search_run-{r}_space-T1w_desc-preproc_bold.nii.gz")
        tsfile_l = op.join(preproc_dir,
                            f"{subj}_ses-01_task-search_run-{r}_hemi-L_space-fsnative_bold.func.gii")
        tsfile_r = op.join(preproc_dir,
                            f"{subj}_ses-01_task-search_run-{r}_hemi-R_space-fsnative_bold.func.gii")

        ts = np.concatenate([nib.load(tsfile_l).agg_data(), nib.load(tsfile_r).agg_data()])
        ts = np.expand_dims(ts, [1, 2])
        timeseries.append(nib.Nifti2Image(ts, affine=np.eye(4)))

        c = fmriprep.load_confounds(tsfile_ref, strategy=["motion", "scrub"],
                                     motion="basic", scrub=0, fd_threshold=1.0)
        confounds.append(c[0])

    # Only the residuals are used downstream -- the fitted betas themselves
    # aren't read by anything.
    first_level_model = FirstLevelModel(T_R, slice_time_ref=T_R / 2, mask_img=mask, minimize_memory=False)
    first_level_model = first_level_model.fit(timeseries, events=events, confounds=confounds)
    resids = first_level_model.residuals

    # For each run and ROI: z-score this run's residuals (first across time
    # within each vertex, then across vertices within each timepoint), then
    # average each trial's HRF-delayed ITI window (see HRF_DELAY/MIN_ITI
    # above) into a single per-trial, per-vertex value.
    subj_df = []
    for r_i, run in enumerate(rnums):
        ev = sbehav.query("run==@run")

        for r in ROIS:
            verts = np.squeeze(np.argwhere(roi_masks[r]))

            bi_ts = np.squeeze(resids[r_i].get_fdata()[roi_masks[r]])
            bi_ts = stats.zscore(bi_ts, axis=1)
            bi_ts = stats.zscore(bi_ts, axis=0, nan_policy="omit")

            start_iti_tr = (ev["onset"].values / T_R).astype(int) + HRF_DELAY
            end_iti_tr = (ev["onset"].values / T_R).astype(int) + MIN_ITI + HRF_DELAY

            trial_ts = [np.mean(bi_ts[:, s:e], axis=1) for (s, e) in zip(start_iti_tr, end_iti_tr)]

            for t_i, t in enumerate(trial_ts):
                rdf = pd.DataFrame({"bold": t, "roi": r, "subj": subj, "vertex": verts,
                                     "run": run, "trial": t_i + 1})
                subj_df.append(rdf)
    subj_df = pd.concat(subj_df).reset_index(drop=True)

    mod_dir = op.join(deriv_dir, "glm-loc_resids", subj, "ses-01")
    os.makedirs(mod_dir, exist_ok=True)
    subj_df.to_parquet(op.join(mod_dir, f"{subj}_ses-01_glm_loc_resids_winroizscore.parquet"))


def main():
    """CLI entry point: fit and write residual timeseries for each subject in ``--subjects``."""
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", default=None,
                         help="Comma-separated subject list, e.g. sub-01 for a 1-subject test. "
                              "Defaults to the full cohort (mgvs_release.config.SUBJECTS).")
    parser.add_argument("--bids-dir", default=config.BIDS_DIR)
    parser.add_argument("--deriv-dir", default=config.DERIV_DIR)
    args = parser.parse_args()

    subjects = args.subjects.split(",") if args.subjects else config.SUBJECTS
    for subj in subjects:
        run_subject(args.bids_dir, args.deriv_dir, subj)


if __name__ == "__main__":
    main()
