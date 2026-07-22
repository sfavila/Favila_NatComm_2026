"""Standalone precompute script for the per-stimulus scene GLM.

For each subject, fits one first-level GLM over all search-task runs (one
regressor per unique scene image shown) and saves a per-stimulus contrast
t-map. These are the maps ``figures/figure3.ipynb``'s pattern-similarity and
univariate analyses (Fig. 3B/C, Supp. Figs. 3-4) read back from
``<deriv_dir>/glm-scene_nii/sub-NN/ses-01/<stim>.nii.gz``.

Usage::

    python precompute/glm_scene_nii.py --subjects sub-01
    python precompute/glm_scene_nii.py   # full cohort
"""
import argparse
import os
import os.path as op

import nibabel as nib
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces import fmriprep

from mgvs_release import config
from mgvs_release.bids import load_events

T_R = 1.5


def run_subject(bids_dir, deriv_dir, subj):
    """Fit the per-stimulus GLM for one subject and write its contrast maps.

    Writes one ``<stim>.nii.gz`` t-map per unique scene stimulus shown to
    ``subj``, to ``<deriv_dir>/glm-scene_nii/<subj>/ses-01/``.
    """
    print(subj)

    sbehav = load_events(bids_dir, subj, "search")
    rnums = np.unique(sbehav["run"].values)
    if subj == "sub-09":
        # sub-09's run 6 is excluded for excessive motion
        rnums = rnums[rnums != 6]
        sbehav = sbehav.query("run!=6")

    preproc_dir = op.join(deriv_dir, "fmriprep", subj, "ses-01", "func")
    brain_mask = op.join(preproc_dir, f"{subj}_ses-01_task-search_run-1_space-T1w_desc-brain_mask.nii.gz")

    # Build per-run event tables (one row per stimulus) and matching BOLD
    # timeseries + motion confounds, one regressor per unique stimulus shown.
    timeseries, events, confounds = [], [], []
    for r in rnums:
        ev = sbehav.query("run==@r")[["stim_name", "onset", "stim_dur"]]
        ev = ev.rename(columns={"stim_name": "trial_type", "stim_dur": "duration"})
        events.append(ev)

        tsfile = op.join(preproc_dir, f"{subj}_ses-01_task-search_run-{r}_space-T1w_desc-preproc_bold.nii.gz")
        timeseries.append(nib.load(tsfile))

        c = fmriprep.load_confounds(tsfile, strategy=["motion"], motion="basic", scrub=0, fd_threshold=1.0)
        confounds.append(c[0])

    first_level_model = FirstLevelModel(T_R, slice_time_ref=T_R / 2, mask_img=brain_mask, n_jobs=-2)
    first_level_model = first_level_model.fit(timeseries, events=events, confounds=confounds)
    design_matrix = first_level_model.design_matrices_

    # One identity contrast per design-matrix column (i.e. per regressor), per run.
    contrast_matrix = [np.eye(d.shape[1]) for d in design_matrix]
    contrasts = [{column: contrast_matrix[i][j] for j, column in enumerate(design_matrix[i].columns)}
                 for i in np.arange(len(rnums))]

    mod_dir = op.join(deriv_dir, "glm-scene_nii", subj, "ses-01")
    os.makedirs(mod_dir, exist_ok=True)

    # Write one t-map per unique stimulus, combining its contrast across every run it appeared in.
    for s in sbehav["stim_name"].unique():
        out_name = "castlealt" if s == "castle-alt" else s
        # nilearn's design-matrix column sanitization for "castle-alt" (the
        # only condition name in this dataset with a special character)
        # differs by version -- try the literal name first, falling back to
        # the sanitized one.
        lookup = s if any(s in c for c in contrasts) else out_name
        cons = [contrasts[i][lookup] if lookup in contrasts[i] else np.zeros(design_matrix[i].shape[1])
                for i in np.arange(len(rnums))]
        tmap = first_level_model.compute_contrast(cons)
        nib.save(tmap, op.join(mod_dir, out_name + ".nii.gz"))


def main():
    """CLI entry point: fit and write per-stimulus GLM contrast maps for each subject in ``--subjects``."""
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
