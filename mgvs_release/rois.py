"""Shared ROI-mask loading.

Loads per-subject binary ROI masks from FreeSurfer/retinotopy-derived NIFTI
files in the derivatives tree.
"""
import os.path as op

import nibabel as nib
import numpy as np

ALL_ROIS = [
    "V1_V3", "hipp",
    "hipp_anterior", "hipp_posterior",
    "phc", "rsc", "CoS-places", "DAN",
]

_HAND_DRAWN = ("V1", "V2", "V3")
_BILATERAL_NII = ("phc", "rsc", "hipp_anterior", "hipp_posterior")
_LH_RH_NII = ("hipp", "CoS-places")
_NETWORK_LABELS = {"DAN": 3}


def load_roi_masks(deriv_dir, subjects, rois=ALL_ROIS):
    """Load binary ROI masks for every subject, keyed by 'sub-NN' -> roi name.

    'V1_V3' is a derived key: V1 | V2 | V3, unioned from the individual
    hand-drawn masks (there's no single 'V1_V3' file on disk).
    """
    roi_masks = dict()
    for subj in subjects:
        roi_masks[subj] = dict()
        sub_roi_path = op.join(deriv_dir, "rois", subj, "ses-01")

        if "V1_V3" in rois:
            v1_v3 = None
            for hand_r in _HAND_DRAWN:
                lh_r = nib.load(op.join(sub_roi_path, f"lh.{hand_r}_hand.nii.gz")).get_fdata().astype(bool)
                rh_r = nib.load(op.join(sub_roi_path, f"rh.{hand_r}_hand.nii.gz")).get_fdata().astype(bool)
                bi_r = np.logical_or(lh_r, rh_r)
                v1_v3 = bi_r if v1_v3 is None else np.logical_or(v1_v3, bi_r)
            roi_masks[subj]["V1_V3"] = v1_v3

        for r in rois:
            if r == "V1_V3":
                continue
            if r in _BILATERAL_NII:
                bi_r = nib.load(op.join(sub_roi_path, f"{r}.nii.gz")).get_fdata().astype(bool)
            else:
                if r in _LH_RH_NII:
                    lh_r = nib.load(op.join(sub_roi_path, f"lh.{r}.nii.gz")).get_fdata().astype(bool)
                    rh_r = nib.load(op.join(sub_roi_path, f"rh.{r}.nii.gz")).get_fdata().astype(bool)
                elif r in _NETWORK_LABELS:
                    lh_r = nib.load(op.join(sub_roi_path, "lh.7Networks.nii.gz")).get_fdata()
                    rh_r = nib.load(op.join(sub_roi_path, "rh.7Networks.nii.gz")).get_fdata()
                    label = _NETWORK_LABELS[r]
                    lh_r = lh_r == label
                    rh_r = rh_r == label
                bi_r = np.logical_or(lh_r, rh_r)
            roi_masks[subj][r] = bi_r

    return roi_masks
