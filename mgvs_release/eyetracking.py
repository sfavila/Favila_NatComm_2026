"""Shared eye-tracking helpers: coordinate transforms and raw saccade parsing.
"""
import math
import os.path as op
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat


def ang_to_xy(ang, eccen):
    x = math.cos(np.deg2rad(ang)) * eccen
    y = math.sin(np.deg2rad(ang)) * eccen
    return x, y


def xy_to_ang(x, y):
    return np.rad2deg(np.arctan2(y, x)) % 360


def xy_to_eccen(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def ang_dist(x1, y1, x2, y2):
    point1 = (x1, y1)
    point2 = (x2, y2)
    # NaN coordinates (e.g. a trial with no scoreable response) 
    # is legitimate.
    with np.errstate(invalid='ignore'):
        det = np.linalg.det([point1, point2])
    dot = np.dot(point1, point2)
    dist = np.arctan2(det, dot)
    return np.abs(dist * 180 / math.pi)


def euclidian_dist(x1, y1, x2, y2):
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))


def _load_saccade_run(sacc_file, pre_file, subj, run, epoch):
    """Parse one subject/run's .mat saccade file, filtered to `epoch`."""
    s = loadmat(sacc_file, struct_as_record=False, squeeze_me=True)['ii_sacc']
    sdf = pd.DataFrame(dict(
        subject=subj,
        run=run,
        trial=s.trial_start,
        timestamp_start=s.t[:, 0],
        timestamp_end=s.t[:, 1],
        epoch=s.epoch_start,
        x_start=s.X_start,
        y_start=s.Y_start,
        x_end=s.X_end,
        y_end=s.Y_end,
        # needed to match output/pass tests.
        x_trace=[np.atleast_1d(t).reshape(-1, 1) for t in s.X_trace],
        y_trace=[np.atleast_1d(t).reshape(-1, 1) for t in s.Y_trace],
    ))

    # extract timestamps for trial onsets from eyetracker file and
    # calculate saccade onset
    allxdat = loadmat(pre_file, struct_as_record=False, squeeze_me=True)['ii_data'].XDAT
    xdat_delta = allxdat - np.roll(allxdat, 1)
    et_onsets = np.squeeze(np.argwhere(np.logical_and(xdat_delta == 1, allxdat == 2))) / 1000
    sdf = sdf.merge(pd.DataFrame(dict(trial=np.arange(len(et_onsets)) + 1, et_stim_onset=et_onsets)))
    sdf['sacc_onset'] = sdf['timestamp_start'] - sdf['et_stim_onset']
    sdf['sacc_offset'] = sdf['timestamp_end'] - sdf['et_stim_onset']

    sdf = sdf.query("epoch==@epoch").reset_index(drop=True)
    sdf = sdf.drop_duplicates(subset=['trial', 'x_start', 'x_end', 'y_start', 'y_end',
                                       'timestamp_start', 'timestamp_end'])
    return sdf


def code_saccade_order(sdf):
    """Filter to large saccades and label first/last/count per trial."""
    sdf = sdf.copy()
    sdf['end_dist'] = sdf.apply(lambda x: np.sqrt(x['x_end'] ** 2 + x['y_end'] ** 2), axis=1)
    sdf = sdf.query("end_dist >= 1")
    if len(sdf) == 0:
        return sdf

    trial = sdf['trial'].values
    sacc_first = np.full(trial.shape, False, dtype='bool')
    sacc_last = np.full(trial.shape, False, dtype='bool')
    first_ind = np.argwhere(trial - np.roll(trial, 1) > 0)
    sacc_first[first_ind] = True
    sacc_last[first_ind - 1] = True

    snum = pd.DataFrame(dict(trial=trial, sacc_num=trial)).groupby('trial').count()['sacc_num']
    tot_sacc = np.concatenate([np.repeat(s, s) for s in snum])
    sacc_num = np.concatenate([np.arange(s) + 1 for s in snum])

    sdf = sdf.merge(pd.DataFrame(dict(timestamp_start=sdf['timestamp_start'],
                                       sacc_first=sacc_first, sacc_last=sacc_last,
                                       tot_sacc=tot_sacc, sacc_num=sacc_num)))
    return sdf


def parse_response_saccades(deriv_dir, subjects, task, epoch, code_order=True):
    """Parse response-period saccades for every subject/run of a task.

    `task` selects the .mat file prefix (e.g. 'learn' or 'search') and
    `epoch` selects which task epoch counts as the response period.
    """
    sacc = []
    for subj in subjects:
        sacc_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                          f"{subj}_*task-{task}_*desc-preprocsacc_eyetrack.mat")))
        pre_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                         f"{subj}_*task-{task}_*desc-preproc_eyetrack.mat")))

        for f_i, f in enumerate(sacc_files):
            run = int(re.search(r'run-(\d+)', op.basename(f)).group(1))
            sdf = _load_saccade_run(f, pre_files[f_i], subj, run, epoch)
            if code_order:
                sdf = code_saccade_order(sdf)
            if len(sdf) > 0:
                sacc.append(sdf)
    return pd.concat(sacc).reset_index(drop=True)


def parse_fixation_saccades(deriv_dir, subjects, task, epoch):
    """Parse saccades made during a fixation-enforced epoch (e.g. the ITI)."""
    fix_break = []
    for subj in subjects:
        sacc_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                          f"{subj}_*task-{task}_*desc-preprocsacc_eyetrack.mat")))
        pre_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                         f"{subj}_*task-{task}_*desc-preproc_eyetrack.mat")))

        for f_i, f in enumerate(sacc_files):
            run = int(re.search(r'run-(\d+)', op.basename(f)).group(1))
            sdf = _load_saccade_run(f, pre_files[f_i], subj, run, epoch)
            sdf['x_start_abs'] = abs(sdf['x_start'])
            sdf['y_start_abs'] = abs(sdf['y_start'])
            sdf['end_dist'] = sdf.apply(lambda x: np.sqrt(x['x_end'] ** 2 + x['y_end'] ** 2), axis=1)
            fix_break.append(sdf)
    return pd.concat(fix_break).reset_index(drop=True)


# --- QC metrics reported in the manuscript's Methods text  ---

def usable_sample_qc(deriv_dir, subjects):
    """Percent of non-NaN gaze samples per task epoch, from preproc .mat files."""
    records = []

    for subj in subjects:
        pre_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                         f"{subj}_*task-search_*desc-preproc_eyetrack.mat")))

        for f in pre_files:
            dat = loadmat(f, struct_as_record=False, squeeze_me=True)['ii_data']

            x = dat.X
            y = dat.Y
            epoch = dat.XDAT

            valid = ~np.isnan(x) & ~np.isnan(y)

            for e in np.unique(epoch):
                idx = epoch == e
                percent_valid = valid[idx].mean() * 100
                records.append({"subject": subj, "epoch": e, "percent_valid": percent_valid})

    qc = pd.DataFrame(records)
    qc_subj = qc.groupby(["subject", "epoch"])["percent_valid"].mean().reset_index()
    return qc_subj.groupby("epoch")["percent_valid"].agg(["mean", "std", "min", "max"])


def iti_fixation_precision(deriv_dir, subjects):
    """Per-subject spatial precision (std of gaze position) during ITI fixation."""
    precision_records = []

    for subj in subjects:
        pre_files = sorted(glob(op.join(deriv_dir, "eyetracking-preproc", subj, "ses-*",
                                         f"{subj}_*task-search_*desc-preproc_eyetrack.mat")))

        for run_i, f in enumerate(pre_files):
            dat = loadmat(f, struct_as_record=False, squeeze_me=True)['ii_data']

            x = dat.X
            y = dat.Y
            epoch = dat.XDAT

            fix_idx = epoch == 1
            x_fix = x[fix_idx]
            y_fix = y[fix_idx]

            precision = np.nanmean([np.nanstd(x_fix), np.nanstd(y_fix)])
            precision_records.append({"subject": subj, "run": run_i + 1, "precision": precision})

    precision_df = pd.DataFrame(precision_records)
    return precision_df.groupby("subject")["precision"].mean().reset_index()


def parse_calibration_qc(bids_dir, subjects):
    """Per-subject Eyelink calibration/validation QC, parsed from .asc message logs."""
    rows = []

    for subj in subjects:
        asc_files = sorted(glob(op.join(bids_dir, "sourcedata", "eyetracking", subj,
                                        f"{subj}_ses-*_task-search_run-*_eyetrack.asc")))

        for f in asc_files:
            basename = op.basename(f)
            match = re.search(r'run-(\d+)', basename)
            run = int(match.group(1)) if match else None

            cal_msgs = []
            val_msgs = []

            with open(f) as file:
                for line in file:
                    if line.startswith("MSG"):
                        if "!CAL" in line:
                            cal_msgs.append(line.strip())
                        if "VALIDATION" in line and "deg" in line:
                            val_msgs.append(line.strip())

            cal_performed = True if cal_msgs else False

            if cal_msgs:
                res_match = re.search(r'Resolution \(upd\).*X=([\d.]+), Y=([\d.]+)', " ".join(cal_msgs))
                cal_precision = ((float(res_match.group(1)) + float(res_match.group(2))) / 2
                                 if res_match else None)

                fixup_vals = re.findall(r'Quadrant fixup\[\d\]\s*=\s*([\d.]+),([\d.]+)', " ".join(cal_msgs))
                max_quadrant_fixup = (max(max(float(x), float(y)) for x, y in fixup_vals)
                                      if fixup_vals else None)

                gain_match = re.search(r'Gain Change Proportion: X:\s*([\d.]+) Y:\s*([\d.]+)',
                                        " ".join(cal_msgs))
                gain_change = ((float(gain_match.group(1)) + float(gain_match.group(2))) / 2
                               if gain_match else None)

                cal_success = any("GOOD" in line and "CALIBRATION" in line for line in cal_msgs)
            else:
                cal_precision = max_quadrant_fixup = gain_change = None
                cal_success = False

            if val_msgs:
                final_val = val_msgs[-1]
                val_match = re.search(r'(\d+\.\d+)\s+deg', final_val)
                val_precision = float(val_match.group(1)) if val_match else None
                val_performed = True
            else:
                val_precision = None
                val_performed = False

            rows.append(dict(subject=subj, run=run, cal_performed=cal_performed, cal_success=cal_success,
                              cal_precision=cal_precision, max_quadrant_fixup=max_quadrant_fixup,
                              gain_change=gain_change, val_performed=val_performed,
                              val_precision=val_precision))

    cal_df = pd.DataFrame(rows)

    summary = cal_df.groupby("subject").agg(
        n_runs=("run", "count"),
        n_cal_performed=("cal_performed", "sum"),
        n_cal_success=("cal_success", "sum"),
        n_val_performed=("val_performed", "sum"),
        mean_cal_precision=("cal_precision", "mean"),
        sd_cal_precision=("cal_precision", "std"),
        mean_fixup=("max_quadrant_fixup", "mean"),
        sd_fixup=("max_quadrant_fixup", "std"),
        mean_gain=("gain_change", "mean"),
        sd_gain=("gain_change", "std"),
        mean_val_precision=("val_precision", "mean"),
    ).reset_index()

    summary["pct_cal_performed"] = 100 * summary["n_cal_performed"] / summary["n_runs"]
    summary["pct_val_performed"] = 100 * summary["n_val_performed"] / summary["n_runs"]
    return summary
