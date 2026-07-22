"""BIDS-events adapter layer.

Reads a subject's per-run `events.tsv` files and concatenates them into one
dataframe per subject/task, dropping the `duration`/`trial_type`/`stim_file`
columns BIDS requires. Column names and values otherwise pass through unchanged 
from the BIDS release.
"""
import os.path as op
from glob import glob

import pandas as pd

_BIDS_ONLY_COLUMNS = ("duration", "trial_type", "stim_file")


def load_events(bids_dir, subj, task, ses=None):
    """Concatenate a subject's per-run `events.tsv` for `task` into one df.

    `ses` is only needed if a task name is ambiguous across sessions; by
    default it's globbed (e.g. `task-learn` only ever lives under
    `ses-train`, `task-search` only under `ses-01`).
    """
    ses_glob = ses if ses else "*"
    pattern = op.join(bids_dir, subj, f"ses-{ses_glob}", "*",
                       f"{subj}_ses-{ses_glob}_task-{task}_run-*_events.tsv")
    files = sorted(glob(pattern))

    runs = [_drop_bids_only_columns(pd.read_csv(f, sep="\t", na_values="n/a")) for f in files]
    return pd.concat(runs).reset_index(drop=True)


def _drop_bids_only_columns(df):
    """Drop the `duration`/`trial_type`/`stim_file` columns BIDS adds that the analysis code never uses."""
    return df.drop(columns=[c for c in _BIDS_ONLY_COLUMNS if c in df.columns])
