"""Search-task preparatory-effect prep.

Builds the per-trial "next saccade" dataframe: each trial's row is
attached to the saccade outcome (e.g. ``next_targ_resp``,
``next_sacc_targ_err``) of the trial that *follows* it, so callers can
test whether something about a trial (a pattern-similarity rank, BOLD
residuals, etc.) predicts search performance on the next trial (a
preparatory effect). Shared by ``figures/figure3.ipynb`` and
``precompute/fig4_glm_resids_aggregate.py``.
"""
import os.path as op

import numpy as np
import pandas as pd

from mgvs_release.bids import load_events


def prepare_sacc_both(bids_dir, dataframes_dir, subjects, roll_cols, strict_err_thresh=None):
    """Rebuild the ``sacc_both`` per-trial dataframe.

    Merges each subject's first- and last-saccade behavioral data
    (``dataframes/search_data_all.csv``) against the BIDS search-task design
    to reindex both onto a common (subject, run, trial) index, then adds a
    "next saccade" column for each name in `roll_cols` (that trial's value,
    shifted via ``np.roll`` to align with the *following* trial under a 
    ``next_`` prefix.

    Every run has exactly 56 trials, so trial 56 is always a run's last
    trial; the roll wraps trial 56's "next" value around to trial 1's -- not
    a real next-trial outcome. Callers that use the rolled columns need to
    exclude ``trial==56``.

    `strict_err_thresh`: if given, also adds ``next_targ_resp_strict``, a
    stricter version of ``next_targ_resp``
    """
    search_design = []
    for s in subjects:
        df = load_events(bids_dir, s, "search")
        df = df.assign(subject=s)
        search_design.append(df)
    search_design = pd.concat(search_design).reset_index()

    search_data = pd.read_csv(op.join(dataframes_dir, "search_data_all.csv"))

    sf = search_data.query("sacc_first==True").reset_index(drop=True)
    sf = sf.set_index(["subject", "run", "trial"])
    sf = sf.reindex(search_design.set_index(["subject", "run", "trial"]).index, axis="index").reset_index()
    sf["sacc_type"] = "first"

    sl = search_data.query("sacc_last==True").reset_index(drop=True)
    sl = sl.set_index(["subject", "run", "trial"])
    sl = sl.reindex(search_design.set_index(["subject", "run", "trial"]).index, axis="index").reset_index()
    sl["sacc_type"] = "last"

    sacc_ = pd.concat([sf, sl])

    sacc_both = []
    for (s, r, st), df in sacc_.groupby(["subject", "run", "sacc_type"]):
        df = df.copy()
        for c in roll_cols:
            df["next_" + c] = np.roll(df[c], -1).copy().astype(np.float64)
        sacc_both.append(df)
    sacc_both = pd.concat(sacc_both).reset_index(drop=True)

    if strict_err_thresh is not None:
        sacc_both["next_targ_resp_strict"] = sacc_both["next_sacc_targ_err"] < strict_err_thresh

    return sacc_both
