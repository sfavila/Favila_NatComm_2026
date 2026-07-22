"""Shared statistical reporting helpers.
"""
import os
os.environ.setdefault("OUTDATED_IGNORE", "1")

import numpy as np
import pandas as pd
import pingouin as pg


def paired_ttest(x, y):
    """Paired t-test reporting T, dof, p, Cohen's dz, and 95% CI.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return pg.ttest(x, y, paired=True)


def print_dist_stats(resp_by_dist_long):
    """Paired t-tests (target vs. competitor vs. control saccade probability)
    at each target-competitor distance (45/90/135 degrees).
    """
    with pd.option_context("display.width", 200):
        for d in (45, 90, 135):
            targ = resp_by_dist_long.query("targ_comp_dist==@d & resp_type=='targ_resp'")['prob']
            comp = resp_by_dist_long.query("targ_comp_dist==@d & resp_type=='comp_resp'")['prob']
            cont = resp_by_dist_long.query("targ_comp_dist==@d & resp_type=='cont_resp'")['prob']

            print('---', d, 'degree pairs ---')
            print('target vs competitor')
            print(paired_ttest(targ, comp))
            print('competitor vs control')
            print(paired_ttest(comp, cont))
            print('target vs control')
            print(paired_ttest(targ, cont))
