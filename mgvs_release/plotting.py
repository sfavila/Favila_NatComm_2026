"""Shared matplotlib/seaborn style setup and plot builders reused across the
figure notebooks."""
import os.path as op
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

from . import config


# --- General, used across notebooks ---

def save_fig(fig_obj, save_stem):
    """Save `fig_obj` (anything with a `.savefig(path)` method: a Figure,
    a seaborn FacetGrid, a nilearn display) to `save_stem + '.svg'` and an
    identical `save_stem + '.pdf'`.
    """
    fig_obj.savefig(save_stem + '.svg')
    fig_obj.savefig(save_stem + '.pdf')


def setup_style():
    sns.set_context("paper", font_scale=.8)

    # Known-benign warnings from pingouin's/pandas' own internals
    warnings.filterwarnings("ignore", message=r".*DataFrame\.groupby with axis=1 is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*DataFrameGroupBy\.diff with axis=1 is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*divide by zero encountered in scalar divide.*")
    warnings.filterwarnings("ignore", message=r".*DataFrameGroupBy\.apply operated on the grouping columns.*")
    # Swarmplot warning about overlapping points
    warnings.filterwarnings("ignore", message=r".*points cannot be placed.*")

    font_files = font_manager.findSystemFonts(fontpaths=[config.FONT_DIR])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    mpl.rcParams['font.family'] = 'Arial Unicode MS'
    mpl.rcParams['font.size'] = 7.5
    mpl.rcParams['xtick.labelsize'] = 7.5
    mpl.rcParams['ytick.labelsize'] = 7.5
    mpl.rcParams['legend.fontsize'] = 6.5
    mpl.rcParams['legend.title_fontsize'] = 6.5
    mpl.rcParams['axes.titlesize'] = 7.5
    mpl.rcParams['axes.labelsize'] = 7.5


def push_to_back(ax):
    """Sends every artist currently on `ax` behind whatever gets plotted next.
    """
    for artist in ax.get_children():
        artist.set_zorder(-1000)


# --- Fig 3C rank-bin effect (figure3.ipynb, supp_figure3.ipynb, supp_figure4.ipynb) ---

def plot_rank_bin_effect(data, y, color, title, ylabel, ylim, yticks=None,
                          figsize=(2.3, 1.8), save_path=None):
    """Single-panel stripplot+pointplot of a rank-binned (more/less distinct
    pattern-similarity, 2 bins) effect vs. `y`. Used for Fig 3C
    and its supplement counterparts across ROIs and saccade
    types (supp_figure3.ipynb, supp_figure4.ipynb).

    `data` must already be filtered to the roi/sacc_type of interest and
    grouped to one row per (subj, rank_bin), e.g.
    ``df.query(...).groupby(['subj', 'rank_bin']).mean(numeric_only=True).reset_index()``.
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)

    n_subj = data['subj'].nunique()
    sns.stripplot(x='rank_bin', y=y, ax=ax, hue='subj', palette=[color] * n_subj,
                  alpha=.4, s=2, data=data)

    push_to_back(ax)

    sns.pointplot(x='rank_bin', y=y, errorbar='se', ax=ax,
                  color=color, markersize=3.5, linewidth=2,
                  data=data)
    ax.legend_ = None

    ax_kwargs = dict(ylabel=ylabel, ylim=ylim, xlabel='Scene pair pattern similarity',
                      xticks=range(2), xticklabels=['more distinct', 'less distinct'], title=title)
    if yticks is not None:
        ax_kwargs['yticks'] = yticks
    ax.set(**ax_kwargs)

    sns.despine()
    f.tight_layout()
    save_fig(f, op.splitext(save_path)[0])


# --- Supp Fig 2: target-competitor distance (supp_figure2.ipynb) ---

def compute_resp_by_dist(data, sacc_col):
    """Per-subject, per-target-competitor-distance target/competitor/control
    saccade probability, long-format. Used identically for the first- and
    last-saccade sections of supp_figure2.ipynb, differing only in which
    saccade boolean column (`sacc_first`/`sacc_last`) selects the trials.
    """
    resp_by_dist_wide = data.query(
        f"trial!=1 & {sacc_col}==True & target_validity=='valid'"
    ).groupby(['subject', 'targ_comp_dist']).mean(numeric_only=True)[
        ['targ_resp', 'comp_resp', 'cont_resp']].reset_index()
    return resp_by_dist_wide.melt(id_vars=['subject', 'targ_comp_dist'],
                                   value_vars=['targ_resp', 'comp_resp', 'cont_resp'],
                                   var_name='resp_type', value_name='prob')


def plot_dist_error_hist(data, sacc_col, ylabel, ylim, save_path):
    """Directional angular-error histograms for the 45/90/135-degree
    target-competitor pairs, one facet per distance, with target/competitor/
    control reference lines. Used identically for the first- and last-saccade
    sections of supp_figure2.ipynb
    """
    plot_data = data.query(
        f"trial!=1 & {sacc_col}==True & target_validity=='valid' & targ_comp_dist!=180")
    g = sns.displot(x='targ_err_toward', alpha=1, kind='hist',
                     col='targ_comp_dist', height=1.8, aspect=1,
                     color='#74a9cf',
                     data=plot_data)
    g.axes[0][0].axvline(x=0, c="#3fa76cb3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][0].axvline(x=45, c="#e37103b3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][0].axvline(x=-45, c="#aaaaaa", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][0].axvline(x=0, c="#3fa76cb3", ls="--", zorder=-100)
    g.axes[0][0].axvline(x=45, c="#e37103b3", ls="--", zorder=-100)
    g.axes[0][0].axvline(x=-45, c="#aaaaaa", ls="--", zorder=-100)
    g.axes[0][0].set_title('45 degree pairs')
    g.axes[0][0].text(80, 170, 'target', c='#3fa76cb3', fontsize=6.5)
    g.axes[0][0].text(80, 150, 'competitor', c='#e37103b3', fontsize=6.5)
    g.axes[0][0].text(80, 130, 'control', c='#aaaaaa', fontsize=6.5)

    g.axes[0][1].axvline(x=0, c="#3fa76cb3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][1].axvline(x=90, c="#e37103b3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][1].axvline(x=-90, c="#aaaaaa", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][1].axvline(x=0, c="#3fa76cb3", ls="--", zorder=-100)
    g.axes[0][1].axvline(x=90, c="#e37103b3", ls="--", zorder=-100)
    g.axes[0][1].axvline(x=-90, c="#aaaaaa", ls="--", zorder=-100)
    g.axes[0][1].set_title('90 degree pairs')

    g.axes[0][2].axvline(x=0, c="#3fa76cb3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][2].axvline(x=135, c="#e37103b3", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][2].axvline(x=-135, c="#aaaaaa", ls="-", lw=11.5, alpha=.2, zorder=-100)
    g.axes[0][2].axvline(x=0, c="#3fa76cb3", ls="--", zorder=-100, label='target')
    g.axes[0][2].axvline(x=135, c="#e37103b3", ls="--", zorder=-100, label='competitor')
    g.axes[0][2].axvline(x=-135, c="#aaaaaa", ls="--", zorder=-100, label='control')
    g.axes[0][2].set_title('135 degree pairs')

    g.set(xlabel='Directional angular error', ylabel=ylabel, xticks=[-180, -90, 0, 90, 180], ylim=[0, ylim])

    sns.despine()
    plt.gcf().tight_layout(w_pad=1.8)
    save_fig(plt.gcf(), op.splitext(save_path)[0])


def plot_dist_resp_swarm(resp_by_dist_long, ylabel, save_path):
    """Target/competitor/control saccade-probability swarm+mean-line plot,
    one facet per target-competitor distance (45/90/135 degrees). Used
    identically for the first- and last-saccade sections of
    supp_figure2.ipynb.
    """
    pal = ["#3fa76c", "#e37103", "#aaaaaa"]

    g = sns.catplot(x='resp_type', y='prob', col='targ_comp_dist', kind='swarm', height=1.8, aspect=1,
                     size=2.5, alpha=.6, order=['targ_resp', 'comp_resp', 'cont_resp'],
                     palette=pal, hue='resp_type', legend=False,
                     data=resp_by_dist_long.query("targ_comp_dist!=180"))
    g.set(xlabel='', xticks=range(3), xticklabels=['Target', 'Competitor', 'Control'],
          ylabel=ylabel, ylim=[-.05, 1])
    g.set_titles('')

    dists = [(0, 45, '45 degree pairs'), (1, 90, '90 degree pairs'), (2, 135, '135 degree pairs')]
    for ax_i, dist, title in dists:
        pairs = resp_by_dist_long.query(
            "targ_comp_dist==@dist").groupby(['resp_type']).mean(numeric_only=True).reset_index()
        for i, cat in zip(np.arange(3), ['targ_resp', 'comp_resp', 'cont_resp']):
            cat_mean = pairs.query("resp_type == @cat")['prob']
            g.axes[0][ax_i].plot([i - .3, i + .3], [cat_mean, cat_mean], c=pal[i], lw=2)
        g.axes[0][ax_i].set_title(title)

    sns.despine()
    plt.gcf().tight_layout(w_pad=1.8)
    save_fig(plt.gcf(), op.splitext(save_path)[0])


def plot_dist180_error_hist(data, sacc_col, binwidth, ylim, ylabel, save_path):
    """Absolute angular-error histogram for the 180-degree target-competitor
    pairs. Used identically for the first- and last-saccade sections of
    supp_figure2.ipynb.
    """
    plot_data = data.query(
        f"trial!=1 & {sacc_col}==True & target_validity=='valid' & targ_comp_dist==180")
    g = sns.displot(x='sacc_targ_err', alpha=1, kind='hist',
                     col='targ_comp_dist', binwidth=binwidth, height=1.8,
                     color='#74a9cf',
                     data=plot_data)
    g.axes[0][0].axvline(x=12.25, c="#3fa76cb3", ls="-", lw=12, alpha=.2, zorder=-100)
    g.axes[0][0].axvline(x=167.5, c="#e37103b3", ls="-", lw=12, alpha=.2, zorder=-100)
    g.axes[0][0].axvline(x=0, c="#3fa76cb3", ls="--", zorder=-100)
    g.axes[0][0].axvline(x=180, c="#e37103b3", ls="--", zorder=-100)
    g.axes[0][0].set_title('180 degree pairs')

    g.set(xlabel='Absolute angular error', ylabel=ylabel, xticks=[0, 90, 180], ylim=[0, ylim])

    sns.despine()
    plt.gcf().tight_layout()
    save_fig(plt.gcf(), op.splitext(save_path)[0])


def plot_dist180_resp_swarm(resp_by_dist_long, ylabel, save_path):
    """Target/competitor saccade-probability swarm+mean-line plot for the
    180-degree target-competitor pairs. Used identically for the first- and
    last-saccade sections of supp_figure2.ipynb.
    """
    pal = ["#3fa76c", "#e37103", "#aaaaaa"]

    g = sns.catplot(x='resp_type', y='prob', col='targ_comp_dist', kind='swarm', height=1.8,
                     size=2.5, alpha=.6, order=['targ_resp', 'comp_resp'],
                     palette=pal[:2], hue='resp_type', legend=False,
                     data=resp_by_dist_long.query("targ_comp_dist==180 & resp_type!='cont_resp'"))
    g.set(xlabel='', xticks=range(2), xticklabels=['Target', 'Competitor'],
          ylabel=ylabel, ylim=[-.05, 1])
    g.set_titles('')

    pairs_180 = resp_by_dist_long.query(
        "targ_comp_dist==180").groupby(['resp_type']).mean(numeric_only=True).reset_index()
    for i, cat in zip(np.arange(3), ['targ_resp', 'comp_resp']):
        cat_mean = pairs_180.query("resp_type == @cat")['prob']
        g.axes[0][0].plot([i - .3, i + .3], [cat_mean, cat_mean], c=pal[i], lw=2)
    g.axes[0][0].set_title('180 degree pairs')

    sns.despine()
    plt.gcf().tight_layout()
    save_fig(plt.gcf(), op.splitext(save_path)[0])


# --- Supp Fig 5/6: BOLD response by pRF location (supp_figure5.ipynb, supp_figure6.ipynb) ---

COND_ORDER = ["target", "competitor", "control", "other"]
COND_TICKLABELS = ["targ", "comp", "cont", "other"]


def plot_behav_response_panels(data, save_path):
    """3-panel swarm+point plot of EVC BOLD response by pRF location
    (target/competitor/control/other), split by behavioral response type
    (search success / interference error / other error). Used for both the
    last- and first-saccade "individual data points" panels in
    supp_figure5.ipynb
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.2, 2), sharey=False)

    panels = [
        (ax1, "warp", sns.desaturate("#40a86c", .85), "Search Successes",
         [-.1, .1], "EVC ITI BOLD response (z)"),
        (ax2, "competitor", sns.desaturate("#de730e", .85), "Interference Errors", [-.3, .3], ""),
        (ax3, "other", "#888888", "Other Errors", [-.15, .15], ""),
    ]
    for ax, resp_type, color, title, ylim, ylabel in panels:
        panel_data = data.query("cue_validity=='valid' & resp_type==@resp_type")
        sns.swarmplot(x="cond", y="bold_z_all", size=2.5, alpha=.4,
                      edgecolor=".5", linewidth=.2,
                      order=COND_ORDER, zorder=-1, color="#bbbbbb",
                      data=panel_data, ax=ax)
        sns.pointplot(x="cond", y="bold_z_all", errorbar="se", markers="_", markersize=20,
                      linewidth=2.5, linestyle="none",
                      order=COND_ORDER, zorder=2, color=color,
                      err_kws={"alpha": 0.6},
                      data=panel_data, ax=ax)
        ax.set(ylim=ylim, ylabel=ylabel, xlabel="pRF Location", title=title,
               xticks=range(len(COND_TICKLABELS)), xticklabels=COND_TICKLABELS)

    f.tight_layout(w_pad=.5)
    sns.despine()
    save_fig(f, op.splitext(save_path)[0])


def plot_roi_rank_swarm(data, ylim, less_title, more_title, save_path,
                         ylabel_left="EVC ITI BOLD response (z)"):
    """2-panel ("less distinct" / "more distinct" pattern-similarity rank
    bin) swarm+point plot of BOLD response by pRF location. Used once per
    ROI in supp_figure6.ipynb (HIPP, Ant HIPP, Post HIPP, PPA, RSC, PHC,
    DAN)
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.2, 2.2))

    for ax, rank_bin in [(ax1, 2), (ax2, 1)]:
        panel_data = data.query("rank_bin==@rank_bin")
        sns.swarmplot(x="cond", y="bold_z", color="#bbbbbb", size=2.5, alpha=.4,
                      edgecolor=".5", linewidth=.2,
                      order=COND_ORDER, zorder=-1,
                      data=panel_data, ax=ax)
        sns.pointplot(x="cond", y="bold_z", errorbar="se", markers="_", markersize=20,
                      linewidth=2.5, linestyle="none",
                      color="#045a8d", zorder=2,
                      err_kws={"alpha": 0.6},
                      order=COND_ORDER,
                      data=panel_data, ax=ax)

    ax1.set(ylim=ylim, ylabel=ylabel_left, xlabel="pRF location",
            xticks=range(len(COND_TICKLABELS)), xticklabels=COND_TICKLABELS)
    ax2.set(ylim=ylim, ylabel="", yticklabels=[], xlabel="pRF location",
            xticks=range(len(COND_TICKLABELS)), xticklabels=COND_TICKLABELS)
    ax1.set_title(less_title)
    ax2.set_title(more_title)

    sns.despine()
    f.tight_layout(w_pad=4)
    save_fig(f, op.splitext(save_path)[0])
