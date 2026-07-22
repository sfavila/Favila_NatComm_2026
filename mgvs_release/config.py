"""Central configuration: paths, cohort membership, and analysis parameters
shared across the figure notebooks and precompute scripts.
"""
import os

BIDS_DIR = os.environ.get("MGVS_BIDS_DIR", "/home/sef2177/MGVS_fmri_BIDS")
DERIV_DIR = os.environ.get("MGVS_DERIV_DIR", os.path.join(BIDS_DIR, "derivatives"))

# Optional: a directory of font files. If unset/missing, matplotlib silently falls
# back to its default system font. Most users won't have this directory and 
# don't need to set it.
FONT_DIR = os.environ.get("MGVS_FONT_DIR", "/home/sef2177/fonts/")

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06",
    "sub-07", "sub-08", "sub-09", "sub-10", "sub-11",
    "sub-12", "sub-13", "sub-14", "sub-15", "sub-16",
    "sub-17", "sub-18", "sub-19", "sub-20", "sub-21",
    "sub-22", "sub-23", "sub-25", "sub-26", "sub-28", 
    "sub-29", "sub-31", "sub-32", "sub-33", "sub-34", 
    "sub-35", "sub-36",
]

TARGET_ECCEN = 3.3
