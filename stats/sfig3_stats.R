# Relationship between hippocampal pattern similarity and behavior
#
# Run from the repo root: Rscript stats/sfig3_stats.R

library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(parameters)

options(scipen = 999)

# Read in data
df = read.csv("dataframes/pattern_sim_behav_data.csv")
df$subj <- as.factor(df$subj)

hipp_df = df[df$roi=='hipp', ]
hipp_df = hipp_df[hipp_df$sacc_type=='first', ]

# Continuous version of analysis reported in Supplementary Fig 3D

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(next_sacc_targ_err ~ corr_z_norm + (1 + corr_z_norm | subj),
#                data=hipp_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")

# Reduced model reported in manuscript.
m_reduced <- lmer(next_sacc_targ_err ~ corr_z_norm + (1 | subj),
                  data=hipp_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")


# Relationship between pattern similarity and number of saccades made.
# Reported in manuscript.
m_full <- lmer(tot_sacc ~ corr_z_norm + (1 + corr_z_norm | subj),
               data=hipp_df, REML=T)
summary(m_full)
model_parameters(m_full, ci_method = "satterthwaite")
