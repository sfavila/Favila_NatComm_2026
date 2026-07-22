# Relationship between EVC preparatory activity and behavior (last saccade)
#
# Run from the repo root: Rscript stats/fig4_stats.R

library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(parameters)

options(scipen = 999)

# Read in data
df = read.csv("dataframes/evc_prep_curve_last.csv")
df$cond <- as.factor(df$cond)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Valid trials only
valid_df = df[df$cue_validity=='valid', ]
valid_df$behav_outcome <- ifelse(valid_df$next_warp_resp == TRUE, 'success', 'other')
valid_df$behav_outcome[valid_df$next_comp_resp == TRUE] <- 'interference'

# Interaction between pRF location and behavioral outcome. Reported in Figure 4D.
# First-saccade version is stats/sfig5_stats.R.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * behav_outcome + (1 + cond + behav_outcome | subj),
#            data=valid_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)
#
# emm <- emmeans(m_full, ~ cond | behav_outcome, lmer.df = "satterthwaite", lmerTest.limit = 75144)
# pairs(emm, adjust = "none")

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * behav_outcome + (1 | subj),
           data=valid_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | behav_outcome, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")
confint(pairs(emm, adjust = "none"))
