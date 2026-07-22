# Relationship between hippocampal pattern similarity and EVC preparatory activity
#
# Run from the repo root: Rscript stats/fig5_stats.R

library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(parameters)

options(scipen = 999)

# Read in data
df = read.csv("dataframes/hipp_rank_corr.csv") # all valid trials
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs

pairs45_df = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x
# rank bin (low vs high pattern similarity). Reported in Figure 5C.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)
#
# emm <- emmeans(m_full, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
# pairs(emm, adjust = "none")
# confint(pairs(emm, adjust = "none"))

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (1 | subj),
                  data=pairs45_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")
confint(pairs(emm, adjust = "none"))

# Other pair distances (90/135/180 degrees). Reported in manuscript.

pairs90_df = df[df$next_targ_comp_dist==90, ]
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs90_df, REML=T)
# anova(m_full, type = 3)
m_reduced <- lmer(bold ~ cond * rank_bin + (1 + rank_bin | subj),
               data=pairs90_df, REML=T)
anova(m_reduced, type = 3)

pairs135_df = df[df$next_targ_comp_dist==135, ]
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs135_df, REML=T)
# anova(m_full, type = 3)
m_reduced <- lmer(bold ~ cond * rank_bin + (1 + rank_bin | subj),
                  data=pairs135_df, REML=T)
anova(m_reduced, type = 3)

pairs180_df = df[df$next_targ_comp_dist==180, ]
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs180_df, REML=T)
# anova(m_full, type = 3)
m_reduced <- lmer(bold ~ cond * rank_bin + (1 + rank_bin | subj),
                  data=pairs180_df, REML=T)
anova(m_reduced, type = 3)

pairs45_df = df[df$next_targ_comp_dist==45, ]
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# anova(m_full, type = 3)
m_reduced <- lmer(bold ~ cond * rank_bin + (1 | subj),
                  data=pairs45_df, REML=T)
anova(m_reduced, type = 3)

# Three-way interaction: pair distance (45, 90, 135, 180) x condition
# (targ, comp, other) x rank bin (low vs high pattern similarity)
#
# Control condition dropped -- some distances (e.g. 180 degrees) can't
# define a control location, so including it leaves non-estimable cells.
# Reported in manuscript.

# m_full <- lmer(bold ~ next_targ_comp_dist * cond * rank_bin + (rank_bin | subj), data=df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
df_drop = df[df$cond!='control', ]
m_reduced <- lmer(bold ~ next_targ_comp_dist * cond * rank_bin + (rank_bin | subj), data=df_drop, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)
