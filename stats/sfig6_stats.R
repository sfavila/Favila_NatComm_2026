# Relationship between pattern similarity (or univariate activity) and EVC
# preparatory activity, other ROIs
#
# Run from the repo root: Rscript stats/sfig6_stats.R

library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(parameters)

options(scipen = 999)

### Anterior Hippocampus ###

# Read in data
df = read.csv("dataframes/hipp_anterior_rank_corr.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df_ant = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x rank bin (low vs
# high pattern similarity). Reported in Supplementary Fig 6B.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df_ant, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (rank_bin | subj),
                  data=pairs45_df_ant, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")
confint(pairs(emm, adjust="none"))

### Posterior Hippocampus ###

# Read in data
df = read.csv("dataframes/hipp_posterior_rank_corr.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df_post = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x rank bin (low vs
# high pattern similarity). Reported in Supplementary Fig 6B.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df_post, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (rank_bin | subj),
                  data=pairs45_df_post, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")

# Three-way interaction: hippocampal long-axis position (anterior, posterior)
# x condition (targ, comp, control, other) x rank bin (low vs high pattern
# similarity). Reported in manuscript.

pairs45_df_ant['roi'] = 'ant_hipp'
pairs45_df_post['roi'] = 'post_hipp'
pairs45_df_all <- rbind(pairs45_df_ant, pairs45_df_post)

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ roi * cond * rank_bin + (1 + roi + cond + rank_bin | subj), data=pairs45_df_all, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ roi * cond * rank_bin + (1 + rank_bin | subj), data=pairs45_df_all, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

### PPA ###

# Read in data
df = read.csv("dataframes/CoS-places_rank_corr.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x rank bin (low vs high pattern
# similarity). Reported in Supplementary Fig 6C.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (1 | subj),
                  data=pairs45_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")

### RSC ###

# Read in data
df = read.csv("dataframes/rsc_rank_corr.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x rank bin (low vs high pattern
# similarity). Reported in Supplementary Fig 6C.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (rank_bin | subj),
                  data=pairs45_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")
confint(pairs(emm, adjust = "none"))

### PHC ###

# Read in data
df = read.csv("dataframes/phc_rank_corr.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x rank bin (low vs high pattern
# similarity). Reported in Supplementary Fig 6C.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (rank_bin | subj),
                  data=pairs45_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")

### DAN (univariate) ###

# Read in data
df = read.csv("dataframes/DAN_rank_univariate.csv")
df$cond <- as.factor(df$cond)
df$rank_bin <- as.factor(df$rank_bin)
df$next_targ_comp_dist <- as.factor(df$next_targ_comp_dist)
df$subj <- as.factor(df$subj)

# Select 45-degree pairs
pairs45_df = df[df$next_targ_comp_dist==45, ]

# Two-way interaction: condition (targ, comp, control, other) x
# rank bin (low vs high DAN activity). Reported in Supplementary Fig 6C.

# Maximal random-effects structure model. Fit for robustness/reviewer transparency.
# m_full <- lmer(bold ~ cond * rank_bin + (1 + cond + rank_bin | subj),
#                data=pairs45_df, REML=T)
# summary(m_full)
# model_parameters(m_full, ci_method = "satterthwaite")
#
# anova(m_full, type = 3)

# Reduced model reported in manuscript.
m_reduced <- lmer(bold ~ cond * rank_bin + (rank_bin | subj),
                  data=pairs45_df, REML=T)
summary(m_reduced)
model_parameters(m_reduced, ci_method = "satterthwaite")

anova(m_reduced, type = 3)

emm <- emmeans(m_reduced, ~ cond | rank_bin, lmer.df = "satterthwaite", lmerTest.limit = 75144)
pairs(emm, adjust = "none")
