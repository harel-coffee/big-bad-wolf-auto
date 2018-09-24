library(arm)
library(brms)

df = read.csv("defnite.csv")
head(df)

## Compute simple Generalized Linear models
lr.wolf <- glm(wolf ~ timebin + rrh + opening, data=df, family="binomial")
summary(lr.wolf)

# This can't be done, because of quasi-separation problems (cf. the errors)
lr.rrh <- glm(rrh ~ timebin + wolf + opening, data=df, family="binomial")
summary(lr.rrh)

## Compute Bayes Generalized Linear Models according to Gelman et al.
blr.wolf <- bayesglm(wolf ~ timebin + rrh + opening, data=df, family="binomial")
display(blr.wolf)

blr.rrh <- bayesglm(rrh ~ timebin + wolf + opening, data=df, family="binomial")
display(blr.rrh)

# While Gelman's analysis using an EM optimalization overcomes the problem of 
# regular GLMs for quasi separation, it doesn't allow us to check for certain 
# important concepts in Bayesian Statistics, such as the 95% Credible Interval.
# Therefore, we will compute a fullfledged Bayesian GLM with NUTS:
brm.wolf <- brm(wolf ~ timebin + rrh + opening, data=df, family="bernoulli")
summary(brm.wolf)

brm.rrh <- brm(rrh ~ timebin + wolf + opening, data=df, family="bernoulli")
summary(brm.rrh)

# TODO: Further analyses and model criticism