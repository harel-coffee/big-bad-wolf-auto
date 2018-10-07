library(arm)
library(brms)
require(gridExtra)
library(scales)


df <- read.csv("defnite.csv")
df$rrh = as.factor(df$rrh)
df$wolf = as.factor(df$wolf)
df$opening = as.factor(df$opening)
head(df)

## Compute simple Generalized Linear models
lr_wolf <- glm(wolf ~ timebin, data = df, family = "binomial")
summary(lr_wolf)

# This can't be done, because of quasi-separation problems (cf. the errors)
lr_rrh <- glm(rrh ~ timebin + opening,  data = df, family = "binomial")
summary(lr_rrh)

## Compute Bayes Generalized Linear Models according to Gelman et al.
blr_wolf <- bayesglm(wolf ~ timebin,
                     data = df, family = "binomial")
display(blr_wolf)

blr_rrh <- bayesglm(rrh ~ timebin,
                    data = df, family = "binomial")
display(blr_rrh)

# While Gelman's analysis using an EM optimalization overcomes the problem of 
# regular GLMs for quasi separation, it doesn't allow us to check for certain 
# important concepts in Bayesian Statistics, such as the 95% Credible Interval.
# Therefore, we will compute a fullfledged Bayesian GLM with NUTS:

# We have no real a priori confidence that the estimates will be close to zero. 
# As such a normal distribution with a relatively large scale would be appropriate
# a a prior, or we could use distributions with heavier tails (do deal with potential
# outliers), such as the Student t distribution. Being on the uninformed side, 
# we'll use a weakly-informative prior in the form of a unit student-t prior with 
# 10 degrees of freedom for both models.
brm_wolf <- brm(wolf ~ timebin, data = df,
                prior = c(set_prior("student_t(10,0,1)", class = "b")),
                family = "bernoulli", iter = 10000)
summary(brm_wolf)
plot(brm_wolf)

# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm_wolf, "timebin") < 0)
hypothesis(brm_wolf, "timebin = 0") # alternatively

# From: https://www.r-bloggers.com/diffusion-wiener-model-analysis-with-brms-part-iii-hypothesis-tests-of-parameter-estimates/
# Thus, the resulting value is a probability (i.e., ranging from 0 to 1), 
# with values near zero denoting evidence for a difference, and values 
# near one provide some evidence against a difference. Thus, in contrast 
# to a classical p-value it is a continuous measure of evidence for (when
#  near 0) or against (when near 1) a difference between the parameter 
# estimates. Given its superficial similarity with classical p-values 
# (i.e., low values are seen as evidence for a difference), we could call 
# this it a version of a Bayesian p-value or pB for short. In the present 
# case we could say: The pB value for a difference between speed and 
# accuracy conditions in drift rate across word and non-word stimuli 
# is .13, indicating that the evidence for a difference is at best weak. 

plot(marginal_effects(brm_wolf, "timebin"))

brm_rrh <- brm(rrh ~ timebin, data = df,
               prior = c(set_prior("student_t(10, 0, 1)", class = "b")),
               family = "bernoulli", iter = 10000)
summary(brm_rrh)
plot(brm_rrh)

# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm_rrh, "timebin") < 0)
hypothesis(brm_rrh, "timebin < 0") # alternatively
# And finally, what is the probability that the definiteness of RRH has no effect?

plot(marginal_effects(brm_rrh, "timebin"))

## Analysis of the effect of time on the presence of formulaic openings

brm_opening <- brm(opening ~ timebin, data = df,
                   prior = c(set_prior("student_t(10, 0, 1)", class = "b")),
                   family = "bernoulli", iter = 10000)
summary(brm_opening)
plot(brm_opening)

mean(posterior_samples(brm_opening, "timebin") < 0)
hypothesis(brm_opening, "timebin < 0")

plot(marginal_effects(brm_opening, "timebin"))[[1]] +
    scale_x_discrete(
        name ="Time period", 
        limits = c("1800-1850","1850-1900","1900-1950","1950-2000")) +
    scale_y_continuous(
        name = "Probability of formulaic opening",
        labels = percent,
        limits = c(0, 1)) + 
    theme_bw()

exp(quantile(as.matrix(brm_opening)[,2], probs=c(.5, .025, .975)))


## interaction

brm_rrh_opening <- brm(rrh ~ timebin * opening, data = df,
               prior = c(set_prior("student_t(10, 0, 1)", class = "b")),
               family = "bernoulli", iter = 10000)
summary(brm_rrh_opening)
plot(brm_rrh_opening)

## Bayes model comparison rrh|tijd - rrh|tijd*opening 
