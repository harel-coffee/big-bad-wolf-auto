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

# We have no real a priori confidence that the estimates will be close to zero. 
# As such a normal distribution with a relatively large scale would be appropriate
# a a prior, or we could use distributions with heavier tails (do deal with potential
# outliers), such as the Student t distribution. Being on the uninformed side, 
# we'll use a unit student-t prior with 10 degrees of freedom for both models.
brm.wolf <- brm(wolf ~ timebin + rrh + opening, data=df, 
                prior=c(set_prior("student_t(10,0,1)", class="b")), 
                family="bernoulli", iter=10000)
summary(brm.wolf)
plot(brm.wolf)
marginal_effects(brm.wolf)

brm.rrh <- brm(rrh ~ timebin + wolf + opening, data=df, 
               prior=c(set_prior("student_t(10,0,1)", class="b")), 
               family="bernoulli", iter=10000)
summary(brm.rrh)
plot(brm.rrh)
marginal_effects(brm.rrh)
