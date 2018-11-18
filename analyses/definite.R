library(brms)
library(lme4)
require(gridExtra)
library(scales)
library(tidyr)
library(tidybayes)
library(cowplot)
library(OneR)
library(effects)
library(car)
library(purrr)
library(data.table)
library(bayesplot)
library(projpred)
library(standardize)
library(sjstats)


df <- read.csv("../data/definite.csv")
df <- df[df$reprint == "no",] # exclude remaining reprints
df$year = df$year_estimated
df$rrh = factor(df$rrh_new, levels=0:1, labels=c("indefinite", "definite"))
df$wolf = factor(df$wolf, levels=0:1, labels=c("indefinite", "definite"))
df$any_picture_wolf = as.factor(df$any_picture_wolf - df$wolf_cover)
df$any_picture_rrh = as.factor(df$any_picture_rrh - df$rrh_cover)
df$opening = as.factor(df$opening)
head(df)

## ////////////////////////////////////////////////////////////////////
## Overimputation based on measurement errors
## ////////////////////////////////////////////////////////////////////

overimpute <- function(df, column, reference, subset=NULL, sd=1) {
    df = copy(df)
    df[subset, column] <- sapply(df[subset, column], partial(rnorm, n=1, sd=sd))
    df = predict(reference, newdata = df, random=F, response=T)
    return(df)
}

sd = 9.140180259205387 # from datereg.py on training data
with_error = df$exact_date == "False"

## ////////////////////////////////////////////////////////////////////
## The wolf
## ////////////////////////////////////////////////////////////////////

regr = standardize(wolf ~ any_picture_wolf + year, data=df, family = "binomial", scale=0.5)

# compute n datasets with slightly different years for estimated years
n_sims = 100
bdf <- lapply(seq_len(n_sims),
              function (x) overimpute(df, "year", regr, with_error, sd=sd))

auto_prior(regr$formula, regr$data, gaussian = F)

brm.wolf <- brm_multiple(regr$formula,
                         data = bdf,
                         control = list(adapt_delta=0.99),
                         prior = c(set_prior("normal(0, 2.5)", class="b", coef="any_picture_wolf1"),
                                   set_prior("normal(0, 5.0)", class="b", coef="year"),
                                   set_prior("normal(0, 10)", class="Intercept")),
                         family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf)

# Next, what is the probability that time has no positive effect?
mean(posterior_samples(brm.wolf, "year") < 0)
hypothesis(brm.wolf, "year < 0")
hypothesis(brm.wolf, "any_picture_wolf1 > 0")

samples = posterior_samples(brm.wolf, "year", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

wolf_year_density <- qplot(x, y, data = samples, geom = "line",
                      ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm.wolf, "any_picture_wolf", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

wolf_picture_density <- qplot(x, y, data = samples, geom = "line",
                      ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

wolf_plots <- cowplot::plot_grid(wolf_year_density, wolf_picture_density,
                        labels = c("a)", "b)"), align="h")

cowplot::save_plot("../images/wolf.png", wolf_plots, dpi=300, base_width=10, base_height=4,
          base_aspect_ratio = 1.3)

vs.wolf = varsel(brm.wolf, method="forward", cv_method='LOO')
varsel_plot(vs.wolf, stats = c('elpd', 'rmse', 'acc'), deltas=F)
(nv <- suggest_size(vs.wolf, alpha=0.1)) # 0.1
projrhs <- project(vs.wolf, nv = nv, ns = 4000)
round(colMeans(as.matrix(projrhs)), 3)
(postint  <- round(posterior_interval(as.matrix(projrhs), prob=0.95),3))

mcmc_areas_ridges(as.matrix(projrhs), pars = rownames(postint),
           transformations=exp, prob_outer=0.99, )


## ////////////////////////////////////////////////////////////////////
## Red Riding Hood
## ////////////////////////////////////////////////////////////////////
regr = standardize(rrh ~ any_picture_rrh + year + opening, data=df, family = "binomial", scale=0.5)

# compute n datasets with slightly different years for estimated years
n_sims = 100
bdf <- lapply(seq_len(n_sims),
              function (x) overimpute(df, "year", regr, with_error, sd=sd))

auto_prior(regr$formula, regr$data, gaussian = F)

brm.rrh <- brm_multiple(regr$formula,
                        data = bdf,
                        control = list(adapt_delta=0.99),
                        prior = c(set_prior("normal(0, 2.5)", class="b", coef="any_picture_rrh1"),
                                  set_prior("normal(0, 5.0)", class="b", coef="year"),
                                  set_prior("normal(0, 2.5)", class="b", coef="opening1"),
                                  set_prior("normal(0, 10)", class="Intercept")),
                        family = "bernoulli", sample_prior=TRUE)
summary(brm.rrh)

# Next, what is the probability that time has no positive effect?
mean(posterior_samples(brm.rrh, "year") < 0)
hypothesis(brm.rrh, "year > 0") # alternatively
hypothesis(brm.rrh, "any_picture_rrh1 > 0")

samples = posterior_samples(brm.rrh, "year", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

rrh_year_density <- qplot(x, y, data = samples, geom = "line",
                      ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm.rrh, "any_picture_rrh", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

rrh_picture_density <- qplot(x, y, data = samples, geom = "line",
                      ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

rrh_plots <- cowplot::plot_grid(rrh_year_density, rrh_picture_density,
                        labels = c("a)", "b)"), align="h")

cowplot::save_plot("../images/rrh.png", rrh_plots, dpi=300, base_width=10, base_height=4,
          base_aspect_ratio = 1.3)


vs.rrh = varsel(brm.rrh, method="forward", cv_method='LOO')
varsel_plot(vs.rrh, stats = c('elpd', 'rmse', 'acc'), deltas=F)
(nv <- suggest_size(vs.rrh, alpha=0.1))
projrhs <- project(vs.rrh, nv = nv, ns = 4000)
round(colMeans(as.matrix(projrhs)), 1)
(postint  <- round(posterior_interval(as.matrix(projrhs)),1))

mcmc_areas(as.matrix(projrhs), 
           pars = rownames(postint))


## ////////////////////////////////////////////////////////////////////
## Analysis of the effect of time on the presence of formulaic openings
## ////////////////////////////////////////////////////////////////////
regr = standardize(opening ~ year, data=df, family = "binomial", scale=0.5)

# compute n datasets with slightly different years for estimated years
n_sims = 100
bdf <- lapply(seq_len(n_sims),
              function (x) overimpute(df, "year", regr, with_error, sd=sd))

auto_prior(regr$formula, regr$data, gaussian = F)

brm.opening <- brm_multiple(regr$formula,
                            data = bdf,
                            prior = c(set_prior("normal(0, 5.0)", class="b"),
                                      set_prior("normal(0, 10)", class="Intercept")),
                            family = "bernoulli", sample_prior=TRUE)
summary(brm.opening)

mean(posterior_samples(brm.opening, "year") < 0)
hypothesis(brm.opening, "year > 0")
