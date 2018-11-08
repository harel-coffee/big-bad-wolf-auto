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

# from Gelman 2007
# See: http://www.stat.columbia.edu/~gelman/standardize/standardize.R
rescale <- function (x, binary.inputs){
# function to rescale by subtracting the mean and dividing by 2 sd's
  x.obs <- x[!is.na(x)]
  if (!is.numeric(x)) x <- as.numeric(factor(x))
  if (length(unique(x.obs))==2){
    x <- (x - min(x.obs)) / (max(x.obs) - min(x.obs))
    if (binary.inputs=="0/1") return (x)
    else if (binary.inputs=="-0.5,0.5") return (x - 0.5)
    else if (binary.inputs=="center") return (x - mean(x.obs))
    else if (binary.inputs=="full") return ((x - mean(x.obs)) / (2 * sd(x.obs)))
  }      
  else {
    return ((x - mean(x.obs)) / (2 * sd(x.obs)))
  }
}

df <- read.csv("../data/definite.csv")
df <- df[df$reprint == "no",] # exclude remaining reprints
df$year = df$year_estimated
df$rrh = factor(df$rrh_new, levels=0:1, labels=c("indefinite", "definite"))
df$wolf = factor(df$wolf, levels=0:1, labels=c("indefinite", "definite"))
df$any_picture_wolf = df$any_picture_wolf - df$wolf_cover
df$any_picture_rrh = df$any_picture_rrh - df$rrh_cover
head(df)

## ////////////////////////////////////////////////////////////////////
## Overimputation based on measurement errors
## ////////////////////////////////////////////////////////////////////

overimpute <- function(df, column, subset=NULL, sd=1, scale=FALSE) {
    df = copy(df)
    df[subset, column] <- sapply(df[subset, column], partial(rnorm, n=1, sd=sd))
    if (scale) {
        df[, column] = (df[, column] - min(df[, column])) / 50
        df[, column] = rescale(df[, column])
    }
    return(df)
}

sd = 9.140180259205387 # from datereg.py on training data
with_error = df$exact_date == "False"

# compute n datasets with slightly different years for estimated years
n_sims = 10
bdf <- lapply(seq_len(n_sims),
              function (x) overimpute(df, "year", with_error, sd=sd, scale=T))

## ////////////////////////////////////////////////////////////////////
## The wolf
## ////////////////////////////////////////////////////////////////////

brm.wolf <- brm_multiple(wolf ~ (any_picture_wolf + year),
                         data = bdf,
                         control = list(adapt_delta=0.99),
                         prior = c(set_prior("student_t(7, 0, 2.5)", class = "b")),
                         family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf)
plot(brm.wolf)

# Next, what is the probability that time has no positive effect?
mean(posterior_samples(brm.wolf, "year") < 0)
hypothesis(brm.wolf, "year < 0")
hypothesis(brm.wolf, "any_picture_wolf > 0")

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

brm.rrh <- brm_multiple(rrh ~ (year + any_picture_rrh) + opening,
                        data = bdf,
                        control = list(adapt_delta=0.99),
                        prior = c(set_prior("student_t(7, 0, 2.5)", class="b")),
                        family = "bernoulli", sample_prior=TRUE)
summary(brm.rrh)
plot(brm.rrh)

# Next, what is the probability that time has no positive effect?
mean(posterior_samples(brm.rrh, "year") < 0)
hypothesis(brm.rrh, "year > 0") # alternatively
hypothesis(brm.rrh, "any_picture_rrh > 0")

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

brm.opening <- brm_multiple(opening ~ year,
                            data = bdf,
                            prior = c(set_prior("student_t(7, 0, 2.5)", class="b")),
                            family = "bernoulli", sample_prior=TRUE)
summary(brm.opening)
plot(brm.opening)

mean(posterior_samples(brm.opening, "year") < 0)
hypothesis(brm.opening, "year < 0")
