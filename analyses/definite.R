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
df$year = df$year_estimated# rescale((df$year_estimated - min(df$year_estimated) + 1) / 50) # rescaling is done in resampling datasets below
df$rrh = factor(df$rrh_new, levels=0:1, labels=c("indefinite", "definite"))
df$wolf = factor(df$wolf, levels=0:1, labels=c("indefinite", "definite"))
df$picture_wolf = rescale(df$picture_wolf, "center")
df$picture_rrh = rescale(df$picture_rrh, "center")
df$any_picture_wolf = rescale(df$any_picture_wolf, "center")
df$any_picture_rrh = rescale(df$any_picture_rrh, "center")
df$length = rescale(1 + log(df$length))
df$opening = factor(
    df$opening, levels=0:1, labels=c("non-traditional", "traditional"))
## df$picture_wolf = factor(df$picture_wolf + df$wolf_cover)# + df$wolf_cover, levels=0:1, labels=c("no", "yes"))
## df$picture_rrh = factor(df$picture_rrh)# + df$rrh_cover, levels=0:1, labels=c("no", "yes"))
## df$any_picture_rrh = factor(df$any_picture_rrh - df$wolf_cover)
    ## df$any_picture_rrh - df$rrh_cover, levels=0:1, labels=c("no", "yes"))
## df$any_picture_wolf = factor(df$any_picture_wolf)
    ## df$any_picture_wolf - df$wolf_cover, levels=0:1, labels=c("no", "yes"))
## df$length = scale(log(df$length), scale=F)
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
n_sims = 5
bdf <- lapply(seq_len(n_sims),
              function (x) overimpute(df, "year", with_error, sd=sd, scale=T))

## df$year = minmax(df$year)
## ////////////////////////////////////////////////////////////////////
## The wolf
## ////////////////////////////////////////////////////////////////////

brm.wolf <- brm_multiple(wolf ~ any_picture_wolf + year + length,
                         data = bdf,
                         control = list(adapt_delta=0.95),
                         prior = c(set_prior("student_t(7, 0, 2.5)", class = "b")),
                         family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf)
plot(brm.wolf)

# Next, what is the probability that time has no positive effect?
mean(posterior_samples(brm.wolf, "year") < 0)
plot(hypothesis(brm.wolf, "year = 0"))
hypothesis(brm.wolf, "picture_wolf = 0")

vs.wolf = varsel(brm.wolf, method="forward", cv_method='LOO')
varsel_plot(vs.wolf, stats = c('elpd', 'rmse', 'acc'), deltas=TRUE)
(nv <- suggest_size(vs.wolf, alpha=0.32)) # 0.1
projrhs <- project(vs.wolf, nv = nv, ns = 4000)
round(colMeans(as.matrix(projrhs)), 1)
(postint  <- round(posterior_interval(as.matrix(projrhs)),1))

mcmc_areas(as.matrix(projrhs), 
           pars = rownames(postint), transformations=exp)


## ////////////////////////////////////////////////////////////////////
## Red Riding Hood
## ////////////////////////////////////////////////////////////////////

brm.rrh <- brm_multiple(rrh_new ~ year + any_picture_rrh + length,
                        data = bdf,
                        prior = c(set_prior("student_t(7, 0, 2.5)", class = "b")),
                        family = "bernoulli", sample_prior=TRUE)
summary(brm.rrh)
plot(brm.rrh)

# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm.rrh, "year") < 0)
hypothesis(brm.rrh, "year > 0") # alternatively
hypothesis(brm.rrh, "any_picture_rrhyes > 0")

vs.rrh = varsel(brm.rrh, method="forward", cv_method='LOO')
varsel_plot(vs.rrh, stats = c('elpd', 'rmse', 'acc'), deltas=F)
(nv <- suggest_size(vs.rrh, alpha=0.32)) # 0.1
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
hypothesis(brm.opening, "year > 0")

## ////////////////////////////////////////////////////////////////////
## RRH without opening
## ////////////////////////////////////////////////////////////////////

df_rrh <- df[df$opening == "non-traditional",]
with_error_rrh <- df_rrh$exact_date == "False"

n_sims = 5
bdf_rrh <- lapply(seq_len(n_sims),
              function (x) overimpute(df_rrh, "year", with_error_rrh, sd=sd, scale=T))

brm_rrh_no_opening <- brm_multiple(rrh_new ~ year + any_picture_rrh + length,
                                   data = bdf_rrh,
                                   prior = c(
                                       set_prior("student_t(7, 0, 2.5)", class = "b")),
                                   control = list(adapt_delta=0.95),                                   
                                   family = "bernoulli", sample_prior=TRUE)
summary(brm_rrh_no_opening)
plot(brm_rrh_no_opening)

hypothesis(brm_rrh_no_opening, "year > 0")

vs.rrh.no.opening = varsel(brm_rrh_no_opening, method="forward", cv_method='LOO')
varsel_plot(vs.rrh.no.opening, stats = c('elpd', 'rmse', 'acc'), deltas=F)
(nv <- suggest_size(vs.rrh.no.opening, alpha=0.32)) # 0.1
projrhs <- project(vs.rrh.no.opening, nv = nv, ns = 4000)
round(colMeans(as.matrix(projrhs)), 1)
(postint  <- round(posterior_interval(as.matrix(projrhs)),1))

mcmc_areas(as.matrix(projrhs), 
           pars = rownames(postint))
