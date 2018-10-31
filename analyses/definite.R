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


df <- read.csv("definite-anonymous.csv")
## df <- df[df$reprint == "no",]
df$year = df$year_corrected
df = df[df$year >= 1875,]
df$rrh = factor(df$rrh, levels=0:1, labels=c("indefinite", "definite"))
df$rrh_new = factor(df$rrh_new, levels=0:1, labels=c("indefinite", "definite"))
df$wolf = factor(df$wolf, levels=0:1, labels=c("indefinite", "definite"))
df$opening = factor(
    df$opening, levels=0:1, labels=c("non-traditional", "traditional"))
df$timebin = as.integer(bin(df$year_corrected, 6, method="length"))

df$picture_wolf = factor(df$picture_wolf + df$wolf_cover, levels=0:1, labels=c("no", "yes"))
df$picture_rrh = factor(df$picture_rrh + df$rrh_cover, levels=0:1, labels=c("no", "yes"))

minmax <- function(x) {
    (x - min(x)) / (max(x) - min(x))
}

df$year = ((df$year_corrected %/% 10 * 10) - mean(df$year_corrected)) / 100

## df$year = scale(df$year_corrected)

df$length = scale(log(df$length))

df$any_picture_rrh = factor(
    df$any_picture_rrh - df$rrh_cover, levels=0:1, labels=c("no", "yes"))
df$any_picture_wolf = factor(
    df$any_picture_wolf - df$wolf_cover, levels=0:1, labels=c("no", "yes"))
df_rrh <- df[df$opening == "non-traditional",]

periods = character(length(unique(df$timebin)))
for (i in unique(df$timebin)) {
    periods[i] <- paste(min(df[df$timebin == i,]$year) %/% 10 * 10, "-",
                        max(df[df$timebin == i,]$year) %/% 10 * 10, sep="")
}
periods
table(df$timebin)

## ////////////////////////////////////////////////////////////////////
## The wolf
## ////////////////////////////////////////////////////////////////////

m = glm(wolf ~ year + picture_wolf + length, data=df, family = "binomial")
summary(m)
Anova(m)
plot(effect("year", m))

m = glm(rrh_new ~ year + picture_rrh + length, data=df, family = "binomial")
summary(m)
Anova(m)
plot(effect("year", m))

m = glm(rrh_new ~ scale(year) + picture_rrh + length, data=df_rrh, family = "binomial")
summary(m)
Anova(m)
plot(effect("length", m))

m = glm(opening ~ year + length, data=df, family = binomial("logit"))
summary(m)
Anova(m)

brm.wolf <- brm(wolf ~ any_picture_wolf + year + length,
                data = df,
                control = list(adapt_delta=0.999),
                prior = c(set_prior("student_t(7, 0, 1)", class = "b")),
                family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf)
plot(brm.wolf)

brm.wolf <- brm(rrh_new ~ any_picture_rrh + year + length + genre,
                data = df,
                control = list(adapt_delta=0.999),
                prior = c(set_prior("student_t(7, 0, 1)", class = "b")),
                family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf)
plot(brm.wolf)


brm.wolf.author <- brm(wolf ~ any_picture_wolf + scale(year) + (1|author),
                data = df,
                control = list(adapt_delta=0.999),
                prior = c(set_prior("student_t(7, 0, 1)", class = "b")),
                family = "bernoulli", sample_prior = TRUE)
summary(brm.wolf.author)
loo(brm.wolf, brm.wolf.author)


brm.wolf <- stan_glmer(wolf ~ any_picture_wolf + scale(year) + (1|author),
                       data = df,
                       prior = student_t(7, 0, 1),
                       prior_intercept = cauchy(0, 10),
                       family = "binomial")


# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm.wolf, "year") < 0)

plot(hypothesis(brm.wolf, "scaleyear > 0"))

plot(hypothesis(brm.wolf, "any_picture_wolfyes > 0"))

wolf_effect_plot <- df %>%
    add_fitted_draws(brm.wolf) %>%
    ggplot(aes(x = year, y = wolf)) +
    stat_lineribbon(aes(y = .value), color="#03DAC6") +
    scale_fill_brewer(name="CI", palette = "Greys") +
    scale_x_continuous() +
    ## scale_x_discrete(
    ##     name ="Time period", 
    ##     limits = periods) +
    scale_y_continuous(
        name = "P(definite introduction)",
        labels = percent,
        limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 25, vjust=0.6)) + 
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm_wolf, "timebin", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

wolf_posterior_plot <- qplot(x, y, data = samples, geom = "line",
                             ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

wolf_plots <- plot_grid(wolf_posterior_plot, wolf_effect_plot,
          labels = c("a)", "b)"), align="h")

save_plot("wolf.png", wolf_plots, dpi=300, base_width=10, base_height=4,
          base_aspect_ratio = 1.3)

exp(quantile(as.matrix(brm_wolf)[,2], probs=c(.5, .025, .975)))


## ////////////////////////////////////////////////////////////////////
## Red Riding Hood
## ////////////////////////////////////////////////////////////////////

summary(glm(rrh ~ timebin, data=df, family = "binomial"))
summary(glmer(rrh ~ timebin + (1|author), data = df, family = "binomial"))

brm_rrh <- brm(rrh_new ~ timebin + any_picture_rrh, data = df,
               prior = c(set_prior("student_t(7, 0, 1)", class = "b")),
               family = "bernoulli", iter = 8000, sample_prior=TRUE)
summary(brm_rrh)
plot(brm_rrh)

# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm_rrh, "timebin") < 0)
hypothesis(brm_rrh, "timebin > 0") # alternatively

rrh_effect_plot <- df %>%
    add_fitted_draws(brm_rrh) %>%
    ggplot(aes(x = timebin, y = rrh)) +
    stat_lineribbon(aes(y = .value), color="#03DAC6") +
    scale_fill_brewer(name="CI", palette = "Greys") +
    scale_x_discrete(
        name ="Time period", 
        limits = periods) +
    scale_y_continuous(
        name = "P(definite introduction)",
        labels = percent,
        limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 25, vjust=0.6)) + 
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm_rrh, "timebin", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

rrh_posterior_plot <- qplot(x, y, data = samples, geom = "line",
                             ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

rrh_plots <- plot_grid(rrh_posterior_plot, rrh_effect_plot,
          labels = c("a)", "b)"), align="h")

save_plot("rrh.png", rrh_plots, dpi=300, base_width=10, base_height=4,
          base_aspect_ratio = 1.3)

exp(quantile(as.matrix(brm_rrh)[,2], probs=c(.5, .025, .975)))


## ////////////////////////////////////////////////////////////////////
## Analysis of the effect of time on the presence of formulaic openings
## ////////////////////////////////////////////////////////////////////

brm_opening <- brm(opening ~ timebin + (1|author), data = df,
                   ## prior = c(set_prior("normal(0, .2)", class = "b")),
                   prior = c(set_prior("student_t(30, 0, 0.5)", class="b")),
                   control = list(adapt_delta=0.95),
                   family = "bernoulli", iter = 8000, sample_prior=TRUE)
summary(brm_opening)
plot(brm_opening)

mean(posterior_samples(brm_opening, "timebin") > 0)
hypothesis(brm_opening, "timebin = 0")

opening_effect_plot <- df %>%
    add_fitted_draws(brm_opening) %>%
    ggplot(aes(x = timebin, y = opening)) +
    stat_lineribbon(aes(y = .value), color="#03DAC6") +
    scale_fill_brewer(name="CI", palette = "Greys") +
    scale_x_discrete(
        name ="Time period", 
        limits = periods) +
    scale_y_continuous(
        name = "P(definite introduction)",
        labels = percent,
        limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 25, vjust=0.6)) + 
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm_opening, "timebin", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

opening_posterior_plot <- qplot(x, y, data = samples, geom = "line",
                             ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

opening_plots <- plot_grid(opening_posterior_plot, opening_effect_plot,
                           labels = c("a)", "b)"), align="h")

save_plot("opening.png", opening_plots, dpi=300, base_width=10, base_height=4,
          base_aspect_ratio = 1.3)


exp(quantile(as.matrix(brm_opening)[,2], probs=c(.5, .025, .975)))


## ////////////////////////////////////////////////////////////////////
## RRH without opening
## ////////////////////////////////////////////////////////////////////

df_rrh <- df[df$opening == 0,]

summary(glm(rrh ~ timebin + pictures, data=df_rrh, family = "binomial"))
m = glmer(rrh ~ timebin + (1|author), data=df_rrh, family = "binomial")

m  <- stan_glmer(rrh ~ timebin + (1|author), data=df_rrh, family="binomial",
                 prior = student_t(df = 7, 0, 1),
                 prior_intercept = cauchy(0,10))


brm_rrh_no_opening <- brm(rrh_new ~ timebin + any_picture_rrh + (1|book_type),
                          data = df_rrh,
                          prior = c(
                              set_prior("student_t(7, 0, 1)", class = "b")),
                          control = list(adapt_delta=0.999),
                          family = "bernoulli", iter = 8000, sample_prior=TRUE)
summary(brm_rrh_no_opening)
plot(brm_rrh_no_opening)

hypothesis(brm_rrh_no_opening, "timebin > 0")

rrh_no_opening_effect_plot <- df %>%
    add_fitted_draws(brm_rrh_no_opening) %>%
    ggplot(aes(x = timebin, y = rrh)) +
    stat_lineribbon(aes(y = .value), color="#03DAC6") +
    scale_fill_brewer(name="CI", palette = "Greys") +
    scale_x_discrete(
        name ="Time period", 
        limits = periods) +
    scale_y_continuous(
        name = "P(definite introduction)",
        labels = percent,
        limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 25, vjust=0.6)) + 
    background_grid(major = "xy", minor = "none")

samples = posterior_samples(brm_rrh_no_opening, "timebin", as.array = T)
samples.dens <- density(samples)
samples.q25 <- quantile(samples, .025)
samples.q975 <- quantile(samples, .975)
samples <- with(samples.dens, data.frame(x,y))

rrh_no_opening_posterior_plot <- qplot(x, y, data = samples, geom = "line",
                                       ylab="density", xlab="estimate") +
    geom_ribbon(data = subset(samples, x > samples.q25 & x < samples.q975),
                aes(ymax = y), ymin = 0, fill="#03DAC6", alpha = 0.8) +
    background_grid(major = "xy", minor = "none")

rrh_no_opening_plots <- plot_grid(rrh_no_opening_posterior_plot,
                                  rrh_no_opening_effect_plot,
                           labels = c("a)", "b)"), align="h")

save_plot("rrh_no_opening.png", rrh_no_opening_plots, dpi=300,
          base_width=10, base_height=4,
          base_aspect_ratio = 1.3)

exp(quantile(as.matrix(brm_rrh_no_opening)[,2], probs=c(.5, .025, .975)))

