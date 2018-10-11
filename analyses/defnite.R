library(arm)
library(brms)
require(gridExtra)
library(scales)
library(tidyr)
library(tidybayes)
library(cowplot)
library(OneR)

                                        # 5, 1820
df <- read.csv("defnite.csv")
df <- df[df$year >= 1820,]
df$rrh = as.factor(df$rrh)
df$wolf = as.factor(df$wolf)
df$opening = as.factor(df$opening)
df$timebin = as.integer(bin(df$year, 5, method="length"))
df_rrh <- df[df$opening == 0,]

periods = character(length(unique(df$timebin)))
for (i in unique(df$timebin)) {
    periods[i] <- paste(min(df[df$timebin == i,]$year) %/% 10 * 10, "-",
                        max(df[df$timebin == i,]$year) %/% 10 * 10, sep="")
}
periods


## ////////////////////////////////////////////////////////////////////
## The wolf
## ////////////////////////////////////////////////////////////////////

brm_wolf <- brm(wolf ~ timebin, data = df,
                prior = c(set_prior("student_t(7,0,1)", class = "b")),
                family = "bernoulli", iter = 8000, sample_prior = TRUE)
summary(brm_wolf)
plot(brm_wolf)

# Next, what is the probability that timebin has no positive effect?
mean(posterior_samples(brm_wolf, "timebin") < 0)
hypothesis(brm_wolf, "timebin < 0") # alternatively

wolf_effect_plot <- df %>%
    add_fitted_draws(brm_wolf) %>%
    ggplot(aes(x = timebin, y = wolf)) +
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

brm_rrh <- brm(rrh ~ timebin, data = df,
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

brm_opening <- brm(opening ~ timebin, data = df,
                   prior = c(set_prior("student_t(7, 0, 1)", class = "b")),
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
brm_rrh_no_opening <- brm(rrh ~ timebin, data = df_rrh,
                          prior = c(
                              set_prior("student_t(7, 0, 1)", class = "b")),
                          family = "bernoulli", iter = 8000, sample_prior = TRUE)
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
