# =============================================================================
# Figure 4. The association is robust to estimator choice, permutation, unmeasured confounding and entry period
#   a  Five learners x two doubly robust estimators (DML-PLR, AIPW)
#   b  Permutation test
#   c  E-values
#   d  Replication across entry periods
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ----------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; GRID <- "#e4e3df"
NS <- "#4a4a4a"; NS_FILL <- "#d9d8d4"; IRR_FILL <- "#F9C4A2"
SIG_COL <- c(`TRUE` = IRR, `FALSE` = NS)
RED <- "#B23A3A"
theme_fig <- function(base = 8.5) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(axis.line = element_line(colour = INK, linewidth = 0.35),
          axis.ticks = element_line(colour = INK, linewidth = 0.35),
          axis.text = element_text(colour = INK, size = 7.5), legend.text = element_text(colour = INK, size = 7.5),
          axis.title = element_text(colour = INK, size = 8),
          plot.title = element_text(size = 8.5, face = "bold", colour = INK),
          plot.caption = element_text(size = 7.5, colour = INK, hjust = 0),
          plot.tag = element_text(face = "bold", size = 13),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, colour = INK, size = 8.5))
}
fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
OUT4 <- c("VAS", "DI", "CMI", "Locking")

# =============================================================================
# (a) Five learners x two estimators
#     Symbol: vertical tick (no model symbol - these are not Models 1-4).
#     Line type: solid = DML-PLR, dashed = AIPW.
# =============================================================================
LEARNERS <- c("Lasso", "GBM", "DNN", "Gated Transformer", "Ensemble")
fa <- read_excel(xlsx, sheet = "Fig4a_learners") %>%
  filter(Outcome %in% c("DI", "CMI", "Locking", "VAS")) %>%
  mutate(Outcome = factor(Outcome, levels = c("DI", "CMI", "Locking", "VAS"),
                          labels = c('bold("DI (")*bolditalic(B)*bold(")")', 'bold("CMI (")*bolditalic(B)*bold(")")',
                                     'bold("Locking (RD)")', 'bold("VAS (")*bolditalic(B)*bold(")")')),
         Learner = factor(Learner, levels = rev(LEARNERS)),
         Estimator = factor(Estimator, levels = c("AIPW", "DML-PLR")),
         sig = p < 0.05)
dodge <- position_dodge(width = 0.6)

pA <- ggplot(fa, aes(y = Learner, x = Effect, group = Estimator)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high, colour = sig, linetype = Estimator),
                 height = 0, linewidth = 0.8, position = dodge) +
  geom_point(aes(colour = sig), shape = 124, size = 3.2, stroke = 1.2, position = dodge) +
  geom_text(aes(x = Inf, label = fmt_p(p), fontface = ifelse(sig, "bold", "plain")),
            hjust = 1.05, size = 2.5, colour = INK, position = dodge) +
  facet_wrap(~Outcome, nrow = 1, scales = "free_x", labeller = label_parsed) +
  scale_colour_manual(values = SIG_COL, guide = "none") +
  scale_linetype_manual(values = c("DML-PLR" = "solid", "AIPW" = "22"), name = NULL,
                        breaks = c("DML-PLR", "AIPW")) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.42))) +
  labs(x = "Effect of irregular sleep (95% CI); right: p.  RD, risk difference", y = NULL,
       caption = paste("DML-PLR, double/debiased machine learning (partially linear model); AIPW, augmented inverse probability weighting.",
                       "5-fold cross-fitting (5 repeats; single cross-fit for the Gated Transformer), propensity trimming at 0.02; covariates as in Methods. p: Wald test.", sep = "\n")) +
  guides(linetype = guide_legend(override.aes = list(colour = INK, linewidth = 0.8))) +
  theme_fig() +
  theme(panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
        axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        panel.spacing.x = unit(3, "mm"),
        legend.position = "top", legend.justification = "left", legend.margin = margin(0, 0, -4, 0),
        legend.key.width = unit(7, "mm"))

# =============================================================================
# (b) Permutation null distributions (1,000 permutations within entry quarter)
# =============================================================================
pm <- read_excel(xlsx, sheet = "Fig4b_permutation") %>% mutate(Outcome = factor(Outcome, levels = c("DI", "CMI", "Locking", "VAS")))
obs <- pm %>% distinct(Outcome, Observed, Perm_p) %>% mutate(sig = Perm_p < 0.05,
                                                             lab = paste0("p = ", fmt_p(Perm_p)))
PM_LAB <- c(DI = 'bold("DI (")*bolditalic(B)*bold(")")', CMI = 'bold("CMI (")*bolditalic(B)*bold(")")',
            Locking = 'bold("Locking (log OR)")', VAS = 'bold("VAS (")*bolditalic(B)*bold(")")')
pm  <- pm  %>% mutate(OutLab = factor(PM_LAB[as.character(Outcome)], levels = PM_LAB))
obs <- obs %>% mutate(OutLab = factor(PM_LAB[as.character(Outcome)], levels = PM_LAB))

pB <- ggplot(pm, aes(x = Null)) +
  geom_histogram(bins = 35, fill = NS_FILL, colour = "white", linewidth = 0.15) +
  geom_vline(data = obs, aes(xintercept = Observed, colour = sig), linewidth = 1.0) +
  geom_text(data = obs, aes(x = -Inf, y = Inf, label = lab, fontface = ifelse(sig, "bold", "plain")),
            hjust = -0.08, vjust = 1.4, size = 2.65, colour = INK) +
  facet_wrap(~OutLab, nrow = 1, scales = "free", labeller = label_parsed) +
  scale_colour_manual(values = c(`TRUE` = RED, `FALSE` = NS), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_x_continuous(breaks = breaks_pretty(n = 3)) +
  labs(x = "Estimate under permuted exposure (grey) vs observed (line)", y = "Permutations",
       caption = "Exposure permuted 1,000 times within entry quarter; Model 2 covariates. p: two-sided permutation p.") +
  theme_fig() + theme(panel.spacing.x = unit(4, "mm"))

# =============================================================================
# (c) E-values
# =============================================================================
ev <- read_excel(xlsx, sheet = "Fig4c_evalue") %>% filter(Outcome %in% c("DI", "CMI", "Locking", "VAS")) %>%
  mutate(Outcome = factor(Outcome, levels = rev(c("DI", "CMI", "Locking", "VAS")))) %>%
  pivot_longer(c(E_point, E_CI), names_to = "Type", values_to = "E") %>%
  mutate(Type = factor(Type, levels = c("E_CI", "E_point"), labels = c("CI limit", "Point estimate")))

pC <- ggplot(ev, aes(y = Outcome, x = E, fill = Type)) +
  geom_vline(xintercept = 1, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", E)), position = position_dodge(width = 0.75), hjust = -0.2, size = 2.65, colour = INK) +
  scale_fill_manual(values = c("Point estimate" = IRR, "CI limit" = IRR_FILL), name = NULL,
                    breaks = c("Point estimate", "CI limit")) +
  scale_x_continuous(limits = c(0, 1.8), breaks = c(0, 0.5, 1, 1.5), expand = c(0, 0)) +
  coord_cartesian(xlim = c(0.9, 1.8)) +
  labs(x = "E-value (risk-ratio scale)", y = NULL, title = "Unmeasured confounding",
       caption = "Confounder strength (RR) needed\nto explain away the estimate\n(VanderWeele & Ding 2017).") +
  theme_fig() + theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                      axis.text.y = element_text(face = "bold", size = 8),
                      panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
                      legend.position = "top", legend.justification = "left", legend.key.size = unit(3, "mm"),
                      legend.margin = margin(0, 0, -4, 0))

# =============================================================================
# (d) Replication across entry periods (Model 2; square = Model 2, as in Figure 2)
# =============================================================================
rp <- read_excel(xlsx, sheet = "Fig4d_period") %>% filter(Outcome %in% c("DI", "CMI", "Locking", "VAS")) %>%
  mutate(OutLab = factor(c(DI = 'bold("DI (")*bolditalic(B)*bold(")")', CMI = 'bold("CMI (")*bolditalic(B)*bold(")")',
                           Locking = 'bold("Locking (OR)")', VAS = 'bold("VAS (")*bolditalic(B)*bold(")")')[Outcome],
                         levels = c('bold("DI (")*bolditalic(B)*bold(")")', 'bold("CMI (")*bolditalic(B)*bold(")")',
                                    'bold("Locking (OR)")', 'bold("VAS (")*bolditalic(B)*bold(")")')),
         Period = factor(Period, levels = c("2021\u20132022", "2023"), labels = c("2021\u2013\n2022", "2023")),
         ref = ifelse(Outcome == "Locking", 1, 0), sig = p < 0.05)

pD <- ggplot(rp, aes(x = Period, y = Estimate)) +
  geom_hline(aes(yintercept = ref), colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high, colour = sig), width = 0, linewidth = 1.0) +
  geom_point(aes(colour = sig), shape = 22, fill = "white", size = 2.8, stroke = 1.0) +
  geom_text(aes(y = Inf, label = fmt_p(p), fontface = ifelse(sig, "bold", "plain")), vjust = 1.3, size = 2.5, colour = INK) +
  facet_wrap(~OutLab, nrow = 1, scales = "free_y", labeller = label_parsed) +
  scale_colour_manual(values = SIG_COL, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.16))) +
  labs(x = "Entry period", y = "Estimate (95% CI)", title = "Replication across entry periods",
       caption = "Model 2, fitted separately by period. Numbers: p (Wald test;\nHC3 robust SE for DI, CMI, VAS; logistic regression for locking).") +
  theme_fig() + theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35),
                      strip.text = element_text(size = 7.5), axis.text.x = element_text(lineheight = 0.85),
                      panel.spacing.x = unit(3, "mm"))

# =============================================================================
# Assemble
# =============================================================================
row3 <- wrap_elements(full = ((pC + labs(tag = "c")) | (pD + labs(tag = "d"))) + plot_layout(widths = c(0.8, 1.9)))
fig4 <- wrap_elements(full = pA + labs(tag = "a")) / wrap_elements(full = pB + labs(tag = "b")) / row3 +
  plot_layout(heights = c(1.15, 0.75, 0.95))

ggsave(file.path(out_dir, "Figure4.png"),  fig4, width = 180, height = 245, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "Figure4.tiff"), fig4, width = 180, height = 245, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "Figure4.pdf"),  fig4, width = 180, height = 245, units = "mm", device = cairo_pdf)
