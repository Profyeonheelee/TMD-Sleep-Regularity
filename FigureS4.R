# =============================================================================
# Supplementary Figure S4. Heterogeneity of the effect of irregular sleep on jaw dysfunction (exploratory)
#   a  Distribution of individual effects (CATE) for DI from a causal forest
#   b  Which characteristics drive the variation in CATE (causal forest importance)
#   c  Subgroup effects (Model 2, stratified) with interaction tests, DI and CMI
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ----------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; GRID <- "#e4e3df"
NS <- "#4a4a4a"; NS_FILL <- "#d9d8d4"; IRR_FILL <- "#F9C4A2"
RED <- "#B23A3A"; TEAL <- "#1B8F6A"; BAND_GREY <- "#ECEBE8"
SIG_COL <- c(`TRUE` = RED, `FALSE` = NS)
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

# =============================================================================
# (a) Individual effects (CATE) for DI
# =============================================================================
cate <- read_excel(xlsx, sheet = "FigS4_cate_patients")
ate  <- read_excel(xlsx, sheet = "FigS4_ate")
pA <- ggplot(cate, aes(x = CATE)) +
  annotate("rect", xmin = ate$CI_low, xmax = ate$CI_high, ymin = -Inf, ymax = Inf, fill = BAND_GREY) +
  geom_histogram(bins = 45, fill = "#a9a7a2", colour = "white", linewidth = 0.15) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_vline(xintercept = ate$ATE, colour = RED, linewidth = 1.0) +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = 2.65, colour = INK,
           label = sprintf("ATE %.3f\n(95%% CI %.3f to %.3f)", ate$ATE, ate$CI_low, ate$CI_high)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = quote("Individual effect of irregular sleep on DI ("*italic(B)*")"), y = "Patients",
       title = "Individual effects (causal forest)",
       caption = paste0("n = ", format(nrow(cate), big.mark = ","), ". Line and band: average treatment effect and 95% CI.")) +
  theme_fig()

# =============================================================================
# (c) Subgroup effects with interaction p (Model 2, stratified)
# =============================================================================
MOD_ORDER <- c("Age", "Sex", "Stress", "Clenching", "Sleep quality")
fb <- read_excel(xlsx, sheet = "FigS4_interaction") %>%
  mutate(Modifier = factor(Modifier, levels = MOD_ORDER),
         Outcome = factor(Outcome, levels = c("DI", "CMI"),
                          labels = c('bold("DI (")*bolditalic(B)*bold(")")', 'bold("CMI (")*bolditalic(B)*bold(")")')),
         sig = p < 0.05) %>%
  arrange(Modifier) %>%
  group_by(Outcome) %>% mutate(row = rev(seq_len(n()))) %>% ungroup()
# rows: two subgroups per modifier; spacer between modifiers
fb <- fb %>% group_by(Outcome) %>% mutate(y = row + (as.integer(Modifier) - 1) * 0) %>% ungroup()
ylab <- fb %>% filter(Outcome == levels(Outcome)[1]) %>% select(y, Subgroup)
pint <- fb %>% group_by(Outcome, Modifier) %>% summarise(y = mean(y), p_int = first(p_interaction), .groups = "drop")
band <- pint %>% filter(as.integer(Modifier) %% 2 == 1) %>% mutate(ymin = y - 1, ymax = y + 1)

pB <- ggplot(fb, aes(y = y)) +
  geom_rect(data = band, aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax), fill = "#f4f3f0", inherit.aes = FALSE) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high, colour = sig), height = 0, linewidth = 0.9) +
  geom_point(aes(x = B, colour = sig), shape = 22, fill = "white", size = 2.4, stroke = 0.9) +
  geom_text(data = pint, aes(x = Inf, y = y, label = paste0("p for interaction = ", fmt_p(p_int))),
            hjust = 1.05, size = 2.5, colour = INK) +
  facet_wrap(~Outcome, nrow = 1, scales = "free_x", labeller = label_parsed) +
  scale_colour_manual(values = SIG_COL, guide = "none") +
  scale_y_continuous(breaks = ylab$y, labels = ylab$Subgroup, expand = c(0.02, 0.02)) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.6))) +
  labs(x = "Effect of irregular sleep (95% CI)", y = NULL, title = "Subgroup effects",
       caption = paste("Squares: Model 2 estimate within each subgroup (linear regression, HC3 robust SE); red, p < 0.05.",
                       "p for interaction: Wald test for the irregular sleep \u00d7 subgroup interaction term in the full sample.", sep = "\n")) +
  theme_fig() + theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                      panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
                      panel.spacing.x = unit(5, "mm"))

# =============================================================================
# (b) Drivers of effect heterogeneity (causal forest feature importance)
# =============================================================================
FEAT_LAB <- c(Age = "Age", Midsleep_h = "Mid-sleep time", log_dur = "Symptom duration", PSQI_core = "PSQI core score",
              Sleep_duration_h = "Sleep duration", Bilateral_pain = "Bilateral pain", STOP_score = "STOP score",
              Stress = "Stress", Clenching = "Clenching", Female = "Female sex", Bruxism = "Bruxism")
SLEEP_F <- c("Midsleep_h", "PSQI_core", "Sleep_duration_h", "STOP_score")
fc <- read_excel(xlsx, sheet = "FigS4_cate_drivers") %>%
  mutate(Label = FEAT_LAB[Feature], sleep = Feature %in% SLEEP_F,
         Label = factor(Label, levels = Label[order(Importance)]))
pC <- ggplot(fc, aes(y = Label, x = Importance * 100, fill = sleep)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", Importance * 100)), hjust = -0.2, size = 2.65, colour = INK) +
  scale_fill_manual(values = c(`TRUE` = TEAL, `FALSE` = NS_FILL), labels = c(`TRUE` = "Sleep feature", `FALSE` = "Other"),
                    breaks = c("TRUE", "FALSE"), name = NULL) +
  scale_x_continuous(limits = c(0, 23), expand = c(0, 0)) +
  labs(x = "Importance (%)", y = NULL, title = "Drivers of heterogeneity",
       caption = "Causal forest split importance\n(1,000 trees).") +
  theme_fig() + theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                      legend.position = "top", legend.justification = "left", legend.key.size = unit(3, "mm"),
                      legend.margin = margin(0, 0, -4, 0))

# =============================================================================
# Assemble
# =============================================================================
top <- wrap_elements(full = ((pA + labs(tag = "a")) | (pC + labs(tag = "b"))) + plot_layout(widths = c(1.5, 1)))
fig6 <- top / wrap_elements(full = pB + labs(tag = "c")) + plot_layout(heights = c(1, 1.1))
# panels: a individual effects (pA), b drivers (pC), c subgroups (pB)

ggsave(file.path(out_dir, "FigureS4.png"),  fig6, width = 180, height = 175, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "FigureS4.tiff"), fig6, width = 180, height = 175, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "FigureS4.pdf"),  fig6, width = 180, height = 175, units = "mm", device = cairo_pdf)
