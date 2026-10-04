# =============================================================================
# Supplementary Figure S2. Cross-validated predictive performance of five learners
#   R2 (VAS, DI, CMI) and AUC (locking); 5-fold CV x 2 repeats
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
# (a) Cross-validated performance (5-fold x 2 repeats = 10 folds)
# =============================================================================
LEARN <- c("Lasso", "GBM", "DNN", "Gated Transformer", "Ensemble")
LEARN_AB <- c("Lasso" = "Lasso", "GBM" = "GBM", "DNN" = "DNN", "Gated Transformer" = "Gated\nTransformer", "Ensemble" = "Ensemble")
cvf <- read_excel(xlsx, sheet = "FigS2_cv_folds")
cvs <- read_excel(xlsx, sheet = "FigS2_cv_summary") %>%
  mutate(plab = ifelse(Best, "best", fmt_p(p_vs_best_Holm)))
strip_lab <- cvs %>% distinct(Outcome, Friedman_p) %>%
  mutate(lab = paste0(Outcome, ifelse(Outcome == "Locking", " (AUC)", " (R\u00b2)"),
                      ";  Friedman p ", ifelse(Friedman_p < 0.001, "< 0.001", sprintf("= %.3f", Friedman_p))))
LAB <- setNames(strip_lab$lab, strip_lab$Outcome)
cvf <- cvf %>% left_join(cvs %>% select(Outcome, Learner, Best), by = c("Outcome", "Learner")) %>%
  mutate(Outcome = factor(Outcome, levels = OUT4), Learner = factor(Learner, levels = LEARN))
cvs <- cvs %>% mutate(Outcome = factor(Outcome, levels = OUT4), Learner = factor(Learner, levels = LEARN))

set.seed(2)
pA <- ggplot(cvf, aes(x = Learner, y = Score)) +
  geom_jitter(aes(colour = Best), width = 0.12, height = 0, size = 1.1, alpha = 0.75) +
  geom_errorbar(data = cvs, aes(y = Mean, ymin = Mean - SD, ymax = Mean + SD, colour = Best), width = 0, linewidth = 0.8) +
  geom_errorbar(data = cvs, aes(y = Mean, ymin = Mean, ymax = Mean, colour = Best), width = 0.45, linewidth = 1.0) +
  geom_text(data = cvs, aes(y = Inf, label = plab, fontface = ifelse(Best | p_vs_best_Holm >= 0.05, "plain", "bold")),
            vjust = 1.3, size = 2.5, colour = INK) +
  facet_wrap(~Outcome, nrow = 2, scales = "free_y", labeller = labeller(Outcome = LAB)) +
  scale_colour_manual(values = c(`TRUE` = RED, `FALSE` = NS), guide = "none") +
  scale_x_discrete(labels = LEARN_AB) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = NULL, y = "Cross-validated performance",
       caption = paste("Dots: 10 folds (5-fold CV, 2 repeats); bar: mean \u00b1 SD; red: best learner.",
                       "Numbers: p vs best learner, Holm-adjusted (corrected resampled t test [Nadeau\u2013Bengio]; DeLong test for locking).",
                       "Friedman test across the five learners.", sep = "\n")) +
  theme_fig() + theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35),
                      strip.text = element_text(size = 8), panel.spacing.x = unit(8, "mm"), panel.spacing.y = unit(4, "mm"))

figS3 <- pA
ggsave(file.path(out_dir, "FigureS2.png"),  figS3, width = 180, height = 130, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "FigureS2.tiff"), figS3, width = 180, height = 130, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "FigureS2.pdf"),  figS3, width = 180, height = 130, units = "mm", device = cairo_pdf)
