# =============================================================================
# Supplementary Figure S3. ROC curves for locking: five learners
#   Pooled out-of-fold predictions from 5-fold cross-validation (first repeat)
#   AUC 95% CI: DeLong method; pairwise comparison with the best learner:
#   DeLong test, Holm-adjusted
# =============================================================================

library(readxl); library(dplyr); library(ggplot2); library(scales); library(patchwork)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

INK <- "#0b0b0b"; GRID <- "#e4e3df"; IRR <- "#F37021"
LEARNERS <- c("Ensemble", "Lasso", "GBM", "Gated Transformer", "DNN")
LCOL <- c("Ensemble" = IRR, "Lasso" = "#1baf7a", "GBM" = "#7a7873",
          "Gated Transformer" = "#7b3fe4", "DNN" = INK)
LTY  <- c("Ensemble" = "solid", "Lasso" = "solid", "GBM" = "solid", "Gated Transformer" = "solid", "DNN" = "22")

roc <- read_excel(xlsx, sheet = "FigS3_roc") %>% mutate(Learner = factor(Learner, levels = LEARNERS))
# 95% CI band of each ROC curve: 1,000 stratified bootstrap resamples, TPR at fixed FPR grid (percentile)
band <- read_excel(xlsx, sheet = "FigS3_roc_band") %>%
  filter(Learner == "Ensemble") %>%
  mutate(Learner = factor(Learner, levels = LEARNERS))
auc <- read_excel(xlsx, sheet = "FigS3_auc") %>% mutate(Learner = factor(Learner, levels = LEARNERS)) %>% arrange(Learner)

fmt_p <- function(p) ifelse(is.na(p), "Ref.", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
tab <- auc %>% mutate(row = row_number(),
                      lab = sprintf("%.3f (%.3f\u2013%.3f)", AUC, CI_low, CI_high),
                      plab = fmt_p(p_vs_Ensemble_Holm))
p <- ggplot(roc, aes(FPR, TPR, colour = Learner, linetype = Learner)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linewidth = 0.4, linetype = "22") +
  geom_ribbon(data = band, aes(x = FPR, ymin = TPR_low, ymax = TPR_high, fill = Learner),
              alpha = 0.5, colour = NA, inherit.aes = FALSE) +
  geom_step(direction = "hv", linewidth = 0.75) +
  scale_fill_manual(values = LCOL, guide = "none") +
  scale_colour_manual(values = LCOL, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0)) +
  coord_equal() +
  labs(x = "1 \u2212 specificity", y = "Sensitivity",
       title = "Locking: five learners",
       caption = paste0("n = ", format(auc$n[1], big.mark = ","), " (", auc$events[1], " with locking). ",
                        "Pooled out-of-fold predictions, 5-fold cross-validation.\n",
                        "Shaded band: 95% CI of the Ensemble curve (1,000 stratified bootstrap resamples).\nAUC 95% CI, DeLong method. \u2020 DeLong test vs Ensemble, Holm-adjusted.")) +
  theme_classic(base_size = 8.5, base_family = "sans") +
  theme(axis.line = element_line(colour = INK, linewidth = 0.35), axis.ticks = element_line(colour = INK, linewidth = 0.35),
        axis.text = element_text(colour = INK, size = 7.5), axis.title = element_text(colour = INK),
        plot.title = element_text(face = "bold", size = 8.5, colour = INK),
        plot.caption = element_text(size = 7.5, colour = INK, hjust = 0),
        panel.grid.major = element_line(colour = GRID, linewidth = 0.35))

# ---- table under the ROC plot: learner, AUC (95% CI), DeLong p vs Ensemble ----
pt <- ggplot() +
  annotate("text", x = 0.00, y = 0, label = "Learner", hjust = 0, size = 2.65, fontface = "bold", colour = INK) +
  annotate("text", x = 0.52, y = 0, label = "AUC (95% CI)", hjust = 0, size = 2.65, fontface = "bold", colour = INK) +
  annotate("text", x = 1.00, y = 0, label = "p\u2020", hjust = 1, size = 2.65, fontface = "bold.italic", colour = INK) +
  geom_segment(data = tab, aes(x = 0.00, xend = 0.05, y = -row, yend = -row, colour = Learner, linetype = Learner), linewidth = 0.9) +
  geom_text(data = tab, aes(x = 0.07, y = -row, label = Learner), hjust = 0, size = 2.65, colour = INK) +
  geom_text(data = tab, aes(x = 0.52, y = -row, label = lab), hjust = 0, size = 2.65, colour = INK) +
  geom_text(data = tab, aes(x = 1.00, y = -row, label = plab), hjust = 1, size = 2.65, colour = INK) +
  scale_colour_manual(values = LCOL, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(-5.5, 0.5)) +
  theme_void()

figS1 <- p / pt + plot_layout(heights = c(1, 0.32))

ggsave(file.path(out_dir, "FigureS3.png"),  figS1, width = 110, height = 150, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "FigureS3.tiff"), figS1, width = 110, height = 150, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "FigureS3.pdf"),  figS1, width = 110, height = 150, units = "mm", device = cairo_pdf)
