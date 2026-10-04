# =============================================================================
# Supplementary Figure S1. Sensitivity to the threshold defining irregular sleep (>=1 h vs >=2 h)
#   Model 2 estimates for DI, CMI, locking and VAS
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ----------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; INK2 <- "#0b0b0b"; GRID <- "#e4e3df"
IRR_LIGHT <- "#F9B88F"; NS <- "#4a4a4a"; BAND <- "#FDE7D8"
theme_fig <- function(base = 8.5) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(axis.line = element_line(colour = INK2, linewidth = 0.35),
          axis.ticks = element_line(colour = INK2, linewidth = 0.35),
          axis.text = element_text(colour = INK, size = 7.5), legend.text = element_text(colour = INK, size = 7.5), axis.title = element_text(colour = INK),
          panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
          plot.title = element_text(face = "bold", size = base + 0.5, colour = INK),
          plot.tag = element_text(face = "bold", size = 13),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, colour = INK))
}
fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

# =============================================================================
# (b) Threshold sensitivity: ≥1 h vs ≥2 h, Model 2, native scale
# =============================================================================
fb <- read_excel(xlsx, sheet = "FigS1_threshold") %>%
  filter(grepl("^Main|range \u22652 h", Analysis)) %>%
  mutate(Threshold = ifelse(grepl("^Main", Analysis), "\u22651 h", "\u22652 h"),
         Threshold = factor(Threshold, levels = c("\u22651 h", "\u22652 h")),
         Outcome   = factor(Outcome, levels = c("DI", "CMI", "Locking", "VAS"),
                            labels = c('bold("DI (")*bolditalic(B)*bold(")")', 'bold("CMI (")*bolditalic(B)*bold(")")',
                                       'bold("Locking (OR)")', 'bold("VAS (")*bolditalic(B)*bold(")")')),
         ref = ifelse(Effect == "OR", 1, 0),
         sig = p < 0.05,
         lab = ifelse(Effect == "OR", sprintf("%.2f", Estimate),
                      ifelse(grepl("VAS", Outcome), sprintf("%.2f", Estimate), sprintf("%.3f", Estimate))),
         plab = paste0("p = ", fmt_p(p)))

pB <- ggplot(fb, aes(x = Threshold, y = Estimate)) +
  geom_hline(aes(yintercept = ref), colour = INK2, linewidth = 0.35, linetype = "22") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high, colour = sig), width = 0, linewidth = 1.0) +
  geom_point(aes(colour = sig), shape = 22, fill = "white", size = 3.0, stroke = 1.0) +
  geom_text(aes(y = Inf, label = plab, fontface = ifelse(sig, "bold", "plain")), vjust = 1.4, size = 2.65, colour = INK) +
  facet_wrap(~Outcome, nrow = 1, scales = "free_y", labeller = label_parsed) +
  scale_colour_manual(values = c(`TRUE` = IRR, `FALSE` = NS), guide = "none") +
  scale_fill_manual(values = c(`TRUE` = IRR, `FALSE` = "white"), guide = "none") +
  scale_x_discrete(expand = expansion(add = 0.5)) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.16))) +
  labs(x = "Bedtime or wake-time range defining irregular sleep", y = "Adjusted estimate (95% CI)",
       caption = "Model 2. p: Wald test from linear regression with HC3 robust SE (DI, CMI, VAS) or logistic regression (locking).") +
  theme_fig() + theme(panel.grid.major.x = element_blank(),
                      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35))

figS2 <- pB
ggsave(file.path(out_dir, "FigureS1.png"),  figS2, width = 180, height = 85, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "FigureS1.tiff"), figS2, width = 180, height = 85, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "FigureS1.pdf"),  figS2, width = 180, height = 85, units = "mm", device = cairo_pdf)
