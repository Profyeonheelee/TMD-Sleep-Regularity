# =============================================================================
# Supplementary Figure S5. Sleep regularity, timing and duration modeled together
#   Model 2 covariates + irregular sleep + sleep timing (ref: night-aligned)
#   + sleep duration (ref: within NSF age-specific range), n = 3,302-3,314
# =============================================================================
library(readxl); library(dplyr); library(ggplot2)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

IRR <- "#F37021"; INK <- "#0b0b0b"; GRID <- "#e4e3df"; NS <- "#4a4a4a"
EXPO <- c("Irregular sleep", "Advanced timing", "Delayed timing", "Short sleep", "Long sleep")
OUT  <- c("VAS", "PI", "DI", "CMI", "Locking")
OUT_LAB <- c(VAS = 'bold("VAS (")*bolditalic(B)*bold(")")', PI = 'bold("PI (")*bolditalic(B)*bold(")")',
             DI = 'bold("DI (")*bolditalic(B)*bold(")")', CMI = 'bold("CMI (")*bolditalic(B)*bold(")")',
             Locking = 'bold("Locking (OR)")')

s5 <- read_excel(xlsx, sheet = "FigS5_timing_duration") %>%
  mutate(Exposure = factor(Exposure, levels = rev(EXPO)),
         OutLab = factor(OUT_LAB[Outcome], levels = OUT_LAB),
         ref = ifelse(Scale == "OR", 1, 0), sig = p < 0.05,
         lab = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

p <- ggplot(s5, aes(y = Exposure, x = Estimate)) +
  scale_y_discrete() +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 4.5, ymax = 5.5, fill = "#FDE7D8") +
  geom_vline(aes(xintercept = ref), colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high, colour = sig), height = 0, linewidth = 0.9) +
  geom_point(aes(colour = sig), shape = 22, fill = "white", size = 2.4, stroke = 0.9) +
  geom_text(aes(x = Inf, label = lab, fontface = ifelse(sig, "bold", "plain")), hjust = 1.05, size = 2.5, colour = INK) +
  facet_wrap(~OutLab, nrow = 1, scales = "free_x", labeller = label_parsed) +
  scale_colour_manual(values = c(`TRUE` = IRR, `FALSE` = NS), guide = "none") +
  scale_x_continuous(breaks = scales::breaks_extended(n = 3), expand = expansion(mult = c(0.06, 0.45))) +
  labs(x = "Adjusted estimate (95% CI); right: p", y = NULL,
       caption = paste("All exposures entered in one model with Model 2 covariates; references: regular sleep, night-aligned mid-sleep (02:00\u201304:59),",
                       "sleep duration within the NSF age-specific range. Linear regression with HC3 robust SE or logistic regression; orange, p < 0.05.", sep = "\n")) +
  theme_classic(base_size = 8.5, base_family = "sans") +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text = element_text(colour = INK, size = 7.5), axis.text.y = element_text(size = 8),
        axis.title = element_text(colour = INK, size = 8), strip.background = element_blank(),
        strip.text = element_text(hjust = 0, colour = INK, size = 8.5),
        panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35), panel.spacing.x = unit(4, "mm"),
        plot.caption = element_text(size = 7.5, colour = INK, hjust = 0))

ggsave(file.path(out_dir, "FigureS5.png"),  p, width = 180, height = 75, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "FigureS5.tiff"), p, width = 180, height = 75, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "FigureS5.pdf"),  p, width = 180, height = 75, units = "mm", device = cairo_pdf)
