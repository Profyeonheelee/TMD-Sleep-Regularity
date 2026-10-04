# =============================================================================
# Figure 2. Irregular sleep is associated with jaw dysfunction but not with pain
#   a  Pain-function dissociation across Models 1-4 (standardized scale)
#   b  Equivalence test for pain (TOST): VAS and PI
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
# (a) Forest plot: 6 outcomes × 4 models
# =============================================================================
OUT_ORDER <- c("VAS", "PI", "DI", "CMI", "Locking", "Joint noise")
MOD_SHAPE <- c("Model 1" = 21, "Model 2" = 22, "Model 3" = 23, "Model 4" = 24)
MOD_OFF   <- c("Model 1" = 0.27, "Model 2" = 0.09, "Model 3" = -0.09, "Model 4" = -0.27)

fa <- read_excel(xlsx, sheet = "Fig2a_models") %>%
  mutate(Outcome = factor(Outcome, levels = rev(OUT_ORDER)),
         y   = as.numeric(Outcome) + MOD_OFF[Model],
         sig = q_FDR < 0.05)

# right-hand text column: Model 4 estimate (native scale) and FDR q
lab4 <- fa %>% filter(Model == "Model 4") %>%
  # plotmath strings (parsed): italic B for unstandardized regression coefficients
  mutate(txt = ifelse(Scale == "OR",
                      sprintf("'OR %.2f (%.2f to %.2f)'", Estimate, CI_low, CI_high),
                      ifelse(Outcome == "VAS",
                             sprintf("italic(B)~'%.2f (%.2f to %.2f)'", Estimate, CI_low, CI_high),
                             sprintf("italic(B)~'%.3f (%.3f to %.3f)'", Estimate, CI_low, CI_high))),
         qtxt = fmt_p(q_FDR))

X_RANGE <- c(-0.17, 0.32); X_TXT <- 0.345; X_Q <- 0.62
n_out <- length(OUT_ORDER)
domain_band <- tibble(ymin = 1.5, ymax = 4.5)

pA <- ggplot(fa) +
  geom_rect(data = domain_band, aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
            fill = BAND, inherit.aes = FALSE) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbarh(aes(y = y, xmin = Std_low, xmax = Std_high, colour = sig), height = 0, linewidth = 0.9) +
  geom_point(aes(y = y, x = Std_est, shape = Model, colour = sig), fill = "white", size = 2.6, stroke = 0.9) +
  # domain labels
  annotate("text", x = X_RANGE[1], y = n_out + 0.38, label = "Pain", hjust = 0, vjust = 1,
           size = 2.6, fontface = "bold", colour = INK2) +
  annotate("text", x = X_RANGE[1], y = n_out - 1.62, label = "Jaw function", hjust = 0, vjust = 1,
           size = 2.6, fontface = "bold", colour = INK2) +
  # text column (Model 4)
  geom_text(data = lab4, aes(x = X_TXT, y = as.numeric(Outcome), label = txt), parse = TRUE, hjust = 0, size = 2.65, colour = INK) +
  geom_text(data = lab4, aes(x = X_Q, y = as.numeric(Outcome), label = qtxt,
                             fontface = ifelse(q_FDR < 0.05, "bold", "plain")),
            hjust = 1, size = 2.65, colour = INK) +
  annotate("text", x = X_TXT, y = n_out + 0.75, label = "Model 4, estimate (95% CI)", hjust = 0, size = 2.65, fontface = "bold", colour = INK) +
  annotate("text", x = X_Q,  y = n_out + 0.75, label = "q", hjust = 1, size = 2.65, fontface = "bold.italic", colour = INK) +
  # arrows under the axis
  annotate("text", x = -0.01, y = 0.25, label = "\u2190 Lower in irregular", hjust = 1, size = 2.65, colour = INK2) +
  annotate("text", x =  0.01, y = 0.25, label = "Higher in irregular \u2192", hjust = 0, size = 2.65, colour = INK) +
  scale_shape_manual(values = MOD_SHAPE, name = NULL) +
  scale_colour_manual(values = c(`TRUE` = IRR, `FALSE` = NS), guide = "none") +
  scale_fill_manual(values = c(`TRUE` = IRR, `FALSE` = "white"), guide = "none") +
  scale_y_continuous(breaks = seq_len(n_out), labels = rev(OUT_ORDER), expand = c(0, 0)) +
  scale_x_continuous(breaks = seq(-0.1, 0.3, 0.1), labels = label_number(accuracy = 0.1)) +
  coord_cartesian(xlim = c(X_RANGE[1], X_Q), ylim = c(0.05, n_out + 0.95), clip = "off") +
  labs(x = "Standardized effect, irregular vs regular sleepers (SD units)", y = NULL,
       caption = paste("Linear regression with HC3 robust SE (VAS, PI, DI, CMI) or logistic regression (locking, joint noise);",
                       "q: Benjamini\u2013Hochberg FDR across six outcomes within each model.", sep = "\n")) +
  guides(shape = guide_legend(override.aes = list(fill = "white", colour = INK), nrow = 1)) +
  theme_fig() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.y = element_text(colour = INK, face = "bold"),
        axis.line.x = element_line(colour = INK2),
        legend.position = "bottom", legend.margin = margin(-2, 0, 0, 0),
        plot.margin = margin(4, 6, 2, 2))

# =============================================================================
# (b) Equivalence test for pain: two one-sided tests (TOST)
#     90% CI of the Model 1-4 estimate vs equivalence margins.
#     VAS: 1 point (minimal clinically important difference) and 0.2 SD (small effect, Cohen 1988)
#     PI : 0.2 SD
# =============================================================================
eqv <- read_excel(xlsx, sheet = "Fig2b_equivalence") %>%
  mutate(Outcome = factor(Outcome, levels = c("VAS", "PI"),
                          labels = c('bold("VAS (")*bolditalic(B)*bold(", 0–10)")', 'bold("PI (")*bolditalic(B)*bold(", 0–1)")')),
         Model = factor(Model, levels = names(MOD_SHAPE)))
pts  <- eqv %>% distinct(Outcome, Model, B, CI90_low, CI90_high)
mrg  <- eqv %>% distinct(Outcome, Margin_type, Margin) %>%
  mutate(Margin_type = factor(Margin_type, levels = c("MCID 1 point", "0.2 SD")))
ptxt <- eqv %>% group_by(Outcome, Model) %>%
  summarise(lab = paste0("p[TOST] ", paste(ifelse(p_TOST < 0.001, "< 0.001", sprintf("= %.3f", p_TOST)), collapse = " / ")),
            .groups = "drop")
EQ_FILL <- c("MCID 1 point" = "#EEF1F6", "0.2 SD" = "#D6DEEA")

pB <- ggplot(pts, aes(y = Model)) +
  geom_rect(data = mrg, aes(xmin = -Margin, xmax = Margin, ymin = -Inf, ymax = Inf, fill = Margin_type),
            inherit.aes = FALSE) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_errorbarh(aes(xmin = CI90_low, xmax = CI90_high), height = 0, linewidth = 0.9, colour = NS) +
  geom_point(aes(x = B, shape = Model), colour = NS, fill = "white", size = 2.4, stroke = 0.9) +
  geom_text(data = ptxt, aes(x = Inf, y = Model, label = lab), hjust = 1.03, size = 2.5, colour = INK) +
  facet_wrap(~Outcome, nrow = 1, scales = "free_x", labeller = label_parsed) +
  scale_shape_manual(values = MOD_SHAPE, guide = "none") +
  scale_fill_manual(values = EQ_FILL, name = "Equivalence margin",
                    labels = c("MCID 1 point" = "±1 point (MCID)", "0.2 SD" = "±0.2 SD")) +
  scale_y_discrete(limits = rev(names(MOD_SHAPE))) +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.9))) +
  labs(x = "Irregular vs regular sleepers (90% CI)", y = NULL,
       caption = paste("Two one-sided tests (TOST); 90% CI entirely within the margin indicates equivalence.",
                       "p[TOST]: VAS, ±1 point / ±0.2 SD; PI, ±0.2 SD.", sep = "\n")) +
  theme_fig() + theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                      legend.position = "top", legend.justification = "left", legend.key.size = unit(3, "mm"),
                      legend.margin = margin(0, 0, -4, 0), panel.spacing.x = unit(6, "mm"))

# =============================================================================
# Assemble
# =============================================================================
fig2 <- pA / pB + plot_layout(heights = c(1.5, 0.8)) + plot_annotation(tag_levels = "a") &
  theme(plot.caption = element_text(size = 7.5, colour = INK, hjust = 0))

ggsave(file.path(out_dir, "Figure2.png"),  fig2, width = 180, height = 175, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "Figure2.tiff"), fig2, width = 180, height = 175, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "Figure2.pdf"),  fig2, width = 180, height = 175, units = "mm", device = cairo_pdf)
