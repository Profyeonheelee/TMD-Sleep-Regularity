# =============================================================================
# Figure 3. The association is independent of sleep quality, sleep timing,
#           oral behaviours and sleep bruxism, and is not mediated by them
#   a  Bootstrap distributions (violins) of the irregular-vs-regular estimate
#      across five sequential adjustment steps (same patients at every step)
#   b  Prevalence of clenching, bruxism and sleep bruxism by group
#   c  Bootstrap distributions of the total effect vs indirect effects (DI, CMI)
# Symbol rule shared with Figure 2: circle/square/diamond/triangle = Models 1-4
# (inverted triangle = step 5). Square in panel c = Model 2 total effect.
# Indirect effects carry no model symbol (median shown as a bar).
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ----------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; GRID <- "#e4e3df"
NS <- "#4a4a4a"; NS_FILL <- "#d9d8d4"; IRR_FILL <- "#F9C4A2"
SIG_COL  <- c(`TRUE` = IRR, `FALSE` = NS)
SIG_FILL <- c(`TRUE` = IRR_FILL, `FALSE` = NS_FILL)
GCOL <- c("Regular sleepers" = REG, "Irregular sleepers" = IRR)
theme_fig <- function(base = 8.5) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(axis.line = element_line(colour = INK, linewidth = 0.35),
          axis.ticks = element_line(colour = INK, linewidth = 0.35),
          axis.text = element_text(colour = INK, size = 7.5), legend.text = element_text(colour = INK, size = 7.5),
          axis.title = element_text(colour = INK, size = 8),
          plot.title = element_text(size = 8.5, face = "bold", colour = INK),
          plot.tag = element_text(face = "bold", size = 13),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, colour = INK, size = 8.5))
}
fmt_p <- function(p) ifelse(p < 0.001, "p < 0.001", sprintf("p = %.2f", p))

# =============================================================================
# (a) Sequential adjustment: bootstrap violins + point estimate and 95% CI
# =============================================================================
STEP_SHAPE <- c(21, 22, 23, 24, 25)
STEP_KEY <- c("1" = "Base model", "2" = "+ Sleep quality", "3" = "+ Timing & duration",
              "4" = "+ Oral behaviours, stress, OSA risk",
              "5" = "+ Clinician-note sleep bruxism & pain-disrupted sleep")
OUT_LAB <- list(DI = quote(bold("DI (")*bolditalic(B)*bold(")")), CMI = quote(bold("CMI (")*bolditalic(B)*bold(")")),
                Locking = quote(bold("Locking (OR)")), VAS = quote(bold("VAS (")*bolditalic(B)*bold(")")))

pt <- read_excel(xlsx, sheet = "Fig3a_adjustment") %>%
  mutate(sig = p < 0.05, Step = factor(Step, levels = 1:5), ref = ifelse(Scale == "OR", 1, 0))
bs <- read_excel(xlsx, sheet = "Fig3a_bootstrap") %>% filter(!is.na(Estimate)) %>%
  mutate(Step = factor(Step, levels = 1:5)) %>%
  left_join(pt %>% select(Outcome, Step, sig), by = c("Outcome", "Step"))

panel_a <- function(o, show_y = FALSE) {
  b <- filter(bs, Outcome == o); p <- filter(pt, Outcome == o)
  g <- ggplot(b, aes(x = Step, y = Estimate)) +
    geom_hline(data = p[1, ], aes(yintercept = ref), colour = INK, linewidth = 0.45, linetype = "22") +
    geom_violin(aes(fill = sig, colour = sig), width = 0.9, linewidth = 0.35, scale = "width") +
    geom_errorbar(data = p, aes(ymin = CI_low, ymax = CI_high, colour = sig, y = NULL), width = 0, linewidth = 0.8) +
    geom_point(data = p, aes(shape = Step, colour = sig), fill = "white", size = 2.3, stroke = 0.9) +
    geom_text(data = p, aes(x = Step, y = Inf, label = ifelse(p < 0.001, "<.001", sub("^0", "", sprintf("%.3f", p))),
                            fontface = ifelse(sig, "bold", "plain")), vjust = 1.3, size = 2.5, colour = INK) +
    scale_colour_manual(values = SIG_COL, guide = "none") +
    scale_fill_manual(values = SIG_FILL, guide = "none") +
    scale_shape_manual(values = STEP_SHAPE, labels = paste(names(STEP_KEY), STEP_KEY), name = NULL, drop = FALSE) +
    scale_y_continuous(expand = expansion(mult = c(0.04, 0.14))) +
    labs(title = OUT_LAB[[o]], x = "Adjustment step", y = if (show_y) "Irregular vs regular estimate" else NULL) +
    theme_fig() + theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35))
  if (o == "Locking") g <- g + scale_y_log10(breaks = c(0.8, 1, 1.25, 1.6, 2), expand = expansion(mult = c(0.04, 0.14)))
  g
}
pA <- (panel_a("DI", TRUE) + labs(tag = "a") | panel_a("CMI") | panel_a("Locking") | panel_a("VAS")) +
  plot_layout(guides = "collect") &
  guides(shape = guide_legend(ncol = 3, byrow = TRUE, override.aes = list(colour = INK, fill = "white", size = 2.4))) &
  theme(legend.position = "bottom", legend.margin = margin(-2, 0, 0, 0), legend.key.width = unit(3, "mm"))
pA <- pA + plot_annotation(caption = paste(
  "Numbers above violins: p (Wald test; linear regression with HC3 robust SE for DI, CMI, VAS; logistic regression for locking).",
  "Violins: 500 bootstrap resamples; symbols and lines: estimate and 95% CI.", sep = "\n"),
  theme = theme(plot.caption = element_text(size = 7.5, colour = INK, hjust = 0)))

# =============================================================================
# (b) Prevalence of oral behaviours by group (no model symbols: raw proportions)
# =============================================================================
pr <- read_excel(xlsx, sheet = "Fig3b_prevalence") %>%
  mutate(Behaviour = recode(Behaviour, "Sleep bruxism (clinician notes)" = "Sleep bruxism\n(clinician notes)"),
         Behaviour = factor(Behaviour, levels = c("Clenching", "Bruxism", "Sleep bruxism\n(clinician notes)")),
         Group = factor(Group, levels = names(GCOL)))
pr_p <- pr %>% group_by(Behaviour) %>% summarise(top = max(Pct), p = first(p_chi2), .groups = "drop")
YMAX <- ceiling(max(pr$Pct) / 10) * 10 + 15

pB <- ggplot(pr, aes(x = Behaviour, y = Pct, fill = Group)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72, colour = NA) +
  geom_text(aes(label = sprintf("%.1f", Pct)), position = position_dodge(width = 0.78),
            vjust = -0.4, size = 2.65, colour = INK) +
  geom_segment(data = pr_p, aes(x = as.numeric(Behaviour) - 0.2, xend = as.numeric(Behaviour) + 0.2,
                                y = top + 6.5, yend = top + 6.5), inherit.aes = FALSE, colour = INK, linewidth = 0.35) +
  geom_text(data = pr_p, aes(x = Behaviour, y = top + 9.5, label = fmt_p(p)), inherit.aes = FALSE,
            size = 2.65, colour = INK) +
  scale_fill_manual(values = GCOL, name = NULL) +
  scale_y_continuous(limits = c(0, YMAX), breaks = seq(0, YMAX, 20), expand = c(0, 0)) +
  labs(x = NULL, y = "Patients (%)", title = "Do irregular sleepers clench or grind more?", caption = "p: \u03c7\u00b2 test") +
  theme_fig() +
  theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35),
        plot.caption = element_text(size = 7.5, colour = INK, hjust = 0),
        legend.position = "top", legend.justification = "left", legend.margin = margin(0, 0, -6, 0),
        legend.key.size = unit(3, "mm"))

# =============================================================================
# (c) Total vs indirect effects: bootstrap violins
# =============================================================================
PATHS <- c("Total effect", "via clenching", "via bruxism", "via sleep bruxism")
mb <- read_excel(xlsx, sheet = "Fig3c_mediation_boot") %>%
  mutate(Path = factor(Path, levels = PATHS), Outcome = factor(Outcome, levels = c("DI", "CMI")),
         sig = Path == "Total effect")
# bootstrap p: twice the smaller tail proportion on either side of 0
bp <- mb %>% group_by(Outcome, Path) %>%
  summarise(p = min(1, 2 * min(mean(Effect <= 0), mean(Effect >= 0))), .groups = "drop")
pct <- read_excel(xlsx, sheet = "Fig3c_mediation") %>%
  transmute(Outcome = factor(Outcome, levels = c("DI", "CMI")),
            Path = factor(recode(Mediator, "Clenching" = "via clenching", "Bruxism" = "via bruxism",
                                 "Sleep bruxism (clinician notes)" = "via sleep bruxism"), levels = PATHS),
            pm = Pct_mediated) %>%
  bind_rows(tibble(Outcome = factor(c("DI", "CMI"), levels = c("DI", "CMI")),
                   Path = factor("Total effect", levels = PATHS), pm = NA_real_)) %>%
  left_join(bp, by = c("Outcome", "Path")) %>%
  mutate(lab = ifelse(is.na(pm), sprintf("p = %.3f", p), sprintf("%.1f%%, p = %.2f", pm, p)))
tot_pt <- mb %>% filter(sig) %>% group_by(Outcome, Path) %>% summarise(Effect = median(Effect), .groups = "drop")
ind_md <- mb %>% filter(!sig) %>% group_by(Outcome, Path) %>% summarise(Effect = median(Effect), .groups = "drop")

mb  <- mb  %>% mutate(Path = factor(Path, levels = rev(PATHS)))
pct <- pct %>% mutate(Path = factor(Path, levels = rev(PATHS)))
tot_pt <- tot_pt %>% mutate(Path = factor(Path, levels = rev(PATHS)))
ind_md <- ind_md %>% mutate(Path = factor(Path, levels = rev(PATHS)))
pC <- ggplot(mb, aes(y = Path, x = Effect)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_violin(aes(fill = sig, colour = sig), width = 0.95, linewidth = 0.35, scale = "width") +
  geom_point(data = tot_pt, colour = IRR, shape = 22, fill = "white", size = 2.4, stroke = 0.9) +
  geom_errorbarh(data = ind_md, aes(xmin = Effect, xmax = Effect), colour = NS, height = 0.45, linewidth = 0.9) +
  geom_text(data = pct, aes(y = Path, x = 0.095, label = lab), hjust = 1, size = 2.65, colour = INK) +
  facet_wrap(~Outcome, ncol = 1) +
  scale_colour_manual(values = SIG_COL, guide = "none") +
  scale_fill_manual(values = SIG_FILL, guide = "none") +
  scale_y_discrete(labels = c("Total effect" = "Total effect", "via clenching" = "via clenching",
                              "via bruxism" = "via bruxism", "via sleep bruxism" = "via sleep bruxism")) +
  scale_x_continuous(limits = c(-0.01, 0.095), breaks = c(0, 0.04), labels = c("0", "0.04")) +
  labs(y = NULL, x = quote("Effect of irregular sleep ("*italic(B)*")"),
       title = "Is the effect carried by oral behaviours?", subtitle = "Right: % mediated, bootstrap p",
       caption = "Product of coefficients (Model 2 covariates);\n1,000 bootstrap resamples.") +
  theme_fig() + theme(plot.caption = element_text(size = 7.5, colour = INK, hjust = 0), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
                      axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                      plot.subtitle = element_text(size = 7.5, colour = INK),
                      panel.spacing.y = unit(3, "mm"))

# =============================================================================
# Assemble
# =============================================================================
bottom <- wrap_elements(full = (pB + labs(tag = "b")) | (pC + labs(tag = "c")) + plot_layout(widths = c(1, 1.1)))
fig3 <- wrap_elements(full = pA) / bottom + plot_layout(heights = c(1.1, 1))

ggsave(file.path(out_dir, "Figure3.png"),  fig3, width = 180, height = 195, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "Figure3.tiff"), fig3, width = 180, height = 195, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "Figure3.pdf"),  fig3, width = 180, height = 195, units = "mm", device = cairo_pdf)
