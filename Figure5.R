# =============================================================================
# Figure 5. Pain and jaw function are driven by different predictors (gated attention Transformer)
#   a  Share of mean |SHAP| by predictor domain
#   b  SHAP beeswarm for DI vs VAS (top 10 features)
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ----------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; GRID <- "#e4e3df"
NS <- "#4a4a4a"
SIG_COL <- c(`TRUE` = IRR, `FALSE` = NS)
theme_fig <- function(base = 8.5) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(axis.line = element_line(colour = INK, linewidth = 0.35),
          axis.ticks = element_line(colour = INK, linewidth = 0.35),
          axis.text = element_text(colour = INK, size = 7.5), legend.text = element_text(colour = INK, size = 7.5),
          legend.title = element_text(colour = INK, size = 7.5),
          axis.title = element_text(colour = INK, size = 8),
          plot.title = element_text(size = 8.5, face = "bold", colour = INK),
          plot.caption = element_text(size = 7.5, colour = INK, hjust = 0),
          plot.tag = element_text(face = "bold", size = 13),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, colour = INK, size = 8.5))
}
fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

# =============================================================================
# (b) Gated Transformer: share of mean |SHAP| by domain
# =============================================================================
DOM_MAP <- c("Regularity" = "Sleep rhythm", "Timing" = "Sleep rhythm", "Duration" = "Sleep rhythm",
             "Satisfaction" = "Sleep quality", "Efficiency" = "Sleep quality", "Alertness" = "Sleep quality",
             "Sleep-disordered breathing" = "OSA risk", "Clinical" = "Clinical", "Behavioral" = "Behavioural",
             "Design" = "Design / missingness", "Missing indicator" = "Design / missingness")
DOM_COL <- c("Sleep rhythm" = IRR, "Sleep quality" = "#1baf7a", "OSA risk" = "#9fd8c3",
             "Clinical" = "#4a4a4a", "Behavioural" = "#8f8d88", "Design / missingness" = "#d9d8d4")
fb <- read_excel(xlsx, sheet = "Fig5a_importance") %>%
  filter(Metric == "Mean |SHAP|") %>%
  mutate(Group = factor(DOM_MAP[Domain], levels = names(DOM_COL))) %>%
  group_by(Outcome, Group) %>% summarise(v = sum(Value), .groups = "drop") %>%
  group_by(Outcome) %>% mutate(share = v / sum(v) * 100) %>% ungroup() %>%
  mutate(Outcome = factor(Outcome, levels = rev(c("VAS", "DI", "CMI", "Locking"))))

pB <- ggplot(fb, aes(y = Outcome, x = share, fill = Group)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.3, position = position_stack(reverse = TRUE)) +
  geom_text(data = filter(fb, Group %in% c("Sleep rhythm", "Sleep quality")),
            aes(label = sprintf("%.0f%%", share)), position = position_stack(vjust = 0.5, reverse = TRUE),
            size = 2.65, colour = INK, fontface = "bold") +
  scale_fill_manual(values = DOM_COL, name = NULL) +
  scale_x_continuous(limits = c(0, 100.5), breaks = seq(0, 100, 25), expand = c(0, 0)) +
  labs(x = "Share of mean |SHAP| (%)", y = NULL, title = "Gated Transformer: what drives each outcome?",
       caption = "Mean |SHAP| (GradientExplainer, 600 patients), summed within domains.") +
  guides(fill = guide_legend(ncol = 1)) +
  theme_fig() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.y = element_text(face = "bold", size = 8),
        legend.position = "right", legend.direction = "vertical", legend.key.size = unit(3, "mm"), legend.margin = margin(-2, 0, 0, 0))

# =============================================================================
# (c) SHAP beeswarm: DI vs VAS, top 10 features each
# =============================================================================
sh <- read_excel(xlsx, sheet = "Fig5b_shap")
top <- sh %>% group_by(Outcome, Label) %>% summarise(m = mean(abs(SHAP)), .groups = "drop") %>%
  group_by(Outcome) %>% slice_max(m, n = 10) %>% ungroup()
fc <- sh %>% semi_join(top, by = c("Outcome", "Label")) %>%
  left_join(top, by = c("Outcome", "Label")) %>%
  mutate(key = paste(Outcome, Label, sep = "__"),
         Outcome = factor(Outcome, levels = c("DI", "VAS"),
                          labels = c("DI (jaw function)", "VAS (pain)")))
ord <- top %>% mutate(key = paste(Outcome, Label, sep = "__")) %>% arrange(Outcome, m) %>% pull(key)
fc <- fc %>% mutate(key = factor(key, levels = ord))
SLEEP_DOM <- c("Regularity", "Timing", "Duration", "Satisfaction", "Efficiency", "Alertness", "Sleep-disordered breathing")
lab_fun <- function(k) sub("^.*__", "", k)

set.seed(1)
pC <- ggplot(fc, aes(x = SHAP, y = key, colour = Feature_value_scaled)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.4, linetype = "22") +
  geom_jitter(height = 0.25, width = 0, size = 0.55, alpha = 0.8) +
  facet_wrap(~Outcome, nrow = 1, scales = "free") +
  scale_colour_gradient(low = "#F2C230", high = "#5B2BB5", name = "Feature value",
                        breaks = c(0, 1), labels = c("Low", "High")) +
  scale_y_discrete(labels = lab_fun) +
  labs(x = "SHAP value (impact on prediction)", y = NULL,
       caption = "Top 10 features per outcome by mean |SHAP| (GradientExplainer, 600 patients). Colour: feature value (min\u2013max scaled).") +
  theme_fig() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
        panel.spacing.x = unit(5, "mm"),
        legend.position = "right", legend.key.height = unit(6, "mm"), legend.key.width = unit(2.5, "mm"))

# =============================================================================
# Assemble
# =============================================================================
fig5 <- wrap_elements(full = pB + labs(tag = "a")) / wrap_elements(full = pC + labs(tag = "b")) +
  plot_layout(heights = c(0.62, 1))

ggsave(file.path(out_dir, "Figure5.png"),  fig5, width = 180, height = 160, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "Figure5.tiff"), fig5, width = 180, height = 160, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "Figure5.pdf"),  fig5, width = 180, height = 160, units = "mm", device = cairo_pdf)
