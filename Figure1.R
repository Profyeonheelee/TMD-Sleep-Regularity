# =============================================================================
# Figure 1. Study cohort and 24-h sleep rhythm of regular vs irregular sleepers
#   a  Flow diagram
#   b  Population actogram (one row = one patient)
#   c  Proportion of patients asleep across the 24-h clock
#   d  Bedtime, mid-sleep and wake time (median, IQR)
# Clock scale in the data: 24 = midnight, values > 24 = next morning (e.g. 31 = 07:00)
# =============================================================================

library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(scales)

xlsx    <- file.path("results", "Figure_data.xlsx")
out_dir <- "figures"

# ---- palette and theme ---------------------------------------------------------
REG <- "#0047FF"; IRR <- "#F37021"; INK <- "#0b0b0b"; INK2 <- "#0b0b0b"; GRID <- "#e4e3df"
REG_DK <- "#0030B0"; IRR_DK <- "#B84A0C"
GCOL <- c("Regular sleepers" = REG, "Irregular sleepers" = IRR)
theme_fig <- function(base = 8.5) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(axis.line = element_line(colour = INK2, linewidth = 0.35),
          axis.ticks = element_line(colour = INK2, linewidth = 0.35),
          axis.text = element_text(colour = INK, size = 7.5), legend.text = element_text(colour = INK, size = 7.5), axis.title = element_text(colour = INK),
          panel.grid.major.x = element_line(colour = GRID, linewidth = 0.35),
          plot.title = element_text(face = "bold", size = base + 0.5, colour = INK),
          plot.tag = element_text(face = "bold", size = 13),
          legend.key.height = unit(3, "mm"), strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0))
}
clock_lab <- function(x) sprintf("%02d", as.integer(round(x)) %% 24)
hhmm      <- function(x) { x <- x %% 24; sprintf("%02d:%02d", floor(x), round((x %% 1) * 60) %% 60) }
X_LIM <- c(19, 37); X_BRK <- seq(20, 36, 2)

# ---- data --------------------------------------------------------------------
flow <- read_excel(xlsx, sheet = "Fig1a_flow")
n_of <- function(i) format(flow$n[i], big.mark = ",")

act <- read_excel(xlsx, sheet = "Fig1b_actogram") %>%
  filter(!is.na(bed_lo), !is.na(wake_lo)) %>%
  mutate(bed_hi  = coalesce(bed_hi,  bed_lo),
         wake_hi = coalesce(wake_hi, wake_lo),
         bed_mid  = (bed_lo + bed_hi) / 2,
         wake_mid = (wake_lo + wake_hi) / 2,
         mid      = (bed_mid + wake_mid) / 2,
         Group    = factor(Group, levels = names(GCOL)))

# =============================================================================
# (a) Flow diagram
# =============================================================================
bx <- tribble(
  ~id, ~x,  ~y,   ~w,  ~h,  ~label,                                                                                   ~fill,     ~edge, ~bold,
  1,   5,   19.0, 9.6, 1.7, paste0("First-visit records, 2020\u20132024\n(n = ", n_of(1), ")"), "white",   INK2,  FALSE,
  2,   5,   15.9, 9.6, 1.7, paste0("Unique patients (n = ", n_of(3), ")\nExcluded: 101 patients with repeat visits"),        "#f6f5f2", INK2,  FALSE,
  3,   5,   12.8, 9.6, 1.7, paste0("Patients, 2020\u20132023 (n = ", n_of(5), ")\nExcluded: 2024 entries (n = ", n_of(4), ")"), "#f6f5f2", INK2, FALSE,
  4,   5,    9.7, 9.6, 1.7, paste0("Analytic cohort (n = ", n_of(7), ")\nExcluded: undeterminable schedule (n = ", n_of(6), ")"), "white", INK, FALSE,
  5,   2.6,  5.9, 4.4, 2.0, paste0("Regular sleepers\nn = ", n_of(8)),                                    "#E8EEFF", REG_DK, TRUE,
  6,   7.4,  5.9, 4.4, 2.0, paste0("Irregular sleepers\nn = ", n_of(9)),                                "#FEF0E7", IRR_DK, TRUE,
  7,   7.4,  2.0, 4.4, 2.6, paste0("Variable timing\nn = ", n_of(10), "\n\nNo fixed schedule\nn = ", n_of(11)), "white", IRR_DK, FALSE)
ar <- tribble(~x, ~y, ~xend, ~yend,
              5, 18.2, 5, 16.7,  5, 15.1, 5, 13.6,  5, 12.0, 5, 10.5,
              4.0, 8.9, 2.6, 6.9,  6.0, 8.9, 7.4, 6.9,  7.4, 4.9, 7.4, 3.3)

pA <- ggplot() +
  geom_rect(data = bx, aes(xmin = x - w/2, xmax = x + w/2, ymin = y - h/2, ymax = y + h/2),
            fill = bx$fill, colour = bx$edge, linewidth = 0.5) +
  geom_text(data = bx, aes(x, y, label = label), size = 2.65, lineheight = 0.95, colour = INK,
            fontface = ifelse(bx$bold, "bold", "plain")) +
  geom_segment(data = ar, aes(x, y, xend = xend, yend = yend), colour = INK2, linewidth = 0.4,
               arrow = arrow(length = unit(1.6, "mm"), type = "closed")) +
  coord_cartesian(xlim = c(0, 10), ylim = c(0.4, 20), expand = FALSE) +
  theme_void() + theme(plot.tag = element_text(face = "bold", size = 13))

# =============================================================================
# (b) Population actogram: random 160 patients per group, sorted by mid-sleep
# =============================================================================
set.seed(4)
act_s <- act %>% group_by(Group) %>% slice_sample(n = 160) %>%
  arrange(mid, .by_group = TRUE) %>% mutate(row = row_number()) %>% ungroup()

pB <- ggplot(act_s) +
  geom_vline(xintercept = 24, colour = INK2, linewidth = 0.35, linetype = "22") +
  # light: reported range of bedtime / wake time
  geom_segment(aes(x = bed_lo,  xend = bed_hi,  y = row, yend = row, colour = Group), linewidth = 0.9, alpha = 0.30) +
  geom_segment(aes(x = wake_lo, xend = wake_hi, y = row, yend = row, colour = Group), linewidth = 0.9, alpha = 0.30) +
  # dark: core sleep period
  geom_segment(aes(x = bed_hi, xend = wake_lo, y = row, yend = row, colour = Group), linewidth = 0.9) +
  facet_wrap(~Group, ncol = 1, scales = "free_x") +
  scale_colour_manual(values = GCOL, guide = "none") +
  scale_x_continuous(limits = X_LIM, breaks = X_BRK, labels = clock_lab, expand = c(0, 0)) +
  scale_y_reverse(expand = expansion(add = 2)) +
  labs(x = "Clock time (h)", y = NULL,
       caption = "Each row = one patient (random 160 per group), sorted by mid-sleep.\nDark: reported sleep period; light: reported bedtime or wake-time range.") +
  theme_fig() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.line.y = element_blank(),
        plot.caption = element_text(size = 7.5, colour = INK, hjust = 1),
        panel.spacing.y = unit(4, "mm"))

# =============================================================================
# (c) % of patients asleep across the 24-h clock (all patients)
# =============================================================================
grid_t <- seq(X_LIM[1], X_LIM[2], by = 0.25)
asleep <- act %>% group_by(Group) %>%
  reframe(t = grid_t, pct = sapply(grid_t, function(tt) mean(bed_mid <= tt & wake_mid > tt)) * 100)

pC <- ggplot(asleep, aes(t, pct, colour = Group)) +
  geom_vline(xintercept = 24, colour = INK2, linewidth = 0.35, linetype = "22") +
  geom_line(linewidth = 0.9) +
  annotate("text", x = 23.4, y = 82, label = "Regular", colour = REG, size = 2.6, hjust = 1, fontface = "bold") +
  annotate("text", x = 31.9, y = 66, label = "Irregular", colour = IRR, size = 2.6, hjust = 0, fontface = "bold") +
  scale_colour_manual(values = GCOL, name = NULL) +
  scale_x_continuous(limits = X_LIM, breaks = X_BRK, labels = clock_lab, expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
  labs(x = "Clock time (h)", y = "Patients asleep (%)") +
  theme_fig() + theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.35),
                      legend.position = "none")

# =============================================================================
# (d) Bedtime, mid-sleep, wake time: median and IQR
# =============================================================================
tim <- act %>%
  transmute(Group, Bedtime = bed_mid, `Mid-sleep` = mid, `Wake time` = wake_mid) %>%
  pivot_longer(-Group, names_to = "Marker", values_to = "h") %>%
  group_by(Group, Marker) %>%
  summarise(q1 = quantile(h, .25, na.rm = TRUE), med = median(h, na.rm = TRUE),
            q3 = quantile(h, .75, na.rm = TRUE), .groups = "drop") %>%
  mutate(Marker = factor(Marker, levels = c("Wake time", "Mid-sleep", "Bedtime")))

f1d  <- read_excel(xlsx, sheet = "Fig1d_tests")
plab <- setNames(paste0(f1d$Marker, "\n", ifelse(f1d$p < 0.001, "p < 0.001", sprintf("p = %.3f", f1d$p))), f1d$Marker)
pD <- ggplot(tim, aes(y = Marker, colour = Group, group = Group)) +
  geom_linerange(aes(xmin = q1, xmax = q3), linewidth = 1.6, position = position_dodge(width = 0.45)) +
  geom_point(aes(x = med), size = 2.6, shape = 21, fill = "white", stroke = 1.2, position = position_dodge(width = 0.45)) +
  geom_text(aes(x = q3, label = hhmm(med)), hjust = -0.3, size = 2.65, colour = INK2,
            position = position_dodge(width = 0.45), show.legend = FALSE) +
  scale_colour_manual(values = GCOL, name = NULL) +
  scale_x_continuous(limits = c(21, 36), breaks = seq(22, 34, 2), labels = clock_lab) +
  scale_y_discrete(labels = plab) +
  labs(x = "Clock time (h), median and IQR", y = NULL, caption = "p: Welch t test") +
  theme_fig() + theme(legend.position = "none", axis.ticks.y = element_blank(),
                      plot.caption = element_text(size = 7.5, colour = INK, hjust = 1))

# =============================================================================
# Assemble
# =============================================================================
fig1 <- pA + (pB / (pC | pD) + plot_layout(heights = c(1.55, 0.8))) +
  plot_layout(widths = c(1.15, 2)) +
  plot_annotation(tag_levels = list(c("a", "b", "c", "d")))

ggsave(file.path(out_dir, "Figure1.png"),  fig1, width = 180, height = 185, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(out_dir, "Figure1.tiff"), fig1, width = 180, height = 185, units = "mm", dpi = 300, bg = "white", compression = "lzw")
ggsave(file.path(out_dir, "Figure1.pdf"),  fig1, width = 180, height = 185, units = "mm", device = cairo_pdf)
