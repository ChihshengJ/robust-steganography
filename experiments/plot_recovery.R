library(ggplot2)
library(dplyr)
library(tidyr)

df <- read.csv("recovery_results.csv", check.names = FALSE)

attacks <- c("local_paraphrase", "synonym", "global_backtranslation", "global_paraphrase")

filtered <- df %>%
  filter(attack_label %in% attacks) %>%
  filter(capacity < 20) %>%
  filter(
    (attack_label == "local_paraphrase" & tampering_level == 0.5) |
    (attack_label == "synonym"          & tampering_level == 0.5) |
    (attack_label == "global_backtranslation") |
    (attack_label == "global_paraphrase")
  )

long <- filtered %>%
  select(system, attack_label, capacity,
         `bit-wise_accuracy`, perfect_stego_rate) %>%
  pivot_longer(
    cols = c(`bit-wise_accuracy`, perfect_stego_rate),
    names_to = "metric",
    values_to = "Accuracy"
  )

long$metric <- factor(long$metric,
                      levels = c("bit-wise_accuracy", "perfect_stego_rate"),
                      labels = c("bitwise", "perfect"))
long$attack_label <- factor(
  long$attack_label,
  levels = c("local_paraphrase", "global_paraphrase",
             "synonym", "global_backtranslation")
)

long$system <- factor(long$system,
                      levels = c("topicqa", "story", "litreview", "discop"),
                      labels = c("QA", "SG", "LR", "Discop"))

attack_labels <- c(
  "local_paraphrase"       = "P (local, p=0.5)",
  "synonym"                = "SS (local, p=0.5)",
  "global_backtranslation" = "RTT (global)",
  "global_paraphrase"      = "P (global)"
)

colors <- c(
  "synonym"       = "#BB5C33FF",
  "global_paraphrase"                = "#15649CFF",
  "global_backtranslation" = "#E3D6BBFF",
  "local_paraphrase"      = "#7296B8FF"
)

p <- ggplot(long, aes(x = capacity, y = Accuracy,
                      color = attack_label, group = attack_label)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3) +
  facet_grid(metric ~ system, scales = "free_x") +
  scale_x_continuous(breaks = sort(unique(long$capacity))) +
  scale_y_continuous(breaks = c(0.2, 0.4, 0.6, 0.8, 1.0)) +
  scale_color_manual(values = colors, labels = attack_labels) +
  labs(x = "Message Length", y = "Accuracy", color = "attack") +
  theme_bw(base_size = 20) +
  theme(
    text             = element_text(face = "bold"),
    axis.title       = element_text(face = "bold", size = 22),
    axis.text        = element_text(face = "bold", size = 18),
    strip.text       = element_text(face = "bold", size = 20),
    legend.title     = element_text(face = "bold", size = 20),
    legend.text      = element_text(face = "bold", size = 18),
    legend.position  = "bottom",
    legend.margin    = margin(t = -10, b = 0),
    legend.box.margin = margin(t = -10, b = 0)
  )

ggsave("recovery_plot.png", p, width = 12, height = 6, dpi = 300)
