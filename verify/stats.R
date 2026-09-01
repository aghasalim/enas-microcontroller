# Redo the statistical inference in README sections 6 and 8, in base R.
#
# Those sections are the honesty of the report: the noise decomposition that
# says nearly half the ranking signal is measurement error, and the paired
# t test on the clean-split retest that says the search overstated itself by
# 1.8x. Both were computed once, in Python, by the same person who wrote the
# claim. This recomputes them from results/search_log.csv and
# results/validation.csv with R's own statistics functions, including a
# two-sided p value that nothing else in the repository checks, and requires
# every figure to appear verbatim in README.md.
#
# No packages. CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

log <- read.csv(file.path(root, "results", "search_log.csv"))
val <- read.csv(file.path(root, "results", "validation.csv"))
readme <- paste(readLines(file.path(root, "README.md"), encoding = "UTF-8", warn = FALSE),
                collapse = "\n")

ok <- log[log$deployable == 1, ]
seed <- log[1, ]
best <- ok[which.max(ok$fitness), ]

# Section 6, the noise decomposition. Accuracy on n validation images is a
# binomial proportion, so its sampling standard deviation is known in closed
# form and can be compared against the spread actually observed across
# candidates. What is left over is architecture and initialisation together.
N_VAL <- 2000
obs_sd <- sd(ok$acc)
binom_sd <- sqrt(mean(ok$acc) * (1 - mean(ok$acc)) / N_VAL)
residual_sd <- sqrt(obs_sd^2 - binom_sd^2)
pooled_se <- sqrt(seed$acc * (1 - seed$acc) / N_VAL + best$acc * (1 - best$acc) / N_VAL)
gap <- best$acc - seed$acc

# Section 6's retest. The two architectures share an initialisation seed at
# every point, so this is a paired comparison and t.test is given paired = TRUE
# rather than being handed two independent spreads.
w <- val$acc[val$arch == "winner"][order(val$seed[val$arch == "winner"])]
s <- val$acc[val$arch == "seed"][order(val$seed[val$arch == "seed"])]
d <- w - s
tt <- t.test(w, s, paired = TRUE)
se <- sd(d) / sqrt(length(d))
SEARCH_GAP <- 0.0370   # what the search reported, from the log above

want <- c(
    sprintf("%d trained candidates", nrow(ok)),
    sprintf("%.4f", obs_sd),
    sprintf("%.4f", binom_sd),
    sprintf("%.0f%%", 100 * binom_sd^2 / obs_sd^2),
    sprintf("%.4f", residual_sd),
    sprintf("%.4f", gap),
    sprintf("%.4f", pooled_se),
    sprintf("z = %.2f", gap / pooled_se),

    sprintf("%d training seeds", length(d)),
    sprintf("| baseline | %.4f |", mean(s)),
    sprintf("| %.4f |", sd(s)),
    sprintf("| winner | %.4f |", mean(w)),
    sprintf("| %.4f |", sd(w)),
    paste0("| ", paste(sprintf("%+.4f", d), collapse = " | "), " |"),
    sprintf("Mean %+.4f", mean(d)),
    sprintf("standard error %.4f", se),
    sprintf("t(%d) = %.2f", length(d) - 1, unname(tt$statistic)),
    sprintf("p = %.3f", unname(tt$p.value)),
    sprintf("ahead on %d of %d seeds", sum(d > 0), length(d)),
    sprintf("**%.1f points**", 100 * mean(d)),
    sprintf("%.1fx", SEARCH_GAP / mean(d)),
    sprintf("%.0f%% of the %.4f effect", 100 * sd(w) / mean(d), mean(d)),
    sprintf("the gap is %+.4f", min(d))
)

failures <- 0
for (x in want) {
    hit <- grepl(x, readme, fixed = TRUE)
    failures <- failures + !hit
    cat(sprintf("  %-4s %s\n", if (hit) "ok" else "FAIL", x))
}

cat(sprintf("\nt(%d) = %.4f, two-sided p = %.5f, %d of %d seeds positive\n",
            length(d) - 1, unname(tt$statistic), unname(tt$p.value),
            sum(d > 0), length(d)))

if (failures > 0) {
    cat(sprintf("%d of %d figures are not in README.md as written\n", failures, length(want)))
    quit(status = 1)
}
cat(sprintf("R reproduces all %d figures in README sections 6 and 8\n", length(want)))
