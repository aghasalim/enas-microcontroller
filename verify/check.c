/*
 * Recompute search-log aggregates in C and check them against README.md.
 *
 * Covers: candidate count, rejected count, trained count, total training
 * minutes, best fitness/accuracy, seed-vs-best comparison, and budget.
 *
 * Build: cc -o verify/check verify/check.c -lm
 * Run:   verify/check [root]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 200
#define LINE_BUF 8192

typedef struct {
    char gen[16];
    char cand[16];
    char mutation[64];
    int  params;
    int  macs;
    int  peak_act;
    double acc;
    double fitness;
    int  deployable;
    double train_s;
    char genome[4096];
} Row;

static Row rows[MAX_ROWS];
static int nrows;

/* Read the README into a malloc'd buffer. */
static char *slurp(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* Parse one CSV field, advancing *p past the delimiter. */
static int next_field(const char **p, char *out, int cap) {
    const char *s = *p;
    int i = 0;
    if (*s == '"') {
        s++;
        while (*s) {
            if (*s == '"' && *(s+1) == '"') { if (i < cap-1) out[i++] = '"'; s += 2; continue; }
            if (*s == '"') { s++; break; }
            if (i < cap-1) out[i++] = *s;
            s++;
        }
        if (*s == ',') s++;
        else if (*s == '\n' || *s == '\r') { while (*s == '\r' || *s == '\n') s++; }
    } else {
        while (*s && *s != ',' && *s != '\n' && *s != '\r') {
            if (i < cap-1) out[i++] = *s;
            s++;
        }
        if (*s == ',') s++;
        else if (*s == '\n' || *s == '\r') { while (*s == '\r' || *s == '\n') s++; }
    }
    out[i] = '\0';
    *p = s;
    return i;
}

static void load_csv(const char *path) {
    char *data = slurp(path);
    const char *p = data;
    /* Skip header line. */
    while (*p && *p != '\n') p++;
    if (*p) p++;
    nrows = 0;
    while (*p && nrows < MAX_ROWS) {
        Row *r = &rows[nrows];
        char tmp[4096];
        next_field(&p, r->gen, sizeof(r->gen));
        next_field(&p, r->cand, sizeof(r->cand));
        next_field(&p, tmp, sizeof(tmp)); /* parent */
        next_field(&p, r->mutation, sizeof(r->mutation));
        next_field(&p, tmp, sizeof(tmp)); r->params = atoi(tmp);
        next_field(&p, tmp, sizeof(tmp)); r->macs = atoi(tmp);
        next_field(&p, tmp, sizeof(tmp)); r->peak_act = atoi(tmp);
        next_field(&p, tmp, sizeof(tmp)); r->acc = atof(tmp);
        next_field(&p, tmp, sizeof(tmp));
        if (strcmp(tmp, "-inf") == 0) r->fitness = -1e30;
        else r->fitness = atof(tmp);
        next_field(&p, tmp, sizeof(tmp)); r->deployable = atoi(tmp);
        next_field(&p, tmp, sizeof(tmp)); r->train_s = atof(tmp);
        next_field(&p, r->genome, sizeof(r->genome));
        nrows++;
    }
    free(data);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : ".";
    char path[512];

    snprintf(path, sizeof(path), "%s/results/search_log.csv", root);
    load_csv(path);

    snprintf(path, sizeof(path), "%s/README.md", root);
    char *readme = slurp(path);

    /* Compute aggregates. */
    int rejected = 0, trained = 0;
    double total_train_s = 0;
    for (int i = 0; i < nrows; i++) {
        if (!rows[i].deployable) rejected++;
        else trained++;
        total_train_s += rows[i].train_s;
    }
    int total_train_min = (int)round(total_train_s / 60.0);

    /* Best by fitness among deployable. */
    int best_i = -1;
    for (int i = 0; i < nrows; i++) {
        if (rows[i].deployable && (best_i < 0 || rows[i].fitness > rows[best_i].fitness))
            best_i = i;
    }

    Row *seed = &rows[0];
    Row *best = &rows[best_i];

    double gap_points = 100.0 * (best->acc - seed->acc);
    int param_diff = best->params - seed->params;
    double mac_pct = 100.0 * (1.0 - (double)best->macs / seed->macs);

    double best_kb = (best->params + best->peak_act) / 1024.0;
    double worst_kb = 0;
    for (int i = 0; i < nrows; i++) {
        if (!rows[i].deployable) continue;
        double kb = (rows[i].params + rows[i].peak_act) / 1024.0;
        if (kb > worst_kb) worst_kb = kb;
    }
    double headroom = 250.0 / worst_kb;

    int more_acc = 0;
    for (int i = 0; i < nrows; i++) {
        if (rows[i].deployable && rows[i].acc > best->acc) more_acc++;
    }
    int acc_rank = more_acc + 1;

    /* Build check strings. */
    char want[30][256];
    int nwant = 0;

    snprintf(want[nwant++], 256, "%d candidates", nrows);
    snprintf(want[nwant++], 256, "%d rejected", rejected);
    snprintf(want[nwant++], 256, "%d minutes of training", total_train_min);
    snprintf(want[nwant++], 256, "%d trained candidates", trained);
    snprintf(want[nwant++], 256, "**%.4f**", best->acc);
    snprintf(want[nwant++], 256, "**%.4f**", best->fitness);
    snprintf(want[nwant++], 256, "%.1f accuracy points", gap_points);
    snprintf(want[nwant++], 256, "%d more", param_diff);
    snprintf(want[nwant++], 256, "%.1f%% fewer", mac_pct);
    snprintf(want[nwant++], 256, "generation %s, candidate %s", best->gen, best->cand);
    snprintf(want[nwant++], 256, "%.1f KB", best_kb);
    snprintf(want[nwant++], 256, "%.1f KB", worst_kb);
    snprintf(want[nwant++], 256, "%.1fx", headroom);
    snprintf(want[nwant++], 256, "%dth by accuracy", acc_rank);

    int failures = 0;
    for (int i = 0; i < nwant; i++) {
        int hit = strstr(readme, want[i]) != NULL;
        printf("  %-4s %s\n", hit ? "ok" : "FAIL", want[i]);
        if (!hit) failures++;
    }

    free(readme);

    if (failures > 0) {
        printf("%d of %d figures are not in README.md as written\n", failures, nwant);
        return 1;
    }
    printf("C reproduces all %d figures from the search log\n", nwant);
    return 0;
}
