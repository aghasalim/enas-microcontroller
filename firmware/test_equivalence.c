/* Does this C forward pass produce what PyTorch produced?
 *
 * The golden file holds inputs and the logits the PyTorch model returned for
 * them, written by export/export_c.py from the unfolded model. This code runs
 * the folded op table. So a pass here checks two things at once: that the
 * batch norm folding is algebraically right, and that every kernel here agrees
 * with the torch operator it stands in for.
 */
#include "micronet.h"
#include "generated/micronet_arch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_SIZE (3 * 32 * 32)
#define TOL 1e-4f

static void *slurp(const char *path, size_t want_bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    const long have = ftell(f);
    if ((size_t)have != want_bytes) {
        fprintf(stderr, "%s: expected %zu bytes, found %ld. Re-run "
                        "export/export_c.py.\n", path, want_bytes, have);
        exit(2);
    }
    rewind(f);
    void *buf = malloc(want_bytes);
    if (!buf || fread(buf, 1, want_bytes, f) != want_bytes) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "generated";
    char path[512];

    snprintf(path, sizeof path, "%s/micronet_weights.bin", dir);
    float *weights = slurp(path, MICRONET_N_WEIGHTS * sizeof(float));

    snprintf(path, sizeof path, "%s/micronet_indices.bin", dir);
    int *indices = slurp(path, MICRONET_N_INDICES * sizeof(int));

    snprintf(path, sizeof path, "%s/micronet_golden.bin", dir);
    const size_t golden_bytes =
        (size_t)MICRONET_N_GOLDEN * (IN_SIZE + MICRONET_CLASSES) * sizeof(float);
    float *golden = slurp(path, golden_bytes);

    const float *inputs = golden;
    const float *expect = golden + (size_t)MICRONET_N_GOLDEN * IN_SIZE;

    float *scratch = malloc(micronet_scratch_bytes());
    MicroNet net = { weights, indices, scratch };

    printf("ops %d, weights %d, peak activation %d floats, scratch %.1f KB\n",
           MICRONET_N_OPS, MICRONET_N_WEIGHTS, MICRONET_PEAK,
           micronet_scratch_bytes() / 1024.0);

    float worst = 0.0f;
    int failures = 0;
    for (int n = 0; n < MICRONET_N_GOLDEN; n++) {
        float logits[MICRONET_CLASSES];
        micronet_forward(&net, inputs + (size_t)n * IN_SIZE, logits);

        float worst_here = 0.0f;
        int arg_c = 0, arg_t = 0;
        for (int k = 0; k < MICRONET_CLASSES; k++) {
            const float d = fabsf(logits[k] - expect[n * MICRONET_CLASSES + k]);
            if (d > worst_here)
                worst_here = d;
            if (logits[k] > logits[arg_c])
                arg_c = k;
            if (expect[n * MICRONET_CLASSES + k] > expect[n * MICRONET_CLASSES + arg_t])
                arg_t = k;
        }
        if (worst_here > worst)
            worst = worst_here;

        const int bad = worst_here > TOL || arg_c != arg_t;
        failures += bad;
        printf("  image %d: max |C - torch| = %.3e   argmax C %d torch %d   %s\n",
               n, worst_here, arg_c, arg_t, bad ? "FAIL" : "ok");
    }

    printf("\nworst difference over %d images: %.3e (tolerance %.0e)\n",
           MICRONET_N_GOLDEN, worst, TOL);
    if (failures) {
        printf("%d of %d images disagree with PyTorch\n", failures,
               MICRONET_N_GOLDEN);
        return 1;
    }
    printf("the C forward pass reproduces PyTorch on every golden image\n");
    return 0;
}
