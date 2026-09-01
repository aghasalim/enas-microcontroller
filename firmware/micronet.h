/* Float reference implementation of the searched network.
 *
 * This interprets the op table in generated/micronet_arch.h rather than
 * hardcoding the architecture, so re-running the search and re-exporting
 * changes the network without touching this file.
 *
 * Float, not int8. The point of this code is to establish that a plain C
 * forward pass reproduces the PyTorch one before any quantisation error is
 * introduced on top. Quantised kernels are the next step, not this step.
 */
#ifndef MICRONET_H
#define MICRONET_H

#include <stddef.h>

typedef struct {
    const float *weights;   /* folded conv and linear weights, flat */
    const int   *indices;   /* residual channel index maps */
    float       *scratch;   /* 3 * MICRONET_PEAK floats, caller owned */
} MicroNet;

/* Runs one 3x32x32 image. logits must hold MICRONET_CLASSES floats. */
void micronet_forward(const MicroNet *net, const float *input, float *logits);

/* Bytes of scratch the caller must provide. */
size_t micronet_scratch_bytes(void);

#endif
