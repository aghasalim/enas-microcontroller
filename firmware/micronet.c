#include "micronet.h"
#include "generated/micronet_arch.h"

#include <string.h>

/* ReLU6 with a leak of 2^-shift.
 *
 * Matches torch.clamp(x, 0, 6) + 2^-s * torch.clamp(x, max=0): for a positive
 * input the second term is zero, for a negative input the first is. Written as
 * a branch here because the leak is a multiply in float. On the device it is a
 * right shift, which is the reason the activation was chosen. */
static float act_p2(float x, int shift)
{
    if (x >= 0.0f)
        return x < 6.0f ? x : 6.0f;

    float scale = 1.0f;
    for (int i = 0; i < shift; i++)
        scale *= 0.5f;
    return x * scale;
}

static void run_conv(const MicroOp *o, const float *w, const float *in, float *out)
{
    const float *kern = w + o->w_off;
    const float *bias = w + o->b_off;
    const int in_per_g = o->in_c / o->groups;
    const int out_per_g = o->out_c / o->groups;

    for (int oc = 0; oc < o->out_c; oc++) {
        const int g = oc / out_per_g;
        for (int oy = 0; oy < o->out_h; oy++) {
            for (int ox = 0; ox < o->out_w; ox++) {
                float acc = bias[oc];
                for (int ic = 0; ic < in_per_g; ic++) {
                    const int src_c = g * in_per_g + ic;
                    for (int ky = 0; ky < o->kh; ky++) {
                        const int iy = oy * o->sh + ky - o->ph;
                        if (iy < 0 || iy >= o->in_h)
                            continue;
                        for (int kx = 0; kx < o->kw; kx++) {
                            const int ix = ox * o->sw + kx - o->pw;
                            if (ix < 0 || ix >= o->in_w)
                                continue;
                            /* torch weight layout: [out_c][in_c/groups][kh][kw] */
                            acc += in[(src_c * o->in_h + iy) * o->in_w + ix]
                                 * kern[((oc * in_per_g + ic) * o->kh + ky) * o->kw + kx];
                        }
                    }
                }
                out[(oc * o->out_h + oy) * o->out_w + ox] = acc;
            }
        }
    }
}

/* Zero parameter spatial mixing: five channel groups, four of them rolled one
 * pixel in a different direction, the fifth left as a centre tap. The roll
 * wraps, matching torch.roll rather than a shift that pads with zeros. */
static void run_shift(const MicroOp *o, const float *in, float *out)
{
    const int g = o->extra;      /* channels / 5 */
    const int H = o->in_h, W = o->in_w;

    for (int c = 0; c < o->in_c; c++) {
        const float *src = in + (size_t)c * H * W;
        float *dst = out + (size_t)c * H * W;
        int dy = 0, dx = 0;
        if (c < g)              dy = 1;    /* roll -1 on dim 2: out[y] = in[y+1] */
        else if (c < 2 * g)     dy = -1;
        else if (c < 3 * g)     dx = 1;    /* roll -1 on dim 3 */
        else if (c < 4 * g)     dx = -1;

        for (int y = 0; y < H; y++) {
            const int sy = ((y + dy) % H + H) % H;
            for (int x = 0; x < W; x++) {
                const int sx = ((x + dx) % W + W) % W;
                dst[y * W + x] = src[sy * W + sx];
            }
        }
    }
}

/* Residual with no projection: average pool in space if the block strided,
 * then gather channels through the exported index map. No weights either way. */
static void run_res_add(const MicroOp *o, const int *indices, const float *res,
                        float *out)
{
    const int s = o->sh;
    const int *idx = indices + o->extra;

    for (int oc = 0; oc < o->out_c; oc++) {
        const float *src = res + (size_t)idx[oc] * o->in_h * o->in_w;
        float *dst = out + (size_t)oc * o->out_h * o->out_w;
        for (int oy = 0; oy < o->out_h; oy++) {
            for (int ox = 0; ox < o->out_w; ox++) {
                float v;
                if (s > 1) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < s; ky++)
                        for (int kx = 0; kx < s; kx++)
                            acc += src[(oy * s + ky) * o->in_w + ox * s + kx];
                    v = acc / (float)(s * s);
                } else {
                    v = src[oy * o->in_w + ox];
                }
                dst[oy * o->out_w + ox] += v;
            }
        }
    }
}

size_t micronet_scratch_bytes(void)
{
    return (size_t)3 * MICRONET_PEAK * sizeof(float);
}

void micronet_forward(const MicroNet *net, const float *input, float *logits)
{
    float *a = net->scratch;
    float *b = a + MICRONET_PEAK;
    float *res = b + MICRONET_PEAK;

    memcpy(a, input, (size_t)3 * 32 * 32 * sizeof(float));

    for (int i = 0; i < MICRONET_N_OPS; i++) {
        const MicroOp *o = &MICRONET_OPS[i];
        const size_t n_in = (size_t)o->in_c * o->in_h * o->in_w;
        const size_t n_out = (size_t)o->out_c * o->out_h * o->out_w;

        switch (o->op) {
        case OP_CONV:
            run_conv(o, net->weights, a, b);
            { float *t = a; a = b; b = t; }
            break;

        case OP_ACT:
            for (size_t k = 0; k < n_in; k++)
                a[k] = act_p2(a[k], o->extra);
            break;

        case OP_SHIFT:
            run_shift(o, a, b);
            { float *t = a; a = b; b = t; }
            break;

        case OP_RES_SAVE:
            memcpy(res, a, n_in * sizeof(float));
            break;

        case OP_RES_ADD:
            run_res_add(o, net->indices, res, a);
            break;

        case OP_POOL: {
            const size_t hw = (size_t)o->in_h * o->in_w;
            for (int c = 0; c < o->in_c; c++) {
                float acc = 0.0f;
                for (size_t k = 0; k < hw; k++)
                    acc += a[c * hw + k];
                b[c] = acc / (float)hw;
            }
            { float *t = a; a = b; b = t; }
            break;
        }

        case OP_LINEAR: {
            const float *w = net->weights + o->w_off;
            const float *bias = net->weights + o->b_off;
            for (int oc = 0; oc < o->out_c; oc++) {
                float acc = bias[oc];
                for (int ic = 0; ic < o->in_c; ic++)
                    acc += a[ic] * w[oc * o->in_c + ic];
                b[oc] = acc;
            }
            { float *t = a; a = b; b = t; }
            break;
        }

        default:
            return;   /* unknown op: refuse to produce a number rather than guess */
        }
        (void)n_out;
    }

    memcpy(logits, a, MICRONET_CLASSES * sizeof(float));
}
