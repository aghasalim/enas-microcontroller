// Inference latency for the searched network.
//
// Builds two ways. On a host it is a benchmark with a main(), which is what
// produces the millisecond figure quoted in the README and what CI compiles.
// Under the Arduino toolchain it exposes setup() and loop() instead, so the
// same measurement can be taken on the board.
//
// The board path has not been compiled or flashed. There is no ARM toolchain
// in this project's CI and none on the machine the searches ran on, so treat
// everything under ARDUINO as unverified code that is structured to work
// rather than as a result.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "micronet.h"
#include "generated/micronet_arch.h"
}

namespace {

constexpr int kInputSize = 3 * 32 * 32;

// Weights live in flash on a device and in a file on a host. Keeping the load
// behind one function is the only place the two targets differ.
std::vector<float> load_floats(const char *path, size_t count)
{
    std::vector<float> out(count);
    FILE *f = std::fopen(path, "rb");
    if (!f || std::fread(out.data(), sizeof(float), count, f) != count) {
        std::fprintf(stderr, "cannot read %zu floats from %s\n", count, path);
        std::exit(2);
    }
    std::fclose(f);
    return out;
}

std::vector<int> load_ints(const char *path, size_t count)
{
    std::vector<int> out(count);
    FILE *f = std::fopen(path, "rb");
    if (!f || std::fread(out.data(), sizeof(int), count, f) != count) {
        std::fprintf(stderr, "cannot read %zu ints from %s\n", count, path);
        std::exit(2);
    }
    std::fclose(f);
    return out;
}

long macs_from_op_table()
{
    long total = 0;
    for (int i = 0; i < MICRONET_N_OPS; i++) {
        const MicroOp &o = MICRONET_OPS[i];
        if (o.op == OP_CONV) {
            total += (long)o.out_c * o.out_h * o.out_w
                   * (o.in_c / o.groups) * o.kh * o.kw;
        } else if (o.op == OP_LINEAR) {
            total += (long)o.in_c * o.out_c;
        }
    }
    return total;
}

}  // namespace

#ifdef ARDUINO
#include <Arduino.h>

static float g_scratch[3 * MICRONET_PEAK];
extern const float micronet_weights[];   // PROGMEM blob, see scripts/pack_weights.sh
extern const int micronet_indices[];

void setup()
{
    Serial.begin(115200);
    Serial.printf("micronet: %d ops, %d weights, %lu MACs\n",
                  MICRONET_N_OPS, MICRONET_N_WEIGHTS, macs_from_op_table());
}

void loop()
{
    static float input[kInputSize];      // a real sketch fills this from a camera
    float logits[MICRONET_CLASSES];
    MicroNet net = { micronet_weights, micronet_indices, g_scratch };

    const unsigned long t0 = micros();
    micronet_forward(&net, input, logits);
    Serial.printf("inference %lu us\n", micros() - t0);
    delay(1000);
}

#else
#include <chrono>

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "generated";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    char path[512];

    std::snprintf(path, sizeof path, "%s/micronet_weights.bin", dir);
    const std::vector<float> weights = load_floats(path, MICRONET_N_WEIGHTS);
    std::snprintf(path, sizeof path, "%s/micronet_indices.bin", dir);
    const std::vector<int> indices = load_ints(path, MICRONET_N_INDICES);

    std::vector<float> scratch(3 * MICRONET_PEAK);
    std::vector<float> input(kInputSize, 0.1f);
    float logits[MICRONET_CLASSES];
    MicroNet net = { weights.data(), indices.data(), scratch.data() };

    micronet_forward(&net, input.data(), logits);   // warm the caches

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        micronet_forward(&net, input.data(), logits);
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    const long macs = macs_from_op_table();

    std::printf("host inference: %.2f ms over %d reps\n", ms, reps);
    std::printf("MACs per image: %ld (from the op table, not from PyTorch)\n", macs);
    std::printf("throughput:     %.1f MMAC/s\n", macs / (ms * 1e3));
    std::printf("working set:    %.1f KB float32, %.1f KB if int8\n",
                (MICRONET_N_WEIGHTS + 3.0 * MICRONET_PEAK) * 4 / 1024,
                (MICRONET_N_WEIGHTS + 3.0 * MICRONET_PEAK) / 1024);
    std::printf("\nThis is an Apple M4 core, not a Cortex-M. It bounds nothing "
                "about the device.\n");
    return 0;
}
#endif
