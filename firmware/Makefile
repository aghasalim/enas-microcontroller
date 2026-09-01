# Host build. Verifies the C forward against the PyTorch outputs in
# generated/micronet_golden.bin. No cross toolchain needed for this.
CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -I.
LDLIBS  := -lm

.PHONY: test clean
test: test_equivalence
	./test_equivalence generated

test_equivalence: test_equivalence.c micronet.c micronet.h generated/micronet_arch.h
	$(CC) $(CFLAGS) -o $@ test_equivalence.c micronet.c $(LDLIBS)

clean:
	rm -f test_equivalence
