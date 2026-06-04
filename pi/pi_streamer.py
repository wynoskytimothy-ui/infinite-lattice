#!/usr/bin/env python3
"""
pi_streamer.py
==============
Stream digits of pi forever to a single file, using only:
    +, -, *, /, sqrt   on   {0, 1, 2, 4, r}
plus the right-triangle Markov recurrence:
    A_new = C/2
    B_new = A_new^2 / (1 + sqrt(1 - A_new^2))     # cancellation-free sagitta
    C_new = sqrt(A_new^2 + B_new^2)
    N_new = 2*N
    S    += N * A * B / 2                          # accumulate triangle areas
=> S -> pi    (for the unit circle r=1)

ARCHITECTURE (epoch-based, correct forever)
-------------------------------------------
Each "epoch" runs the recurrence from scratch at a higher precision than the last.
Epoch n targets ~target_n stable digits, with precision well above that floor.
At the end of each epoch, the file is appended with the new digits beyond the previous
epoch's emission. This is the correct way to stream because precision can't be
"retroactively raised" mid-run -- low-order rounding error in S is permanent.

Memory: O(target_digits) per variable per epoch (5 variables: A, B, C, N, S).
Output: pi_digits.txt -- one continuous string of digits, "3." prefixed.
"""
from mpmath import mp, mpf, sqrt, log10
import time, os, sys

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_digits.txt")
LOG_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_streamer.log")

EPOCH_DIGITS_START = 200    # epoch 1 emits this many digits
EPOCH_DIGITS_GROWTH = 1.5   # each epoch's target = previous * this
PRECISION_BUFFER_BITS = 500 # always keep this much above what's needed
SAFETY_DIGITS = 10          # don't trust the last few digits of an epoch's output

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

def init_output_file():
    """Ensure file exists, return how many digits already written (1 = just '3.')."""
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            f.write("3.")
        return 1
    with open(OUTPUT_FILE, 'r') as f:
        content = f.read().strip()
    if not content.startswith("3."):
        raise RuntimeError(f"{OUTPUT_FILE} doesn't start with '3.' -- refusing to corrupt.")
    return 1 + (len(content) - 2)  # 1 for "3", plus chars after "."

def run_epoch_to_pi(target_digits):
    """
    Run the Markov recurrence at precision sufficient for target_digits stable digits.
    Returns the digit string of pi to target_digits + safety, or None if something fails.
    """
    prec_bits = int(target_digits * 3.5) + PRECISION_BUFFER_BITS
    mp.prec = prec_bits

    A = mpf(1)
    B = mpf(1)
    C = sqrt(mpf(2))
    N = 4
    S = mpf(0)

    # Iterate until tail bound proves target_digits are stable
    max_iter = int(target_digits * 2) + 50
    for it in range(1, max_iter + 1):
        layer = mpf(N) * A * B / 2
        S += layer

        # Update state
        A2 = (C * C) / 4
        B_new = A2 / (1 + sqrt(1 - A2))
        C_new = sqrt(A2 + B_new * B_new)
        A = sqrt(A2)
        B = B_new
        C = C_new
        N *= 2

        # Convergence check: tail < 10^(-(target+safety))
        if it > 5 and layer > 0:
            stable = -log10(layer / 3)
            if stable > target_digits + SAFETY_DIGITS + 5:
                # Done: enough digits stable
                s_str = mp.nstr(S, target_digits + SAFETY_DIGITS + 5, strip_zeros=False)
                return s_str, it

    # Out of iterations
    s_str = mp.nstr(S, target_digits + SAFETY_DIGITS + 5, strip_zeros=False)
    return s_str, max_iter

def stream_pi():
    digits_written = init_output_file()
    log(f"Output file: {OUTPUT_FILE}")
    log(f"Resuming with {digits_written} digit(s) already written.")

    target = EPOCH_DIGITS_START
    while target <= digits_written:
        target = int(target * EPOCH_DIGITS_GROWTH)

    epoch = 0
    total_start = time.time()

    out = open(OUTPUT_FILE, 'a')
    try:
        while True:
            epoch += 1
            t0 = time.time()
            log(f"Epoch {epoch}: target={target} digits, precision={int(target*3.5)+PRECISION_BUFFER_BITS} bits")

            result, iters = run_epoch_to_pi(target)
            if result is None:
                log("Epoch failed -- skipping.")
                continue

            if '.' not in result:
                log(f"Unexpected result format: {result[:50]}...")
                continue
            int_part, frac = result.split('.')
            if int_part != '3':
                log(f"Sum not yet at pi (int part = '{int_part}') -- skipping.")
                continue

            # We've already written "3." plus (digits_written - 1) fractional chars.
            # Write fractional chars from position (digits_written - 1) up to (target - 1).
            already_frac = digits_written - 1
            new_chunk_end = target - 1
            new_chunk = frac[already_frac:new_chunk_end]

            if new_chunk:
                out.write(new_chunk)
                out.flush()
                os.fsync(out.fileno())
                digits_written += len(new_chunk)
                elapsed = time.time() - t0
                total_elapsed = time.time() - total_start
                rate = (digits_written - 1) / total_elapsed
                log(f"Epoch {epoch} done in {elapsed:.1f}s: +{len(new_chunk)} digits "
                    f"(total {digits_written}), {iters} iterations, {rate:.0f} digits/s avg")

            # Plan next epoch
            target = int(target * EPOCH_DIGITS_GROWTH)
    finally:
        out.close()
        log(f"Stopped. Total: {digits_written} digits, "
            f"{time.time()-total_start:.1f}s elapsed, {epoch} epochs.")

if __name__ == "__main__":
    try:
        stream_pi()
    except KeyboardInterrupt:
        log("Interrupted by user -- output file is intact and resumable.")
        sys.exit(0)
