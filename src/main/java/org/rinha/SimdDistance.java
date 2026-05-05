package org.rinha;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated squared Euclidean distance using the JDK Vector API
 * ({@code jdk.incubator.vector}, JEP 508 — 10th incubator in JDK 25).
 *
 * <p><b>Why a separate class?</b><br>
 * Isolating the Vector API imports here means {@link VectorStore} can load
 * successfully even when {@code --add-modules jdk.incubator.vector} is absent.
 * {@code VectorStore} detects availability with a {@code try-catch} around
 * {@code Class.forName("org.rinha.SimdDistance")} in its static initializer;
 * if this class fails to initialize, {@code VectorStore.SIMD_AVAILABLE} stays
 * {@code false} and every call falls back to the explicit scalar unroll.
 *
 * <p><b>Runtime requirement:</b>
 * {@code java --add-modules jdk.incubator.vector -jar app.jar}
 *
 * <h2>Processing layout for DIMS = 14</h2>
 * <pre>
 * SPECIES_128  (SSE,  4 lanes): 3 full FMAs  [0-3][4-7][8-11]  +  2 scalar [12-13]
 * SPECIES_256  (AVX2, 8 lanes): 1 full FMA   [0-7]              +  6 scalar [8-13]
 * SPECIES_512  (AVX-512, 16 l): 0 full FMAs                     + 14 scalar [0-13]
 * </pre>
 *
 * Full-width iterations use {@code fromArray} <em>without</em> a mask, which
 * hits HotSpot's fast intrinsic path and avoids the scalar-fallback "slow path"
 * that a masked load triggers when the array is shorter than the species width.
 * The tail is handled by a short scalar loop (no masked load, no AIOOBE risk).
 *
 * <p>Each full-width step uses {@code fma(d, d, acc)} — a fused multiply-add
 * that computes {@code d² + acc} in a single hardware instruction with one
 * rounding, saving both an instruction and a tiny amount of floating-point error.
 *
 * <p>{@link FloatVector} instances are stack-allocated by HotSpot's escape
 * analysis / scalar-replacement pass — zero heap allocation after JIT warmup.
 */
final class SimdDistance {

    // SPECIES_PREFERRED selects the widest SIMD width the CPU supports:
    //   128-bit (SSE4) → 4 float lanes
    //   256-bit (AVX2) → 8 float lanes  ← most common on modern x86 since 2013
    //   512-bit (AVX-512) → 16 float lanes
    static final VectorSpecies<Float> SPECIES  = FloatVector.SPECIES_PREFERRED;
    static final int                  LANES    = SPECIES.length();

    // Loop bound for full-width passes: `for (i=0; i <= BOUND; i += LANES)`.
    // Negative (e.g. LANES=16, DIMS=14 → BOUND=-2) means no full passes → all scalar.
    static final int BOUND      = VectorStore.DIMS - LANES;

    // First index NOT covered by a full pass → where the scalar tail begins.
    // Examples: LANES=4 → 12,  LANES=8 → 8,  LANES=16 → 0 (all-scalar fallback).
    static final int TAIL_START = (VectorStore.DIMS / LANES) * LANES;

    private SimdDistance() {}

    /**
     * Squared Euclidean distance between two 14-dim float arrays.
     * No sqrt — preserves distance ordering without the transcendental cost.
     *
     * <p>Zero heap allocation: {@link FloatVector} objects are stack values
     * eliminated by HotSpot's scalar-replacement pass after ~10 000 calls.
     *
     * @param a  query vector     (length ≥ {@link VectorStore#DIMS})
     * @param b  reference vector (length ≥ {@link VectorStore#DIMS})
     * @return   squared Euclidean distance ≥ 0
     */
    static float distSq(final float[] a, final float[] b) {
        var acc = FloatVector.zero(SPECIES);

        // ── Full-width SIMD passes ────────────────────────────────────────────
        // fromArray(species, arr, i) accesses arr[i..i+LANES-1], all in-bounds
        // because the loop stops at BOUND = DIMS - LANES.
        // This takes the JIT's fast intrinsic path (no masked-load penalty).
        for (int i = 0; i <= BOUND; i += LANES) {
            final var d = FloatVector.fromArray(SPECIES, a, i)
                                     .sub(FloatVector.fromArray(SPECIES, b, i));
            acc = d.fma(d, acc);   // fused: d² + acc  (1 instruction, 1 rounding)
        }

        // ── Horizontal reduction ──────────────────────────────────────────────
        // Sums all LANES of the accumulator into one scalar.
        float sum = acc.reduceLanes(VectorOperators.ADD);

        // ── Scalar tail ───────────────────────────────────────────────────────
        // Handles DIMS % LANES remaining elements.
        // Avoids masked fromArray — which falls to a slow "vOp/lambda" path in
        // HotSpot when offset+LANES > array.length, with per-lane bounds checks.
        // For LANES=4: tail = 2 ops.  For LANES=8: 6 ops.  For LANES=16: 14 ops.
        for (int i = TAIL_START; i < VectorStore.DIMS; i++) {
            final float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }
}

