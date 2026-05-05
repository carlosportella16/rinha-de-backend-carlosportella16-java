package org.rinha;

/**
 * Micro-benchmark comparing scalar vs SIMD squared Euclidean distance
 * for 14-dimensional float vectors.
 *
 * <p>Run with:
 * <pre>
 *   java --add-modules jdk.incubator.vector \
 *        -cp target/classes \
 *        org.rinha.DistanceBenchmark
 * </pre>
 *
 * <p>No JMH, no frameworks — manual timing with three rounds to show JIT effect.
 * Between rounds the JVM has compiled both methods to native; the best round
 * gives the closest approximation to steady-state throughput.
 */
public final class DistanceBenchmark {

    private static final int DIMS    = VectorStore.DIMS;   // 14
    private static final int ITERS   = 10_000_000;
    private static final int ROUNDS  = 5;
    private static final int WARMUP  = 200_000;

    /**
     * Volatile black hole: writing here forces the JIT to materialise every
     * computed result and prevents Dead Code Elimination of the benchmark loops.
     * One volatile write per round (not per iteration) keeps its overhead negligible.
     */
    private static volatile float SINK = 0f;

    public static void main(String[] args) {
        System.out.println("=== Distance Benchmark (DIMS=" + DIMS + ") ===");
        System.out.println("SIMD available : " + VectorStore.SIMD_AVAILABLE);
        if (VectorStore.SIMD_AVAILABLE) {
            System.out.println("SIMD species   : " + SimdDistance.SPECIES
                    + " (" + SimdDistance.LANES + " lanes, "
                    + SimdDistance.SPECIES.vectorBitSize() + "-bit)");
            System.out.println("Full passes    : " + (DIMS / SimdDistance.LANES)
                    + "  |  scalar tail: " + (DIMS % SimdDistance.LANES) + " elements");
        }
        System.out.println("Iterations     : " + fmt(ITERS) + " per round");
        System.out.println();

        // ── Create test vectors spread across [−1, 1] ────────────────────────
        final float[] a = new float[DIMS];
        final float[] b = new float[DIMS];
        long seed = 0xDEADBEEFCAFEL;
        for (int d = 0; d < DIMS; d++) {
            seed = lcg(seed); a[d] = ((seed >>> 33) / (float)(1L << 30)) - 1f;
            seed = lcg(seed); b[d] = ((seed >>> 33) / (float)(1L << 30)) - 1f;
        }

        // Verify both methods agree (before JIT may change behaviour)
        final float scalarRef = VectorStore.distSqScalar(a, b);
        if (VectorStore.SIMD_AVAILABLE) {
            final float simdRef = SimdDistance.distSq(a, b);
            if (Math.abs(scalarRef - simdRef) > 1e-4f) {
                System.err.printf("[ERROR] Mismatch! scalar=%.6f  simd=%.6f%n", scalarRef, simdRef);
                System.exit(1);
            }
            System.out.printf("Correctness    : PASS  (scalar=%.6f  simd=%.6f  |Δ|=%.3e)%n%n",
                    scalarRef, simdRef, Math.abs(scalarRef - simdRef));
        } else {
            System.out.printf("Correctness    : N/A (SIMD not available)  scalar=%.6f%n%n", scalarRef);
        }

        // ── Warmup — give JIT time to compile both paths to C2 native ────────
        System.out.print("Warming up (" + fmt(WARMUP) + " iters)... ");
        float sink = 0f;
        for (int i = 0; i < WARMUP; i++) {
            sink += VectorStore.distSqScalar(a, b);
            if (VectorStore.SIMD_AVAILABLE) sink += SimdDistance.distSq(a, b);
        }
        SINK = sink;   // volatile write — prevents DCE of the warmup loop
        System.out.println("done");
        System.out.println();

        // ── Benchmark rounds ─────────────────────────────────────────────────
        long bestScalar = Long.MAX_VALUE;
        long bestSimd   = Long.MAX_VALUE;

        System.out.printf("%-8s  %16s  %16s  %10s%n",
                "Round", "scalar (ns/op)", "simd (ns/op)", "speedup");
        System.out.println("─".repeat(58));

        for (int r = 1; r <= ROUNDS; r++) {
            // ── Scalar ──
            // Accumulate into a local, then flush to SINK once per round.
            // This forces the JIT to compute every iteration (uses the accumulator)
            // while keeping the volatile overhead at 1 write per round.
            sink = 0f;
            final long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) sink += VectorStore.distSqScalar(a, b);
            final long scalarNs = System.nanoTime() - t0;
            SINK = sink;   // volatile flush — prevents DCE
            if (scalarNs < bestScalar) bestScalar = scalarNs;

            // ── SIMD ──
            long simdNs = -1;
            if (VectorStore.SIMD_AVAILABLE) {
                sink = 0f;
                final long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) sink += SimdDistance.distSq(a, b);
                simdNs = System.nanoTime() - t1;
                SINK = sink;   // volatile flush
                if (simdNs < bestSimd) bestSimd = simdNs;
            }

            final String speedup = (simdNs > 0)
                    ? String.format("%.2fx", (double)scalarNs / simdNs)
                    : "N/A";
            final String simdStr = (simdNs > 0)
                    ? String.format("%.3f", (double)simdNs / ITERS)
                    : "N/A";

            System.out.printf("%-8d  %16.3f  %16s  %10s%n",
                    r, (double)scalarNs / ITERS, simdStr, speedup);
        }

        System.out.println("─".repeat(58));

        if (VectorStore.SIMD_AVAILABLE) {
            final double scalarNsOp = (double)bestScalar / ITERS;
            final double simdNsOp   = (double)bestSimd   / ITERS;
            final double speedup    = scalarNsOp / simdNsOp;
            final String winner     = speedup > 1.05 ? "← SIMD wins" :
                                      speedup < 0.95 ? "← scalar wins" : "← roughly equal";
            System.out.printf("%-8s  %16.3f  %16.3f  %9.2fx  %s%n",
                    "BEST", scalarNsOp, simdNsOp, speedup, winner);
            System.out.println();
            System.out.println("Interpretation:");
            System.out.println("  • scalar   uses 14 independent FMA pairs; out-of-order CPU");
            System.out.println("    may already issue them in parallel → SIMD gap may be small.");
            System.out.println("  • SIMD     horizontal reduceLanes adds ~" +
                    Integer.bitCount(SimdDistance.LANES - 1) + " adds overhead.");
            if (SimdDistance.TAIL_START == 0) {
                System.out.println("  • NOTE: SPECIES_PREFERRED=" + SimdDistance.LANES +
                        " lanes > DIMS=14 → entire computation is scalar tail!");
                System.out.println("    Consider using SPECIES_128 explicitly for 14-dim vectors.");
            }
        } else {
            System.out.printf("%-8s  %16.3f  %16s  %10s%n",
                    "BEST", (double)bestScalar / ITERS, "N/A", "N/A");
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Simple LCG for reproducible pseudo-random test data (no allocation). */
    private static long lcg(long s) {
        return s * 6364136223846793005L + 1442695040888963407L;
    }

    /** Format long with thousands separators, no boxing. */
    private static String fmt(long n) {
        return String.format("%,d", n);
    }
}

