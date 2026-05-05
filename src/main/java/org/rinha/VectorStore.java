package org.rinha;

import java.util.Arrays;

/**
 * Compact, preallocated store for the reference vector dataset.
 *
 * Memory layout (3 000 000 vectors × 14 dims):
 *   float[][] vectors  — outer array  : 3_000_000 refs  × 8 B =   24 MB
 *                      — 3M float[14] : (12 hdr + 56 data) × 3M = 204 MB
 *   byte[]   labels    — 3_000_000 × 1 B                          =    3 MB
 *   float[]  sortedDim0  — sorted dim-0 values                    =   12 MB
 *   int[]    sortedByDim0 — original indices in sorted order       =   12 MB
 *                                                                 ≈ 255 MB
 *
 * Rules:
 *   - NO new objects after construction (all slots are pre-wired).
 *   - NO List / Map / wrapper types.
 *   - add() copies into the pre-allocated slot — the caller's float[] is NEVER retained.
 *   - Labels encoded as primitives: FRAUD = 1, LEGIT = 0.
 */
final class VectorStore {

    // ── Constants ────────────────────────────────────────────────────────────

    static final int  DIMS      = 14;
    static final int  CAPACITY  = 3_000_000;
    static final byte FRAUD     = 1;
    static final byte LEGIT     = 0;

    // ── Storage ──────────────────────────────────────────────────────────────

    final float[][] vectors;   // vectors[i][dim]  — pre-wired, never null after ctor
    final byte[]    labels;    // labels[i]         — FRAUD | LEGIT
    private int     size;      // number of vectors loaded so far

    // ── Coarse filter index (built once after load, never mutated) ────────────

    /** dim-0 values in sorted order; null until buildDim0Index() is called. */
    float[] sortedDim0;
    /** original vector indices in dim-0 sorted order. */
    int[]   sortedByDim0;

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * Allocates the full store for CAPACITY vectors.
     * Triggers a single large GC-safe allocation burst at startup —
     * nothing is allocated during load or query time.
     */
    VectorStore() {
        this(CAPACITY);
    }

    /**
     * Allocates a store for {@code capacity} vectors.
     * Use a smaller value for tests or when you know the exact dataset size.
     */
    VectorStore(int capacity) {
        vectors = new float[capacity][DIMS]; // pre-allocates ALL inner float[14] arrays
        labels  = new byte[capacity];
        size    = 0;
    }

    // ── Loading ───────────────────────────────────────────────────────────────

    /**
     * Copies {@code vec[0..DIMS-1]} into the next free slot and records {@code label}.
     * <p>
     * The caller's array is NOT retained — copy happens via System.arraycopy.
     * No objects are allocated.
     *
     * @param vec   float[DIMS] produced by the reference-file parser
     * @param label FRAUD or LEGIT
     */
    void add(final float[] vec, final byte label) {
        System.arraycopy(vec, 0, vectors[size], 0, DIMS);
        labels[size++] = label;
    }

    /**
     * Writes 14 individual float values directly into slot {@code idx}.
     * Zero allocation — useful when parsing without an intermediate float[] buffer.
     */
    void set(final int idx,
             final float v0,  final float v1,  final float v2,  final float v3,
             final float v4,  final float v5,  final float v6,  final float v7,
             final float v8,  final float v9,  final float v10, final float v11,
             final float v12, final float v13,
             final byte label) {
        final float[] slot = vectors[idx];
        slot[0]  = v0;  slot[1]  = v1;  slot[2]  = v2;  slot[3]  = v3;
        slot[4]  = v4;  slot[5]  = v5;  slot[6]  = v6;  slot[7]  = v7;
        slot[8]  = v8;  slot[9]  = v9;  slot[10] = v10; slot[11] = v11;
        slot[12] = v12; slot[13] = v13;
        labels[idx] = label;
        if (idx >= size) size = idx + 1;
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /** Number of vectors currently loaded. */
    int size() { return size; }

    /** Called by VectorLoader after a bulk fill to set size without going through add(). */
    void forceSize(int n) { size = n; }

    /**
     * Returns the pre-allocated float[DIMS] slot for vector {@code i}.
     * The array is owned by this store — callers must NOT modify it.
     */
    float[] vectorAt(final int i) { return vectors[i]; }

    /** Returns FRAUD or LEGIT for vector {@code i}. */
    byte labelAt(final int i) { return labels[i]; }

    // ── Squared Euclidean distance ────────────────────────────────────────────

    // ── Squared Euclidean distance ────────────────────────────────────────────

    /**
     * {@code true} when {@link SimdDistance} initialized successfully, meaning
     * {@code jdk.incubator.vector} is loaded and the Vector API is operational.
     *
     * <p>Detected once at class-init via {@code Class.forName}; any failure
     * (missing module, unsupported CPU) silently falls back to scalar.
     *
     * <p>Because this field is {@code static final}, the JIT constant-folds the
     * dispatch in {@link #distSq} — the dead branch is eliminated entirely:
     * SIMD builds inline {@code SimdDistance.distSq}; scalar builds inline
     * {@code distSqScalar}. Zero overhead after the first JIT compilation.
     */
    static final boolean SIMD_AVAILABLE;
    static {
        boolean ok = false;
        try {
            // Eagerly initialize SimdDistance — triggers its static init which
            // references FloatVector.  If jdk.incubator.vector is not in the
            // module graph, this throws NoClassDefFoundError / EIFE → caught.
            Class.forName("org.rinha.SimdDistance");
            ok = true;
        } catch (Throwable t) {
            System.err.println("[WARN] Vector API unavailable — using scalar distSq: " + t);
        }
        SIMD_AVAILABLE = ok;
    }

    /**
     * Squared Euclidean distance dispatcher.
     *
     * <p>Routes to {@link SimdDistance#distSq} (Vector API SIMD) when
     * {@link #SIMD_AVAILABLE} is true, otherwise falls back to
     * {@link #distSqScalar} (explicit 14-dim unroll).
     *
     * <p>The {@code static final} guard is constant-folded by C2 — after JIT
     * compilation the call site contains only the chosen implementation with
     * no branch instruction.
     */
    static float distSq(final float[] a, final float[] b) {
        if (SIMD_AVAILABLE) return SimdDistance.distSq(a, b);
        return distSqScalar(a, b);
    }

    /**
     * Scalar fallback and benchmark baseline — explicit 14-dim unroll.
     *
     * <p>Why keep this alongside SIMD:
     * <ol>
     *   <li>Used automatically when {@code jdk.incubator.vector} is absent.
     *   <li>Serves as the ground-truth oracle in {@link DistanceBenchmark}.
     *   <li>For machines where the JIT's auto-vectoriser already produces
     *       equivalent code, SIMD API overhead (reduceLanes) may make this
     *       equally fast or faster — the benchmark shows the real answer.
     * </ol>
     *
     * <p>The unrolled form exposes 14 independent FMA pairs to the out-of-order
     * execution engine; the processor issues them in parallel limited only by
     * FMA unit count (typically 2 per cycle on modern x86).
     */
    static float distSqScalar(final float[] a, final float[] b) {
        final float d0  = a[0]  - b[0];
        final float d1  = a[1]  - b[1];
        final float d2  = a[2]  - b[2];
        final float d3  = a[3]  - b[3];
        final float d4  = a[4]  - b[4];
        final float d5  = a[5]  - b[5];
        final float d6  = a[6]  - b[6];
        final float d7  = a[7]  - b[7];
        final float d8  = a[8]  - b[8];
        final float d9  = a[9]  - b[9];
        final float d10 = a[10] - b[10];
        final float d11 = a[11] - b[11];
        final float d12 = a[12] - b[12];
        final float d13 = a[13] - b[13];
        return d0*d0   + d1*d1   + d2*d2   + d3*d3
             + d4*d4   + d5*d5   + d6*d6   + d7*d7
             + d8*d8   + d9*d9   + d10*d10 + d11*d11
             + d12*d12 + d13*d13;
    }

    /**
     * Convenience overload: squared distance between query {@code q}
     * and the stored vector at index {@code idx}.
     * Avoids one extra field read compared to {@code distSq(q, vectorAt(idx))}.
     */
    float distSq(final float[] q, final int idx) {
        return distSq(q, vectors[idx]);
    }

    // ── Coarse filter index ───────────────────────────────────────────────────

    /**
     * Builds a sorted index on dimension 0 so that {@link KnnSearch#searchIndexed}
     * can skip large portions of the dataset via a 1D bounding box.
     *
     * <p>Algorithm:
     * Pack each vector as {@code (sortable_float_bits : original_index)} into a
     * {@code long[]}, sort in one pass (no boxing), then unpack into
     * {@link #sortedDim0} and {@link #sortedByDim0}.
     *
     * <p>The float-to-sortable-int trick:
     * <ul>
     *   <li>Positive float: flip only the sign bit → unsigned value preserves order.
     *   <li>Negative float: flip all bits → reverses the negative-magnitude order.
     * </ul>
     * Both cases yield an {@code int} that, when compared as a <em>signed</em> 32-bit
     * integer, respects the original float ordering (including -0 == +0 and NaN last).
     *
     * <p>Cost: ~24 MB temporary + ~24 MB permanent; called once at startup.
     * Uses {@link Arrays#parallelSort} for speed on multi-core hosts.
     */
    void buildDim0Index() {
        final int n = size;

        // Step 1 — pack (sortable key | original index) into one long per vector
        final long[] tmp = new long[n];
        for (int i = 0; i < n; i++) {
            final int bits = Float.floatToRawIntBits(vectors[i][0]);
            // Map float bit-pattern to a value that sorts correctly as signed int:
            //   negative float  → flip all bits  (reverses negative-magnitude desc → asc)
            //   positive float  → flip sign bit   (puts positives above negatives)
            final int key = (bits < 0) ? ~bits : (bits ^ 0x80000000);
            // High 32 bits = sort key; low 32 bits = original index (tiebreaker, harmless)
            tmp[i] = ((long) key << 32) | (i & 0xFFFFFFFFL);
        }

        // Step 2 — sort once; parallelSort uses fork/join for large arrays
        Arrays.parallelSort(tmp);

        // Step 3 — unpack into the two index arrays
        sortedDim0   = new float[n];
        sortedByDim0 = new int[n];
        for (int i = 0; i < n; i++) {
            final int origIdx = (int)(tmp[i] & 0xFFFFFFFFL);
            sortedByDim0[i]   = origIdx;
            sortedDim0[i]     = vectors[origIdx][0];
        }
    }
}

