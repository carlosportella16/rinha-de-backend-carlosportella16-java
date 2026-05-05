package org.rinha;

/**
 * Top-5 nearest-neighbor search.
 *
 * Production path: searchIVF — IVF index with INT8 distances.
 * Test/fallback path: search / searchIndexed — VectorStore (heap float32).
 */
final class KnnSearch {

    static final int K          = 5;
    static final int MAX_NPROBE = 128; // max supported nprobe value

    // ── Per-I/O-thread scratch buffers ────────────────────────────────────────

    static final ThreadLocal<float[]> TL_DIST =
            ThreadLocal.withInitial(() -> new float[K]);
    static final ThreadLocal<int[]> TL_IDX =
            ThreadLocal.withInitial(() -> new int[K]);

    // INT8 quantized query vector (reused per thread, populated in handleFraudScore)
    static final ThreadLocal<byte[]> TL_VEC_INT8 =
            ThreadLocal.withInitial(() -> new byte[OffHeapVectorStore.DIMS]);

    // Scratch arrays for centroid top-nprobe selection (IVF)
    static final ThreadLocal<float[]> TL_PROBE_DIST =
            ThreadLocal.withInitial(() -> new float[MAX_NPROBE]);
    static final ThreadLocal<int[]> TL_PROBE_IDX =
            ThreadLocal.withInitial(() -> new int[MAX_NPROBE]);

    // ── Query quantization ────────────────────────────────────────────────────

    /**
     * Quantizes float query vector to INT8 byte array in place.
     * Maps f ∈ [-1.0, 1.0] → byte ∈ [-127, 127].
     * Sentinel -1.0 → -127 (preserved for correct distance semantics).
     */
    static void quantize(final float[] query, final byte[] out) {
        for (int i = 0; i < OffHeapVectorStore.DIMS; i++) {
            final float f = query[i];
            int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f)); // round
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
            out[i] = (byte) q;
        }
    }

    // ── IVF search (production path) ─────────────────────────────────────────

    /**
     * Searches the IVF index for the K nearest neighbors.
     *
     * Phase 1 — centroid selection:
     *   Scan all C centroids (float32 distance), select top {@code nprobe} closest.
     *   Cost: C × 14 float ops ≈ 0.003 ms for C=512.
     *
     * Phase 2 — candidate evaluation:
     *   For each selected cluster, compute INT8 distSq with early termination.
     *   Cost: nprobe × avgClusterSize × ~7 ns.
     *
     * All scratch arrays are ThreadLocal — zero heap allocation per call.
     */
    static void searchIVF(final OffHeapVectorStore store,
                          final float[]            query,
                          final byte[]             queryInt8,
                          final float[]            topDist,
                          final int[]              topIdx,
                          final int                nprobe) {

        topDist[0] = topDist[1] = topDist[2] = topDist[3] = topDist[4] = Float.MAX_VALUE;
        topIdx[0]  = topIdx[1]  = topIdx[2]  = topIdx[3]  = topIdx[4]  = -1;

        final int     C          = store.numClusters;
        final float[] centroids  = store.centroids;
        final int[]   listOffsets= store.listOffsets;
        final int[]   listSizes  = store.listSizes;
        final int     bound      = Math.min(nprobe, C);

        // ── Phase 1: find top nprobe centroids ────────────────────────────────
        final float[] probeDist = TL_PROBE_DIST.get();
        final int[]   probeIdx  = TL_PROBE_IDX .get();

        for (int i = 0; i < bound; i++) { probeDist[i] = Float.MAX_VALUE; probeIdx[i] = -1; }
        float probeWorst = Float.MAX_VALUE;

        for (int c = 0; c < C; c++) {
            final float cd = centDistSq(query, centroids, c);
            if (cd < probeWorst) {
                insertProbe(probeDist, probeIdx, cd, c, bound);
                probeWorst = probeDist[bound - 1];
            }
        }

        // ── Phase 2: scan IVF candidate lists ─────────────────────────────────
        // worstInt: current K-th distance in INT8 scale; Integer.MAX_VALUE until K found.
        int worstInt = Integer.MAX_VALUE;
        final int[] listData = store.listData; // null for V4 (cluster-ordered)

        for (int p = 0; p < bound; p++) {
            final int c = probeIdx[p];
            if (c < 0) break;
            final int off = listOffsets[c];
            final int sz  = listSizes[c];

            if (listData == null) {
                // V4: vectors are in cluster order — sequential scan (cache-friendly)
                for (int li = 0; li < sz; li++) {
                    final int d = store.distSqInt8(queryInt8, off + li, worstInt);
                    if (d < worstInt) {
                        insert(topDist, topIdx, (float) d, off + li);
                        final float w = topDist[K - 1];
                        worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
                    }
                }
            } else {
                // V3: indirect via listData (original index order)
                for (int li = 0; li < sz; li++) {
                    final int idx = listData[off + li];
                    final int d   = store.distSqInt8(queryInt8, idx, worstInt);
                    if (d < worstInt) {
                        insert(topDist, topIdx, (float) d, idx);
                        final float w = topDist[K - 1];
                        worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
                    }
                }
            }
        }
    }

    // ── Centroid distance ─────────────────────────────────────────────────────

    private static float centDistSq(final float[] q, final float[] c, final int ci) {
        final int b   = ci * OffHeapVectorStore.DIMS;
        final float d0  = q[0]  - c[b];
        final float d1  = q[1]  - c[b+1];
        final float d2  = q[2]  - c[b+2];
        final float d3  = q[3]  - c[b+3];
        final float d4  = q[4]  - c[b+4];
        final float d5  = q[5]  - c[b+5];
        final float d6  = q[6]  - c[b+6];
        final float d7  = q[7]  - c[b+7];
        final float d8  = q[8]  - c[b+8];
        final float d9  = q[9]  - c[b+9];
        final float d10 = q[10] - c[b+10];
        final float d11 = q[11] - c[b+11];
        final float d12 = q[12] - c[b+12];
        final float d13 = q[13] - c[b+13];
        return d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6
              +d7*d7+d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13;
    }

    // ── Probe insertion (variable-K version for centroid top-nprobe) ──────────

    private static void insertProbe(final float[] topDist, final int[] topIdx,
                                     final float d, final int idx, final int k) {
        int pos = k - 1;
        while (pos > 0 && d < topDist[pos - 1]) {
            topDist[pos] = topDist[pos - 1];
            topIdx[pos]  = topIdx[pos - 1];
            pos--;
        }
        topDist[pos] = d;
        topIdx[pos]  = idx;
    }

    // ── Score ─────────────────────────────────────────────────────────────────

    static int fraudCount(final OffHeapVectorStore store, final int[] topIdx) {
        int count = 0;
        for (int i = 0; i < K; i++) {
            final int idx = topIdx[i];
            if (idx >= 0 && store.getLabel(idx) == OffHeapVectorStore.FRAUD) count++;
        }
        return count;
    }

    // ── Heap VectorStore path (kept for tests) ────────────────────────────────

    static void search(final VectorStore store,
                       final float[]     query,
                       final float[]     topDist,
                       final int[]       topIdx) {

        topDist[0] = Float.MAX_VALUE; topDist[1] = Float.MAX_VALUE;
        topDist[2] = Float.MAX_VALUE; topDist[3] = Float.MAX_VALUE;
        topDist[4] = Float.MAX_VALUE;
        topIdx[0]  = -1; topIdx[1] = -1; topIdx[2] = -1;
        topIdx[3]  = -1; topIdx[4] = -1;

        final float[][] vectors = store.vectors;
        final int       n       = store.size();
        float worst = Float.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            final float d = VectorStore.distSq(query, vectors[i]);
            if (d < worst) {
                insert(topDist, topIdx, d, i);
                worst = topDist[K - 1];
            }
        }
    }

    static void searchIndexed(final VectorStore store,
                               final float[]    query,
                               final float[]    topDist,
                               final int[]      topIdx) {

        topDist[0] = Float.MAX_VALUE; topDist[1] = Float.MAX_VALUE;
        topDist[2] = Float.MAX_VALUE; topDist[3] = Float.MAX_VALUE;
        topDist[4] = Float.MAX_VALUE;
        topIdx[0]  = -1; topIdx[1] = -1; topIdx[2] = -1;
        topIdx[3]  = -1; topIdx[4] = -1;

        final float[]   sortedDim0   = store.sortedDim0;
        final int[]     sortedByDim0 = store.sortedByDim0;
        final float[][] vectors      = store.vectors;
        final int       n            = store.size();
        final float     q0           = query[0];

        int lo = 0, hi = n;
        while (lo < hi) {
            final int mid = (lo + hi) >>> 1;
            if (sortedDim0[mid] < q0) lo = mid + 1;
            else hi = mid;
        }

        int   left  = lo - 1;
        int   right = lo;
        float worst = Float.MAX_VALUE;

        while (left >= 0 || right < n) {
            final float dl     = (left  >= 0) ? (q0 - sortedDim0[left])  : Float.MAX_VALUE;
            final float dr     = (right <  n) ? (sortedDim0[right] - q0) : Float.MAX_VALUE;
            final boolean goLeft = (dl <= dr);
            final float   gap    = goLeft ? dl : dr;
            if (gap * gap >= worst) break;
            final int idx = goLeft ? sortedByDim0[left--] : sortedByDim0[right++];
            final float d = VectorStore.distSq(query, vectors[idx]);
            if (d < worst) {
                insert(topDist, topIdx, d, idx);
                worst = topDist[K - 1];
            }
        }
    }

    // ── Heap-store score helpers (tests) ──────────────────────────────────────

    static int fraudCount(final VectorStore store, final int[] topIdx) {
        int count = 0;
        for (int i = 0; i < K; i++) {
            final int idx = topIdx[i];
            if (idx >= 0 && store.labels[idx] == VectorStore.FRAUD) count++;
        }
        return count;
    }

    static float fraudScore(final byte[] labels, final int[] topIdx) {
        int count = 0;
        for (int i = 0; i < K; i++) {
            final int idx = topIdx[i];
            if (idx >= 0 && labels[idx] == VectorStore.FRAUD) count++;
        }
        return count * (1f / K);
    }

    // ── Insertion helper (fixed K=5) ──────────────────────────────────────────

    static void insert(final float[] topDist, final int[] topIdx,
                       final float d, final int idx) {
        int pos = K - 1;
        while (pos > 0 && d < topDist[pos - 1]) {
            topDist[pos] = topDist[pos - 1];
            topIdx[pos]  = topIdx[pos - 1];
            pos--;
        }
        topDist[pos] = d;
        topIdx[pos]  = idx;
    }
}
