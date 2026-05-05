package org.rinha;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Top-5 nearest-neighbor search over the IVF vector index.
 *
 * The main entry point is searchIVF(), which is called for every
 * /fraud-score request. The other search methods (search, searchIndexed)
 * are slower brute-force paths used only in unit tests.
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
     * The main K-NN search. Called once per /fraud-score request.
     *
     * Phase 1: compare the query against all cluster centroids (float32)
     *          and pick the nprobe closest ones. Uses early termination —
     *          once a close centroid is found, distant ones are rejected
     *          after checking just 1-3 dimensions instead of all 14.
     *
     * Phase 2: scan every vector in each of those nprobe clusters using
     *          INT8 SIMD distance. The top-5 heap tightens as better
     *          neighbors are found, which causes early termination to
     *          skip more and more vectors as the scan progresses.
     *
     * All scratch buffers are ThreadLocal — zero heap allocation per call.
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
        // Local reference so the JIT sees a single load instead of repeated
        // field accesses through 'store' inside the inner distance loop.
        final byte[]  vecs       = store.vectors;

        final boolean soa = store.soaLayout;

        // ── Phase 1 — nprobe=1 fast path ─────────────────────────────────────
        // When only one cluster is requested, we just need the best centroid.
        // No sorted list, no scratch arrays.
        if (bound == 1) {
            int   bestC    = 0;
            float bestDist = Float.MAX_VALUE;
            for (int c = 0; c < C; c++) {
                final float cd = centDistSqEt(query, centroids, c, bestDist);
                if (cd < bestDist) { bestDist = cd; bestC = c; }
            }

            // Skip bbox check for nprobe=1: the top-5 heap starts empty
            // (worstInt = MAX_VALUE), so the lower-bound check can never prune.
            final int off = listOffsets[bestC];
            final int sz  = listSizes[bestC];

            if (soa) {
                scanClusterSoA(queryInt8, vecs, off, sz, topDist, topIdx, Integer.MAX_VALUE);
            } else {
                int worstInt = Integer.MAX_VALUE;
                for (int li = 0, vb = off * OffHeapVectorStore.DIMS; li < sz; li++, vb += OffHeapVectorStore.DIMS) {
                    final int d = distSqInt8(queryInt8, vecs, vb, worstInt);
                    if (d < worstInt) {
                        insert(topDist, topIdx, (float) d, off + li);
                        final float w = topDist[K - 1];
                        worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
                    }
                }
            }
            return;
        }

        // ── Phase 1 — general path (nprobe > 1) ──────────────────────────────
        final float[] probeDist = TL_PROBE_DIST.get();
        final int[]   probeIdx  = TL_PROBE_IDX .get();

        for (int i = 0; i < bound; i++) { probeDist[i] = Float.MAX_VALUE; probeIdx[i] = -1; }
        float probeWorst = Float.MAX_VALUE;

        for (int c = 0; c < C; c++) {
            // probeWorst tightens as the top-nprobe list fills up, so later
            // centroids get rejected after just 1-3 dimensions instead of 14.
            final float cd = centDistSqEt(query, centroids, c, probeWorst);
            if (cd < probeWorst) {
                insertProbe(probeDist, probeIdx, cd, c, bound);
                probeWorst = probeDist[bound - 1];
            }
        }

        // ── Phase 2 — general path ────────────────────────────────────────────
        int worstInt = Integer.MAX_VALUE;
        final byte[] bboxMin = store.bboxMinInt8;
        final byte[] bboxMax = store.bboxMaxInt8;

        for (int p = 0; p < bound; p++) {
            final int c = probeIdx[p];
            if (c < 0) break;

            if (bboxMin != null) {
                final int lb = bboxLbInt8(queryInt8, bboxMin, bboxMax, c);
                if (lb >= worstInt) continue;
            }

            final int off = listOffsets[c];
            final int sz  = listSizes[c];

            if (soa) {
                worstInt = scanClusterSoA(queryInt8, vecs, off, sz, topDist, topIdx, worstInt);
            } else {
                for (int li = 0, vb = off * OffHeapVectorStore.DIMS; li < sz; li++, vb += OffHeapVectorStore.DIMS) {
                    final int d = distSqInt8(queryInt8, vecs, vb, worstInt);
                    if (d < worstInt) {
                        insert(topDist, topIdx, (float) d, off + li);
                        final float w = topDist[K - 1];
                        worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
                    }
                }
            }
        }
    }

    /**
     * Scans one cluster using SoA-within-blocks-of-8 memory layout, computing
     * distances to 8 candidates simultaneously via SIMD.
     *
     * Memory layout for a block of 8 vectors:
     *   [8 × dim0 values][8 × dim1 values]...[8 × dim13 values]
     * This lets us load 8 db values for one dimension in a single BYTE_SPEC_64
     * load, broadcast the query value, subtract and square — all in one burst.
     * The last (clusterSize % 8) vectors fall back to scalar AoS.
     */
    private static int scanClusterSoA(final byte[] q, final byte[] vecs,
                                      final int off, final int sz,
                                      final float[] topDist, final int[] topIdx,
                                      int worstInt) {
        final int DIMS        = OffHeapVectorStore.DIMS;
        final int clusterBase = off * DIMS;
        final int blocks      = sz >> 3;   // sz / 8
        final int rem         = sz  & 7;   // sz % 8

        for (int b = 0; b < blocks; b++) {
            final int blockBase = clusterBase + b * DIMS * 8;

            // Accumulate squared distances for 8 candidates at once across all 14 dims.
            // Byte values are widened to short before subtraction (avoids byte overflow),
            // then to int before squaring (short squared can exceed short max).
            IntVector acc = IntVector.zero(INT_SPEC);
            for (int d = 0; d < DIMS; d++) {
                final ByteVector  dbV  = ByteVector.fromArray(BYTE_SPEC, vecs, blockBase + d * 8);
                final ShortVector dbS  = (ShortVector) dbV.convertShape(VectorOperators.B2S, SHORT_SPEC, 0);
                final ShortVector qS   = ShortVector.broadcast(SHORT_SPEC, (short) q[d]);
                final IntVector   diffI = (IntVector) qS.sub(dbS).convertShape(VectorOperators.S2I, INT_SPEC, 0);
                acc = acc.add(diffI.mul(diffI));
            }

            // Update the top-K heap with each of the 8 distances
            final int vecBase = off + (b << 3);
            for (int lane = 0; lane < 8; lane++) {
                final int dist = acc.lane(lane);
                if (dist < worstInt) {
                    insert(topDist, topIdx, (float) dist, vecBase + lane);
                    final float w = topDist[K - 1];
                    worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
                }
            }
        }

        // Scalar AoS fallback for the last (sz % 8) vectors
        final int remBase  = clusterBase + blocks * DIMS * 8;
        final int remStart = off + (blocks << 3);
        for (int i = 0; i < rem; i++) {
            final int dist = distSqInt8(q, vecs, remBase + i * DIMS, worstInt);
            if (dist < worstInt) {
                insert(topDist, topIdx, (float) dist, remStart + i);
                final float w = topDist[K - 1];
                worstInt = (w == Float.MAX_VALUE) ? Integer.MAX_VALUE : (int) w;
            }
        }

        return worstInt;
    }

    // ── Centroid distance (no early termination — kept for reference) ─────────

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

    // ── Centroid distance with early termination ──────────────────────────────

    /**
     * Distance from query q to centroid ci, stopping early if the running
     * total already beats the threshold. Safe to return early because all
     * squared terms are non-negative — the full sum can only be larger.
     *
     * In practice this rejects most centroids after 1-3 dimensions once
     * the search has found a reasonably close cluster to compare against.
     */
    private static float centDistSqEt(final float[] q, final float[] c,
                                      final int ci, final float threshold) {
        final int b = ci * OffHeapVectorStore.DIMS;
        float sum = 0, d;
        d = q[0]  - c[b];     sum += d*d; if (sum >= threshold) return sum;
        d = q[1]  - c[b+1];   sum += d*d; if (sum >= threshold) return sum;
        d = q[2]  - c[b+2];   sum += d*d; if (sum >= threshold) return sum;
        d = q[3]  - c[b+3];   sum += d*d; if (sum >= threshold) return sum;
        d = q[4]  - c[b+4];   sum += d*d; if (sum >= threshold) return sum;
        d = q[5]  - c[b+5];   sum += d*d; if (sum >= threshold) return sum;
        d = q[6]  - c[b+6];   sum += d*d; if (sum >= threshold) return sum;
        d = q[7]  - c[b+7];   sum += d*d; if (sum >= threshold) return sum;
        d = q[8]  - c[b+8];   sum += d*d; if (sum >= threshold) return sum;
        d = q[9]  - c[b+9];   sum += d*d; if (sum >= threshold) return sum;
        d = q[10] - c[b+10];  sum += d*d; if (sum >= threshold) return sum;
        d = q[11] - c[b+11];  sum += d*d; if (sum >= threshold) return sum;
        d = q[12] - c[b+12];  sum += d*d; if (sum >= threshold) return sum;
        d = q[13] - c[b+13];  sum += d*d;
        return sum;
    }

    // ── INT8 vector distance (static, for Phase 2 hot loop) ──────────────────

    // SIMD lane widths for the INT8 distance calculation.
    // We load 8 bytes at once, widen to shorts (so subtraction doesn't overflow),
    // widen again to ints (so squaring doesn't overflow), then sum.
    private static final VectorSpecies<Byte>    BYTE_SPEC  = ByteVector.SPECIES_64;
    private static final VectorSpecies<Short>   SHORT_SPEC = ShortVector.SPECIES_128;
    private static final VectorSpecies<Integer> INT_SPEC   = IntVector.SPECIES_256;

    /**
     * Squared distance between query vector q and one stored vector in vecs[].
     * Returns early if the partial sum already exceeds threshold (the current
     * worst distance in our top-5 heap), because the full distance can only
     * grow — there's no point continuing.
     *
     * Dims 0-7 use SIMD: 8 bytes loaded at once, widened to int, squared and
     * summed in a single hardware burst. Dims 8-13 are scalar because the
     * query array is only 14 bytes wide — we can't safely load 8 bytes starting
     * at index 8 without reading past the end of the array.
     */
    private static int distSqInt8(final byte[] q, final byte[] vecs,
                                  final int base, final int threshold) {
        // Dims 0-7: load 8 bytes from each, widen byte→short→int, subtract, square, sum
        final ShortVector qS = (ShortVector) ByteVector.fromArray(BYTE_SPEC, q, 0)
                .convertShape(VectorOperators.B2S, SHORT_SPEC, 0);
        final ShortVector vS = (ShortVector) ByteVector.fromArray(BYTE_SPEC, vecs, base)
                .convertShape(VectorOperators.B2S, SHORT_SPEC, 0);
        final IntVector dI = (IntVector) qS.sub(vS)
                .convertShape(VectorOperators.S2I, INT_SPEC, 0);
        int sum = dI.mul(dI).reduceLanes(VectorOperators.ADD);
        if (sum >= threshold) return sum;
        // Dims 8-13: scalar early-exit
        int d;
        d = q[8]  - vecs[base+8];   sum += d*d; if (sum >= threshold) return sum;
        d = q[9]  - vecs[base+9];   sum += d*d; if (sum >= threshold) return sum;
        d = q[10] - vecs[base+10];  sum += d*d; if (sum >= threshold) return sum;
        d = q[11] - vecs[base+11];  sum += d*d; if (sum >= threshold) return sum;
        d = q[12] - vecs[base+12];  sum += d*d; if (sum >= threshold) return sum;
        d = q[13] - vecs[base+13];  sum += d*d;
        return sum;
    }

    // ── Bbox lower-bound distance (INT8² scale, fully unrolled) ──────────────

    /**
     * Returns the minimum possible squared INT8 distance from the query vector
     * to any vector inside cluster {@code c}.
     *
     * For each dimension d:
     *   - if query[d] < bbox_min[d]:  contributes (bbox_min[d] - query[d])²
     *   - if query[d] > bbox_max[d]:  contributes (query[d] - bbox_max[d])²
     *   - otherwise: contributes 0
     *
     * Fully unrolled for DIMS=14 — JIT will keep this register-local and may
     * auto-vectorize the 14 clamped-delta additions.
     *
     * Result fits in int (max = 14 × 254² = 903,224).
     */
    private static int bboxLbInt8(final byte[] q,
                                  final byte[] bmin,
                                  final byte[] bmax,
                                  final int c) {
        final int b = c * OffHeapVectorStore.DIMS;
        int sum = 0;
        int qi, lo, hi, diff;

        qi=(int)q[0];  lo=(int)bmin[b];    hi=(int)bmax[b];    if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[1];  lo=(int)bmin[b+1];  hi=(int)bmax[b+1];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[2];  lo=(int)bmin[b+2];  hi=(int)bmax[b+2];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[3];  lo=(int)bmin[b+3];  hi=(int)bmax[b+3];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[4];  lo=(int)bmin[b+4];  hi=(int)bmax[b+4];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[5];  lo=(int)bmin[b+5];  hi=(int)bmax[b+5];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[6];  lo=(int)bmin[b+6];  hi=(int)bmax[b+6];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[7];  lo=(int)bmin[b+7];  hi=(int)bmax[b+7];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[8];  lo=(int)bmin[b+8];  hi=(int)bmax[b+8];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[9];  lo=(int)bmin[b+9];  hi=(int)bmax[b+9];  if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[10]; lo=(int)bmin[b+10]; hi=(int)bmax[b+10]; if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[11]; lo=(int)bmin[b+11]; hi=(int)bmax[b+11]; if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[12]; lo=(int)bmin[b+12]; hi=(int)bmax[b+12]; if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}
        qi=(int)q[13]; lo=(int)bmin[b+13]; hi=(int)bmax[b+13]; if(qi<lo){diff=lo-qi;sum+=diff*diff;}else if(qi>hi){diff=qi-hi;sum+=diff*diff;}

        return sum;
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

    /**
     * Counts fraud labels among the K nearest neighbors.
     * Branch-free inner loop: avoids misprediction on the label check.
     */
    static int fraudCount(final OffHeapVectorStore store, final int[] topIdx) {
        int count = 0;
        final byte[] labels = store.labels;
        for (int i = 0; i < K; i++) {
            final int idx = topIdx[i];
            // idx < 0 means slot unfilled (only possible if store < K vectors)
            if (idx >= 0) count += (labels[idx] == OffHeapVectorStore.FRAUD) ? 1 : 0;
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
