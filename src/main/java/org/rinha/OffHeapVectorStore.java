package org.rinha;

import sun.misc.Unsafe;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Off-heap INT8 vector store with IVF index.
 *
 * Memory layout (version 3):
 *   vectorsBuf: N × 14 bytes (INT8, native memory) — 42 MB for 3M vecs
 *   labelsBuf:  N bytes                            —  3 MB for 3M vecs
 *
 * IVF index (on-heap):
 *   centroids:    C × 14 floats = 28 KB for C=512
 *   listOffsets:  C ints        =  2 KB
 *   listSizes:    C ints        =  2 KB
 *   listData:     N ints        = 12 MB for 3M vecs
 *
 * INT8 encoding: float f → byte = round(f × 127), clamped to [-127, 127].
 * Sentinel -1.0 → -127.  All floats in [0,1] → [0, 127].
 */
final class OffHeapVectorStore {

    static final int  DIMS       = 14;
    static final int  CAPACITY   = 3_000_000;
    static final byte FRAUD      = 1;
    static final byte LEGIT      = 0;
    static final int  VEC_STRIDE = DIMS;       // 14 bytes per INT8 vector

    static final Unsafe UNSAFE;

    static {
        try {
            final Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    // ── Off-heap buffers ──────────────────────────────────────────────────────

    final ByteBuffer vectorsBuf;   // N × 14 INT8 bytes
    final ByteBuffer labelsBuf;    // N bytes
    final long vectorsAddr;
    final long labelsAddr;
    private int size;

    // ── IVF index (on-heap) ───────────────────────────────────────────────────

    float[] centroids;     // numClusters × DIMS floats (float centroids for distance)
    int[]   listOffsets;   // start offset in listData for each cluster
    int[]   listSizes;     // number of vectors in each cluster
    int[]   listData;      // flat array of vector indices, grouped by cluster
    int     numClusters;
    int     defaultNprobe;

    // ── Per-cluster bounding boxes (INT8 scale) ───────────────────────────────
    // Flat arrays: index [c*DIMS + d] gives the min/max INT8 value of dimension d
    // across all vectors assigned to cluster c.  Same signed-byte scale as distSqInt8.
    byte[]  bboxMinInt8;   // C × DIMS — per-dim minimum
    byte[]  bboxMaxInt8;   // C × DIMS — per-dim maximum

    // ── Construction ──────────────────────────────────────────────────────────

    OffHeapVectorStore() { this(CAPACITY); }

    OffHeapVectorStore(int capacity) {
        vectorsBuf  = ByteBuffer.allocateDirect(capacity * VEC_STRIDE)
                                .order(ByteOrder.nativeOrder());
        labelsBuf   = ByteBuffer.allocateDirect(capacity);
        vectorsAddr = MemorySegment.ofBuffer(vectorsBuf).address();
        labelsAddr  = MemorySegment.ofBuffer(labelsBuf) .address();
    }

    int  size()           { return size; }
    void forceSize(int n) { size = n; }

    // ── Label read ────────────────────────────────────────────────────────────

    byte getLabel(final int index) {
        return UNSAFE.getByte(labelsAddr + index);
    }

    // ── INT8 distance — core hot path ─────────────────────────────────────────

    /**
     * Squared Euclidean distance (INT8 scale) between {@code query} and the
     * stored vector at {@code idx}, with early termination.
     *
     * Values are signed bytes in [-127, 127].
     * Max possible result: (127+127)^2 × 14 = 903,224 — fits in int.
     *
     * Returns a partial sum (≥ threshold) when early termination fires.
     * The caller only uses the return value to check d < threshold, so a
     * partial sum still correctly rejects the vector.
     */
    int distSqInt8(final byte[] query, final int idx, final int threshold) {
        final long base = vectorsAddr + (long)idx * VEC_STRIDE;
        int sum = 0, d;
        d = (int)query[0]  - (int)UNSAFE.getByte(base);     sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[1]  - (int)UNSAFE.getByte(base+1);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[2]  - (int)UNSAFE.getByte(base+2);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[3]  - (int)UNSAFE.getByte(base+3);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[4]  - (int)UNSAFE.getByte(base+4);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[5]  - (int)UNSAFE.getByte(base+5);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[6]  - (int)UNSAFE.getByte(base+6);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[7]  - (int)UNSAFE.getByte(base+7);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[8]  - (int)UNSAFE.getByte(base+8);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[9]  - (int)UNSAFE.getByte(base+9);   sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[10] - (int)UNSAFE.getByte(base+10);  sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[11] - (int)UNSAFE.getByte(base+11);  sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[12] - (int)UNSAFE.getByte(base+12);  sum += d*d; if (sum >= threshold) return sum;
        d = (int)query[13] - (int)UNSAFE.getByte(base+13);  sum += d*d;
        return sum;
    }

    // ── Bounding-box construction ─────────────────────────────────────────────

    /**
     * Scans every cluster and records the per-dimension INT8 min/max.
     * Called once after the IVF index is fully loaded.
     * V4: vectors are cluster-ordered (no listData).
     * V3: vector positions are looked up via listData.
     */
    void buildBboxes() {
        final int C = numClusters;
        bboxMinInt8 = new byte[C * DIMS];
        bboxMaxInt8 = new byte[C * DIMS];

        // Initialise: min = +127, max = -127 (full INT8 range)
        for (int i = 0, n = C * DIMS; i < n; i++) {
            bboxMinInt8[i] = (byte)  127;
            bboxMaxInt8[i] = (byte) -127;
        }

        if (listData == null) {
            // V4 — cluster-ordered: offset is the physical start index
            for (int c = 0; c < C; c++) {
                final int off   = listOffsets[c];
                final int sz    = listSizes[c];
                final int bbase = c * DIMS;
                for (int li = 0; li < sz; li++) {
                    final long vbase = vectorsAddr + (long)(off + li) * VEC_STRIDE;
                    for (int d = 0; d < DIMS; d++) {
                        final byte v = UNSAFE.getByte(vbase + d);
                        if (v < bboxMinInt8[bbase + d]) bboxMinInt8[bbase + d] = v;
                        if (v > bboxMaxInt8[bbase + d]) bboxMaxInt8[bbase + d] = v;
                    }
                }
            }
        } else {
            // V3 — indirect via listData
            for (int c = 0; c < C; c++) {
                final int off   = listOffsets[c];
                final int sz    = listSizes[c];
                final int bbase = c * DIMS;
                for (int li = 0; li < sz; li++) {
                    final int  idx   = listData[off + li];
                    final long vbase = vectorsAddr + (long)idx * VEC_STRIDE;
                    for (int d = 0; d < DIMS; d++) {
                        final byte v = UNSAFE.getByte(vbase + d);
                        if (v < bboxMinInt8[bbase + d]) bboxMinInt8[bbase + d] = v;
                        if (v > bboxMaxInt8[bbase + d]) bboxMaxInt8[bbase + d] = v;
                    }
                }
            }
        }
    }

    // ── Write (testing / loading via add()) ───────────────────────────────────

    /**
     * Quantizes {@code vec} to INT8 and stores it at the next free slot.
     * Used only in tests and in paths not performance-critical.
     */
    void add(final float[] vec, final byte label) {
        final long base = vectorsAddr + (long)size * VEC_STRIDE;
        for (int d = 0; d < DIMS; d++) {
            final float f = vec[d];
            int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f));
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
            UNSAFE.putByte(base + d, (byte) q);
        }
        UNSAFE.putByte(labelsAddr + size, label);
        size++;
    }

    /**
     * Copies float vector from native memory into {@code target} (dequantizes INT8→float).
     * Not for hot path — only for debugging and index building.
     */
    void getVector(final int index, final float[] target) {
        final long base = vectorsAddr + (long)index * VEC_STRIDE;
        for (int d = 0; d < DIMS; d++) {
            target[d] = (float)UNSAFE.getByte(base + d) / 127f;
        }
    }
}
