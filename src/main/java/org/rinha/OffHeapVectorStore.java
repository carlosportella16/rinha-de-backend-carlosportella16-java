package org.rinha;

/**
 * IVF vector store with cluster-ordered flat byte[] storage.
 *
 * Memory layout (V4 / cluster-ordered):
 *   vectors:  N × 14 bytes (INT8, on-heap flat array) — 42 MB for 3M vecs
 *   labels:   N bytes      (on-heap flat array)        —  3 MB for 3M vecs
 *
 * In V4 (and V3 after load-time reorder), vectors[c_off * DIMS .. (c_off + c_sz) * DIMS)
 * holds every vector of cluster c contiguously — no indirection, simple index arithmetic,
 * prefetcher-friendly sequential stride of VEC_STRIDE bytes.
 *
 * IVF index arrays (on-heap):
 *   centroids:   C × DIMS floats = 28 KB for C=512
 *   listOffsets: C ints          =  2 KB   (vector index, not byte offset)
 *   listSizes:   C ints          =  2 KB
 *   listData:    null — always null after cluster-ordered layout is established
 *
 * INT8 encoding: float f → byte = round(f × 127), clamped to [-127, 127].
 * Sentinel -1.0 → -127.
 */
final class OffHeapVectorStore {

    static final int  DIMS       = 14;
    static final int  CAPACITY   = 3_000_000;
    static final byte FRAUD      = 1;
    static final byte LEGIT      = 0;
    static final int  VEC_STRIDE = DIMS;   // 14 bytes per INT8 vector

    // ── Flat on-heap storage ──────────────────────────────────────────────────
    // Single contiguous byte[] per field — no object graph, no pointer chasing.
    // Cluster c occupies vectors[listOffsets[c]*DIMS .. (listOffsets[c]+listSizes[c])*DIMS).

    final byte[] vectors;   // N × DIMS INT8 bytes, cluster-ordered
    final byte[] labels;    // N bytes,            cluster-ordered
    private int size;

    // ── IVF index (on-heap flat arrays) ──────────────────────────────────────

    float[] centroids;     // C × DIMS floats
    int[]   listOffsets;   // start *vector index* in vectors[] for each cluster
    int[]   listSizes;     // number of vectors in each cluster
    int[]   listData;      // V3 intermediate only; always null after buildFlatLayout()
    int     numClusters;
    int     defaultNprobe;

    // ── Per-cluster bounding boxes (INT8 scale, flat arrays) ─────────────────
    // bboxMinInt8[c*DIMS + d] = minimum INT8 value of dimension d in cluster c.
    byte[] bboxMinInt8;   // C × DIMS
    byte[] bboxMaxInt8;   // C × DIMS

    // ── Construction ──────────────────────────────────────────────────────────

    OffHeapVectorStore() { this(CAPACITY); }

    OffHeapVectorStore(int capacity) {
        vectors = new byte[capacity * VEC_STRIDE];
        labels  = new byte[capacity];
    }

    int  size()           { return size; }
    void forceSize(int n) { size = n; }

    // ── Label read ────────────────────────────────────────────────────────────

    byte getLabel(final int index) {
        return labels[index];
    }

    // ── INT8 distance — core hot path ─────────────────────────────────────────

    /**
     * Squared Euclidean distance (INT8 scale) between {@code query} and the
     * stored vector at {@code idx}, with early termination.
     *
     * {@code idx} is a vector index; byte offset = idx × VEC_STRIDE.
     * Vectors are cluster-ordered, so sequential calls with idx = off, off+1, off+2, …
     * access contiguous memory — optimal for CPU prefetching and JIT vectorisation.
     *
     * Max result: (127+127)² × 14 = 903,224 — fits in int.
     */
    int distSqInt8(final byte[] query, final int idx, final int threshold) {
        final int base = idx * VEC_STRIDE;
        int sum = 0, d;
        d = query[0]  - vectors[base];     sum += d*d; if (sum >= threshold) return sum;
        d = query[1]  - vectors[base+1];   sum += d*d; if (sum >= threshold) return sum;
        d = query[2]  - vectors[base+2];   sum += d*d; if (sum >= threshold) return sum;
        d = query[3]  - vectors[base+3];   sum += d*d; if (sum >= threshold) return sum;
        d = query[4]  - vectors[base+4];   sum += d*d; if (sum >= threshold) return sum;
        d = query[5]  - vectors[base+5];   sum += d*d; if (sum >= threshold) return sum;
        d = query[6]  - vectors[base+6];   sum += d*d; if (sum >= threshold) return sum;
        d = query[7]  - vectors[base+7];   sum += d*d; if (sum >= threshold) return sum;
        d = query[8]  - vectors[base+8];   sum += d*d; if (sum >= threshold) return sum;
        d = query[9]  - vectors[base+9];   sum += d*d; if (sum >= threshold) return sum;
        d = query[10] - vectors[base+10];  sum += d*d; if (sum >= threshold) return sum;
        d = query[11] - vectors[base+11];  sum += d*d; if (sum >= threshold) return sum;
        d = query[12] - vectors[base+12];  sum += d*d; if (sum >= threshold) return sum;
        d = query[13] - vectors[base+13];  sum += d*d;
        return sum;
    }

    // ── Bounding-box construction ─────────────────────────────────────────────

    /**
     * Scans every cluster and records the per-dimension INT8 min/max.
     * Must be called after buildFlatLayout() so listData is null and
     * vectors[] is in cluster order.
     */
    void buildBboxes() {
        final int C = numClusters;
        bboxMinInt8 = new byte[C * DIMS];
        bboxMaxInt8 = new byte[C * DIMS];
        for (int i = 0, n = C * DIMS; i < n; i++) {
            bboxMinInt8[i] = (byte)  127;
            bboxMaxInt8[i] = (byte) -127;
        }
        for (int c = 0; c < C; c++) {
            final int off   = listOffsets[c];
            final int sz    = listSizes[c];
            final int bbase = c * DIMS;
            for (int li = 0; li < sz; li++) {
                final int vbase = (off + li) * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    final byte v = vectors[vbase + d];
                    if (v < bboxMinInt8[bbase + d]) bboxMinInt8[bbase + d] = v;
                    if (v > bboxMaxInt8[bbase + d]) bboxMaxInt8[bbase + d] = v;
                }
            }
        }
    }

    /**
     * Reorders vectors and labels from original-index order (V3) to cluster-
     * sequential order, then clears listData.
     * After this call the store layout is identical to V4.
     *
     * @param tmpVec    raw INT8 vectors in original-index order (N × DIMS bytes)
     * @param tmpLbl    raw labels in original-index order (N bytes)
     */
    void buildFlatLayout(final byte[] tmpVec, final byte[] tmpLbl) {
        // listData[ni] = original index for cluster-position ni
        final int[] ld = listData;
        for (int ni = 0; ni < size; ni++) {
            final int origIdx = ld[ni];
            System.arraycopy(tmpVec, origIdx * DIMS, vectors, ni * DIMS, DIMS);
            labels[ni] = tmpLbl[origIdx];
        }
        listData = null;   // cluster-ordered now — same as V4, no indirection
    }

    // ── Write (test helper) ───────────────────────────────────────────────────

    /**
     * Quantizes {@code vec} to INT8 and stores it at the next free slot.
     * Used only in tests and non-performance-critical paths.
     */
    void add(final float[] vec, final byte label) {
        final int base = size * VEC_STRIDE;
        for (int d = 0; d < DIMS; d++) {
            final float f = vec[d];
            int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f));
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
            vectors[base + d] = (byte) q;
        }
        labels[size] = label;
        size++;
    }

    /** Dequantizes stored INT8 vector → float. Not for hot path. */
    void getVector(final int index, final float[] target) {
        final int base = index * VEC_STRIDE;
        for (int d = 0; d < DIMS; d++) {
            target[d] = (float) vectors[base + d] / 127f;
        }
    }
}
