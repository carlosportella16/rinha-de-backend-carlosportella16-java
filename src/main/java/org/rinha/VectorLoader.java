package org.rinha;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;

/**
 * Loads the preprocessed binary dataset into OffHeapVectorStore.
 *
 * Supported formats:
 *
 *   Version 2 (float32, no IVF) — legacy, kept for compatibility:
 *     Header 16 bytes | Vectors N×56 float LE | Labels N bytes
 *
 *   Version 3 (INT8 + IVF) — production:
 *     Header 32 bytes | INT8 Vectors N×14 | Labels N | Centroids C×56 | Sizes C×4 | Data N×4
 *
 *     Header layout (little-endian):
 *       [0..3]   magic    = 0x52494E48
 *       [4..7]   version  = 3
 *       [8..11]  count    = N
 *       [12..15] dims     = 14
 *       [16..19] clusters = C
 *       [20..23] nprobe   = default nprobe
 *       [24..31] reserved = 0
 */
final class VectorLoader {

    static final int MAGIC     = 0x52494E48;
    static final int VERSION_2 = 2;
    static final int VERSION_3 = 3;
    static final int VERSION_4 = 4;  // cluster-ordered: sequential scan, no listData

    private static final int HEADER_V2 = 16;
    private static final int HEADER_V3 = 32;   // also used for V4

    // ── Public API ────────────────────────────────────────────────────────────

    /** Loads version 3 (INT8 + IVF) binary into an OffHeapVectorStore. */
    static OffHeapVectorStore loadOffHeap(String path) throws IOException {
        try (FileChannel fc = FileChannel.open(Paths.get(path), StandardOpenOption.READ)) {

            // ── 1. Read common 16-byte header to detect version ───────────────
            final ByteBuffer hdr16 = ByteBuffer.allocate(HEADER_V2)
                                               .order(ByteOrder.LITTLE_ENDIAN);
            readFully(fc, hdr16);
            hdr16.flip();

            final int magic   = hdr16.getInt();
            final int version = hdr16.getInt();
            final int count   = hdr16.getInt();
            final int dims    = hdr16.getInt();

            if (magic != MAGIC) throw new IOException("bad magic: 0x" + Integer.toHexString(magic));
            if (dims  != OffHeapVectorStore.DIMS) throw new IOException("bad dims: " + dims);
            if (count <= 0)                       throw new IOException("bad count: " + count);

            if (version == VERSION_4 || version == VERSION_3) {
                // Read the additional 16 bytes unique to V3/V4 header
                final ByteBuffer hdr3 = ByteBuffer.allocate(HEADER_V3 - HEADER_V2)
                                                   .order(ByteOrder.LITTLE_ENDIAN);
                readFully(fc, hdr3);
                hdr3.flip();
                final int clusters = hdr3.getInt();
                final int nprobe   = hdr3.getInt();
                // skip reserved 8 bytes (already read into hdr3)
                if (version == VERSION_4) return loadV4(fc, count, clusters, nprobe);
                return loadV3(fc, count, clusters, nprobe);
            } else if (version == VERSION_2) {
                return loadV2(fc, count);
            } else {
                throw new IOException("unsupported version: " + version);
            }
        }
    }

    // ── Version 4 loader (cluster-ordered INT8, no listData) ─────────────────
    // Vectors and labels are stored in cluster order: all vectors of cluster 0
    // first, then cluster 1, etc. Phase 2 scanning is sequential (no indirection).

    private static OffHeapVectorStore loadV4(FileChannel fc, int N, int C, int nprobe)
            throws IOException {

        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        // ── INT8 vectors (cluster-ordered): N × 14 bytes ─────────────────────
        final ByteBuffer vecBuf = store.vectorsBuf;
        vecBuf.clear().limit(N * OffHeapVectorStore.VEC_STRIDE);
        readFully(fc, vecBuf);

        // ── Labels (cluster-ordered): N bytes ─────────────────────────────────
        final ByteBuffer lblBuf = store.labelsBuf;
        lblBuf.clear().limit(N);
        readFully(fc, lblBuf);

        // ── Centroids: C × DIMS × 4 bytes (float32 LE) ───────────────────────
        final int centBytes = C * OffHeapVectorStore.DIMS * Float.BYTES;
        final ByteBuffer centBuf = ByteBuffer.allocateDirect(centBytes)
                                             .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, centBuf);
        centBuf.flip();
        final float[] centroids = new float[C * OffHeapVectorStore.DIMS];
        centBuf.asFloatBuffer().get(centroids);

        // ── Cluster sizes: C × 4 bytes ────────────────────────────────────────
        final ByteBuffer szBuf = ByteBuffer.allocateDirect(C * Integer.BYTES)
                                           .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, szBuf);
        szBuf.flip();
        final int[] listSizes = new int[C];
        szBuf.asIntBuffer().get(listSizes);

        // ── Cluster offsets (physical start index per cluster) ────────────────
        final int[] listOffsets = new int[C];
        for (int c = 1; c < C; c++) listOffsets[c] = listOffsets[c-1] + listSizes[c-1];

        // ── Publish (no listData in V4) ───────────────────────────────────────
        store.forceSize(N);
        store.centroids     = centroids;
        store.listOffsets   = listOffsets;
        store.listSizes     = listSizes;
        store.listData      = null;      // not used in V4
        store.numClusters   = C;
        store.defaultNprobe = nprobe;

        return store;
    }

    // ── Version 3 loader (INT8 + IVF) ────────────────────────────────────────

    private static OffHeapVectorStore loadV3(FileChannel fc, int N, int C, int nprobe)
            throws IOException {

        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        // ── INT8 vectors: N × 14 bytes ────────────────────────────────────────
        final ByteBuffer vecBuf = store.vectorsBuf;
        vecBuf.clear().limit(N * OffHeapVectorStore.VEC_STRIDE);
        readFully(fc, vecBuf);

        // ── Labels: N bytes ───────────────────────────────────────────────────
        final ByteBuffer lblBuf = store.labelsBuf;
        lblBuf.clear().limit(N);
        readFully(fc, lblBuf);

        // ── Centroids: C × DIMS × 4 bytes (float32 LE) ───────────────────────
        final int centBytes = C * OffHeapVectorStore.DIMS * Float.BYTES;
        final ByteBuffer centBuf = ByteBuffer.allocateDirect(centBytes)
                                             .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, centBuf);
        centBuf.flip();
        final float[] centroids = new float[C * OffHeapVectorStore.DIMS];
        centBuf.asFloatBuffer().get(centroids);

        // ── Cluster sizes: C × 4 bytes ────────────────────────────────────────
        final ByteBuffer szBuf = ByteBuffer.allocateDirect(C * Integer.BYTES)
                                           .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, szBuf);
        szBuf.flip();
        final int[] listSizes = new int[C];
        szBuf.asIntBuffer().get(listSizes);

        // ── Cluster offsets (compute from sizes) ──────────────────────────────
        final int[] listOffsets = new int[C];
        for (int c = 1; c < C; c++) listOffsets[c] = listOffsets[c-1] + listSizes[c-1];

        // ── IVF list data: N × 4 bytes ────────────────────────────────────────
        final ByteBuffer dataBuf = ByteBuffer.allocateDirect(N * Integer.BYTES)
                                             .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, dataBuf);
        dataBuf.flip();
        final int[] listData = new int[N];
        dataBuf.asIntBuffer().get(listData);

        // ── Publish ───────────────────────────────────────────────────────────
        store.forceSize(N);
        store.centroids     = centroids;
        store.listOffsets   = listOffsets;
        store.listSizes     = listSizes;
        store.listData      = listData;
        store.numClusters   = C;
        store.defaultNprobe = nprobe;

        return store;
    }

    // ── Version 2 loader (float32, no IVF) — legacy fallback ─────────────────

    private static OffHeapVectorStore loadV2(FileChannel fc, int N) throws IOException {
        // Version 2: float32 vectors. We quantize on the fly to INT8 during load.
        // This path is slower (needs to quantize) but allows old binary files to work.
        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        final int CHUNK = 1 << 16;
        final ByteBuffer vecBuf = ByteBuffer.allocateDirect(CHUNK * OffHeapVectorStore.DIMS * Float.BYTES)
                                            .order(ByteOrder.LITTLE_ENDIAN);
        final ByteBuffer outBuf = store.vectorsBuf;
        outBuf.clear();

        int loaded = 0;
        while (loaded < N) {
            final int rows = Math.min(CHUNK, N - loaded);
            final int need = rows * OffHeapVectorStore.DIMS * Float.BYTES;
            vecBuf.clear().limit(need);
            readFully(fc, vecBuf);
            vecBuf.flip();
            // Quantize float→INT8
            for (int i = 0; i < rows * OffHeapVectorStore.DIMS; i++) {
                final float f = vecBuf.getFloat();
                int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f));
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                outBuf.put((byte) q);
            }
            loaded += rows;
        }

        final ByteBuffer lblBuf = store.labelsBuf;
        lblBuf.clear().limit(N);
        readFully(fc, lblBuf);

        store.forceSize(N);
        // No IVF for v2 — caller must build or use brute-force fallback
        store.defaultNprobe = 0; // signals: no IVF available
        return store;
    }

    // ── I/O helper ────────────────────────────────────────────────────────────

    static void readFully(FileChannel fc, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            final int n = fc.read(buf);
            if (n == -1) throw new IOException("unexpected EOF");
        }
    }
}
