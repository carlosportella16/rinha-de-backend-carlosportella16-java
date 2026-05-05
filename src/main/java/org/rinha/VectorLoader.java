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
 *   Version 3 (INT8 + IVF, original order):
 *     Header 32 bytes | INT8 Vectors N×14 | Labels N | Centroids C×56 | Sizes C×4 | Data N×4
 *
 *   Version 4 (INT8 + IVF, cluster-ordered — production):
 *     Header 32 bytes | INT8 Vectors N×14 (cluster order) | Labels N (cluster order)
 *                     | Centroids C×56 | Sizes C×4
 *
 * After loading, both V3 and V4 produce the same in-memory layout:
 *   store.vectors[]     — flat byte[], N × DIMS, cluster-ordered
 *   store.labels[]      — flat byte[], N, cluster-ordered
 *   store.listData      — always null (indirection eliminated at load time for V3)
 *   store.listOffsets[] — start vector-index per cluster
 *   store.listSizes[]   — count per cluster
 *
 *     Header layout (V3/V4, little-endian):
 *       [0..3]   magic    = 0x52494E48
 *       [4..7]   version  = 3 or 4
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

    /** Loads binary dataset into an OffHeapVectorStore with flat cluster-ordered layout. */
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

    // ── Version 4 loader (cluster-ordered, direct flat-array fill) ───────────
    // Vectors and labels are already in cluster order in the file.
    // Read directly into store.vectors[] and store.labels[] — zero extra copies.

    private static OffHeapVectorStore loadV4(FileChannel fc, int N, int C, int nprobe)
            throws IOException {

        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        // ── INT8 vectors (cluster-ordered): N × DIMS bytes ───────────────────
        // ByteBuffer.wrap() uses store.vectors as backing array — no extra copy.
        readFully(fc, ByteBuffer.wrap(store.vectors, 0, N * OffHeapVectorStore.VEC_STRIDE));

        // ── Labels (cluster-ordered): N bytes ────────────────────────────────
        readFully(fc, ByteBuffer.wrap(store.labels, 0, N));

        // ── Centroids: C × DIMS × 4 bytes (float32 LE) ───────────────────────
        final float[] centroids = readCentroids(fc, C);

        // ── Cluster sizes: C × 4 bytes ────────────────────────────────────────
        final int[] listSizes = readSizes(fc, C);

        // ── Cluster offsets (physical start vector-index per cluster) ─────────
        final int[] listOffsets = buildOffsets(listSizes, C);

        // ── Publish ───────────────────────────────────────────────────────────
        store.forceSize(N);
        store.centroids     = centroids;
        store.listOffsets   = listOffsets;
        store.listSizes     = listSizes;
        store.listData      = null;      // cluster-ordered, no indirection needed
        store.numClusters   = C;
        store.defaultNprobe = nprobe;

        store.buildBboxes();

        return store;
    }

    // ── Version 3 loader (original order → reorder to cluster-sequential) ────
    // V3 files store vectors/labels in original (un-clustered) order with a
    // listData[] array for indirection.  We reorder into cluster-sequential layout
    // at load time so that Phase 2 scanning is identical to V4 — no pointer chasing.

    private static OffHeapVectorStore loadV3(FileChannel fc, int N, int C, int nprobe)
            throws IOException {

        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        // ── INT8 vectors (original order) — buffer separately for reorder ─────
        final byte[] tmpVec = new byte[N * OffHeapVectorStore.VEC_STRIDE];
        readFully(fc, ByteBuffer.wrap(tmpVec));

        // ── Labels (original order) ───────────────────────────────────────────
        final byte[] tmpLbl = new byte[N];
        readFully(fc, ByteBuffer.wrap(tmpLbl));

        // ── Centroids: C × DIMS × 4 bytes ────────────────────────────────────
        final float[] centroids = readCentroids(fc, C);

        // ── Cluster sizes ─────────────────────────────────────────────────────
        final int[] listSizes = readSizes(fc, C);

        // ── Cluster offsets ───────────────────────────────────────────────────
        final int[] listOffsets = buildOffsets(listSizes, C);

        // ── IVF list data: N × 4 bytes ────────────────────────────────────────
        final ByteBuffer dataBuf = ByteBuffer.allocateDirect(N * Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, dataBuf);
        dataBuf.flip();
        final int[] listData = new int[N];
        dataBuf.asIntBuffer().get(listData);

        // ── Publish then reorder to cluster-sequential layout ─────────────────
        store.forceSize(N);
        store.centroids     = centroids;
        store.listOffsets   = listOffsets;
        store.listSizes     = listSizes;
        store.listData      = listData;   // temporarily set for buildFlatLayout()
        store.numClusters   = C;
        store.defaultNprobe = nprobe;

        // Reorder: copy tmpVec/tmpLbl in listData order into store.vectors/labels.
        // Sets listData = null when done — identical layout to V4.
        store.buildFlatLayout(tmpVec, tmpLbl);

        store.buildBboxes();

        return store;
    }

    // ── Version 2 loader (float32, no IVF) — legacy fallback ─────────────────

    private static OffHeapVectorStore loadV2(FileChannel fc, int N) throws IOException {
        final OffHeapVectorStore store = new OffHeapVectorStore(N);

        final int CHUNK = 1 << 16;
        final ByteBuffer vecBuf = ByteBuffer.allocateDirect(CHUNK * OffHeapVectorStore.DIMS * Float.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        int loaded = 0;
        int outPos = 0;
        while (loaded < N) {
            final int rows = Math.min(CHUNK, N - loaded);
            final int need = rows * OffHeapVectorStore.DIMS * Float.BYTES;
            vecBuf.clear().limit(need);
            readFully(fc, vecBuf);
            vecBuf.flip();
            // Quantize float→INT8 directly into store.vectors[]
            for (int i = 0; i < rows * OffHeapVectorStore.DIMS; i++) {
                final float f = vecBuf.getFloat();
                int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f));
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                store.vectors[outPos++] = (byte) q;
            }
            loaded += rows;
        }

        // Labels
        readFully(fc, ByteBuffer.wrap(store.labels, 0, N));

        store.forceSize(N);
        store.defaultNprobe = 0; // signals: no IVF available
        return store;
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    private static float[] readCentroids(FileChannel fc, int C) throws IOException {
        final int centBytes = C * OffHeapVectorStore.DIMS * Float.BYTES;
        final ByteBuffer centBuf = ByteBuffer.allocateDirect(centBytes)
                .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, centBuf);
        centBuf.flip();
        final float[] centroids = new float[C * OffHeapVectorStore.DIMS];
        centBuf.asFloatBuffer().get(centroids);
        return centroids;
    }

    private static int[] readSizes(FileChannel fc, int C) throws IOException {
        final ByteBuffer szBuf = ByteBuffer.allocateDirect(C * Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        readFully(fc, szBuf);
        szBuf.flip();
        final int[] listSizes = new int[C];
        szBuf.asIntBuffer().get(listSizes);
        return listSizes;
    }

    private static int[] buildOffsets(int[] listSizes, int C) {
        final int[] listOffsets = new int[C];
        for (int c = 1; c < C; c++) listOffsets[c] = listOffsets[c-1] + listSizes[c-1];
        return listOffsets;
    }

    // ── I/O helper ────────────────────────────────────────────────────────────

    static void readFully(FileChannel fc, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            final int n = fc.read(buf);
            if (n == -1) throw new IOException("unexpected EOF");
        }
    }
}
