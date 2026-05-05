package org.rinha;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

/**
 * Build-time converter: references.json.gz → references.bin (version 3)
 *
 * Run ONCE during Docker image build — not at runtime.
 *
 * What it does:
 *   1. Parse all 3M vectors from JSON.gz into a flat float[] array
 *   2. Run parallel k-means (C=512 clusters, 15 iterations) to build IVF index
 *   3. Quantize float vectors to INT8
 *   4. Write version 3 binary (INT8 vectors + labels + centroids + IVF lists)
 *
 * Version 3 binary layout (little-endian):
 *   Header  32 bytes: magic | version=3 | count | dims | clusters | nprobe | 0 | 0
 *   INT8 vectors: N × 14 bytes
 *   Labels:       N bytes
 *   Centroids:    C × 14 × 4 bytes (float32)
 *   List sizes:   C × 4 bytes (int32)
 *   List data:    N × 4 bytes (int32, flat IVF lists grouped by cluster)
 *
 * Memory required during conversion: ~185 MB (168 MB float + 3 MB labels + small buffers)
 *
 * Usage:
 *   java -cp app.jar org.rinha.ReferenceConverter \
 *        /data/references.json.gz /data/references.bin
 */
public final class ReferenceConverter {

    static final int C             = 512;   // number of IVF clusters
    static final int MAX_ITER      = 15;    // k-means iterations
    static final int DEFAULT_NPROBE = 32;   // default nprobe at query time
    static final int HEADER_BYTES  = 32;    // v4 header size (same as v3)

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: ReferenceConverter <input.json.gz> <output.bin>");
            System.exit(1);
        }
        convert(args[0], args[1]);
    }

    static void convert(String inPath, String outPath) throws Exception {
        System.out.printf("Converting %s → %s%n", inPath, outPath);
        final long t0 = System.currentTimeMillis();

        // ── Step 1: Parse all vectors ─────────────────────────────────────────
        System.out.println("  Parsing vectors...");
        final int MAX_N = 3_200_000;
        final float[] allVecs   = new float[MAX_N * VectorStore.DIMS]; // 179 MB for 3.2M
        final byte[]  allLabels = new byte[MAX_N];
        int count = 0;

        try (InputStream raw = new GZIPInputStream(
                new BufferedInputStream(new FileInputStream(inPath), 1 << 20))) {
            final Scanner sc = new Scanner(raw);
            while (sc.skipTo(VECTOR_KEY)) {
                sc.skipTo("[");
                if (count >= MAX_N) throw new IllegalStateException("dataset larger than MAX_N");
                final int vi = count * VectorStore.DIMS;
                for (int d = 0; d < VectorStore.DIMS; d++) {
                    if (d > 0) sc.skipTo(",");
                    sc.skipWs();
                    allVecs[vi + d] = sc.readFloat();
                }
                sc.skipTo(LABEL_KEY);
                sc.skipTo("\"");
                allLabels[count++] = sc.nextChar() == 'f' ? VectorStore.FRAUD : VectorStore.LEGIT;
                if (count % 500_000 == 0) System.out.printf("  %,d vectors parsed\r", count);
            }
        }
        System.out.printf("%n  Parsed %,d vectors in %,d ms%n", count, System.currentTimeMillis() - t0);

        final int N = count;

        // ── Step 2: K-means clustering ────────────────────────────────────────
        System.out.printf("  Running k-means (C=%d, iter=%d, parallel)...%n", C, MAX_ITER);
        final long tKm = System.currentTimeMillis();

        final float[] centroids  = initCentroids(allVecs, N);
        final int[]   assignments = new int[N];

        for (int iter = 0; iter < MAX_ITER; iter++) {
            final long ti = System.currentTimeMillis();
            parallelAssign(allVecs, N, centroids, assignments);
            updateCentroids(allVecs, N, centroids, assignments);
            System.out.printf("    iter %2d  %,d ms%n", iter + 1, System.currentTimeMillis() - ti);
        }
        System.out.printf("  K-means done in %,d ms%n", System.currentTimeMillis() - tKm);

        // ── Step 3: Build IVF lists ───────────────────────────────────────────
        System.out.println("  Building IVF lists...");
        final int[] listSizes   = new int[C];
        for (int i = 0; i < N; i++) listSizes[assignments[i]]++;

        final int[] listOffsets = new int[C];
        for (int c = 1; c < C; c++) listOffsets[c] = listOffsets[c-1] + listSizes[c-1];

        // Cluster-ordered permutation: permutation[new_idx] = original_idx
        final int[] permutation = new int[N];
        final int[] cursor      = new int[C];
        for (int i = 0; i < N; i++) {
            final int c = assignments[i];
            permutation[listOffsets[c] + cursor[c]++] = i;
        }

        // Print cluster size stats
        int minSz = Integer.MAX_VALUE, maxSz = 0;
        for (int c = 0; c < C; c++) { minSz = Math.min(minSz, listSizes[c]); maxSz = Math.max(maxSz, listSizes[c]); }
        System.out.printf("  Cluster sizes: min=%d, avg=%d, max=%d%n", minSz, N/C, maxSz);

        // ── Step 4: Write version 4 binary (cluster-ordered, no listData) ─────
        System.out.println("  Writing binary (V4 cluster-ordered)...");
        try (FileChannel out = FileChannel.open(
                Paths.get(outPath),
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Header
            final ByteBuffer hdr = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            hdr.putInt(VectorLoader.MAGIC);
            hdr.putInt(VectorLoader.VERSION_4);
            hdr.putInt(N);
            hdr.putInt(VectorStore.DIMS);
            hdr.putInt(C);
            hdr.putInt(DEFAULT_NPROBE);
            hdr.putInt(0); // flags
            hdr.putInt(0); // reserved
            hdr.flip();
            writeFully(out, hdr);

            // INT8 vectors in cluster order
            final int BATCH = 65_536;
            final ByteBuffer vecBuf = ByteBuffer.allocateDirect(BATCH * VectorStore.DIMS);
            for (int start = 0; start < N; start += BATCH) {
                final int end = Math.min(start + BATCH, N);
                vecBuf.clear();
                for (int ni = start; ni < end; ni++) {
                    final int vi = permutation[ni] * VectorStore.DIMS;
                    for (int d = 0; d < VectorStore.DIMS; d++) {
                        final float f = allVecs[vi + d];
                        int q = (int)(f * 127f + (f >= 0f ? 0.5f : -0.5f));
                        if (q < -127) q = -127;
                        else if (q > 127) q = 127;
                        vecBuf.put((byte) q);
                    }
                }
                vecBuf.flip();
                writeFully(out, vecBuf);
            }

            // Labels in cluster order
            final ByteBuffer lblBuf = ByteBuffer.allocateDirect(Math.min(N, BATCH));
            for (int start = 0; start < N; start += BATCH) {
                final int end = Math.min(start + BATCH, N);
                lblBuf.clear();
                for (int ni = start; ni < end; ni++) lblBuf.put(allLabels[permutation[ni]]);
                lblBuf.flip();
                writeFully(out, lblBuf);
            }

            // Centroids (float32 LE)
            final ByteBuffer centBuf = ByteBuffer.allocateDirect(C * VectorStore.DIMS * Float.BYTES)
                                                  .order(ByteOrder.LITTLE_ENDIAN);
            for (float v : centroids) centBuf.putFloat(v);
            centBuf.flip();
            writeFully(out, centBuf);

            // Cluster sizes
            final ByteBuffer szBuf = ByteBuffer.allocateDirect(C * Integer.BYTES)
                                               .order(ByteOrder.LITTLE_ENDIAN);
            for (int s : listSizes) szBuf.putInt(s);
            szBuf.flip();
            writeFully(out, szBuf);

            // No listData in V4 — sequential access replaces it
        }

        final long totalBytes = HEADER_BYTES + (long)N*VectorStore.DIMS + N
                              + (long)C*VectorStore.DIMS*4L + (long)C*4L;
        System.out.printf("  Done in %,d ms  →  %.1f MB%n",
                System.currentTimeMillis() - t0, totalBytes / 1_048_576.0);
    }

    // ── K-means helpers ───────────────────────────────────────────────────────

    private static float[] initCentroids(float[] allVecs, int N) {
        final float[] centroids = new float[C * VectorStore.DIMS];
        final Random rng = new Random(0x42424242L); // fixed seed for reproducibility
        // Pick C distinct random vectors as initial centroids
        final boolean[] used = new boolean[N];
        for (int c = 0; c < C; c++) {
            int idx;
            do { idx = rng.nextInt(N); } while (used[idx]);
            used[idx] = true;
            System.arraycopy(allVecs, idx * VectorStore.DIMS, centroids, c * VectorStore.DIMS, VectorStore.DIMS);
        }
        return centroids;
    }

    private static void parallelAssign(final float[] allVecs, final int N,
                                        final float[] centroids, final int[] assignments) {
        final int DIMS  = VectorStore.DIMS;
        final int C_LOC = C;
        IntStream.range(0, N).parallel().forEach(i -> {
            final int vi = i * DIMS;
            float bestDist = Float.MAX_VALUE;
            int   best     = 0;
            for (int c = 0; c < C_LOC; c++) {
                final float d = vecDistSq14(allVecs, vi, centroids, c * DIMS);
                if (d < bestDist) { bestDist = d; best = c; }
            }
            assignments[i] = best;
        });
    }

    private static void updateCentroids(final float[] allVecs, final int N,
                                         final float[] centroids, final int[] assignments) {
        final int DIMS = VectorStore.DIMS;
        final double[] sums   = new double[C * DIMS];
        final int[]    counts = new int[C];
        for (int i = 0; i < N; i++) {
            final int c  = assignments[i];
            final int vi = i * DIMS;
            final int ci = c * DIMS;
            counts[c]++;
            for (int d = 0; d < DIMS; d++) sums[ci + d] += allVecs[vi + d];
        }
        for (int c = 0; c < C; c++) {
            if (counts[c] > 0) {
                final int    ci  = c * DIMS;
                final double inv = 1.0 / counts[c];
                for (int d = 0; d < DIMS; d++) centroids[ci + d] = (float)(sums[ci + d] * inv);
            }
        }
    }

    private static float vecDistSq14(float[] a, int ai, float[] b, int bi) {
        final float d0  = a[ai]    - b[bi];
        final float d1  = a[ai+1]  - b[bi+1];
        final float d2  = a[ai+2]  - b[bi+2];
        final float d3  = a[ai+3]  - b[bi+3];
        final float d4  = a[ai+4]  - b[bi+4];
        final float d5  = a[ai+5]  - b[bi+5];
        final float d6  = a[ai+6]  - b[bi+6];
        final float d7  = a[ai+7]  - b[bi+7];
        final float d8  = a[ai+8]  - b[bi+8];
        final float d9  = a[ai+9]  - b[bi+9];
        final float d10 = a[ai+10] - b[bi+10];
        final float d11 = a[ai+11] - b[bi+11];
        final float d12 = a[ai+12] - b[bi+12];
        final float d13 = a[ai+13] - b[bi+13];
        return d0*d0 + d1*d1 + d2*d2  + d3*d3
             + d4*d4 + d5*d5 + d6*d6  + d7*d7
             + d8*d8 + d9*d9 + d10*d10 + d11*d11
             + d12*d12 + d13*d13;
    }

    // ── I/O helpers ──────────────────────────────────────────────────────────

    private static void writeFully(FileChannel fc, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) fc.write(buf);
    }

    // ── Streaming byte scanner (identical to v2) ──────────────────────────────

    private static final byte[] VECTOR_KEY = bytes("\"vector\"");
    private static final byte[] LABEL_KEY  = bytes("\"label\"");

    private static byte[] bytes(String s) { return s.getBytes(StandardCharsets.US_ASCII); }

    private static final class Scanner {
        private final InputStream in;
        private int cur;

        Scanner(InputStream in) throws IOException { this.in = in; advance(); }

        private void advance() throws IOException { cur = in.read(); }

        boolean skipTo(String pattern) throws IOException { return skipTo(bytes(pattern)); }

        boolean skipTo(byte[] pat) throws IOException {
            int matched = 0;
            while (cur != -1) {
                if ((byte) cur == pat[matched]) {
                    if (++matched == pat.length) { advance(); return true; }
                } else {
                    matched = ((byte) cur == pat[0]) ? 1 : 0;
                }
                advance();
            }
            return false;
        }

        void skipWs() throws IOException {
            while (cur != -1 && cur <= ' ') advance();
        }

        char nextChar() throws IOException {
            final char c = (char) cur;
            advance();
            return c;
        }

        float readFloat() throws IOException {
            final boolean neg = (cur == '-');
            if (neg) advance();
            long intPart = 0, fracPart = 0;
            int fracDiv = 1;
            boolean frac = false;
            while (cur != -1 && ((cur >= '0' && cur <= '9') || cur == '.')) {
                if (cur == '.') { frac = true; }
                else if (frac)  { fracPart = fracPart * 10 + (cur - '0'); fracDiv *= 10; }
                else            { intPart  = intPart  * 10 + (cur - '0'); }
                advance();
            }
            final float result = (float) intPart + (float) fracPart / (float) fracDiv;
            return neg ? -result : result;
        }
    }
}
