package org.rinha;

/**
 * Standalone correctness tests for KnnSearch.
 *
 * Run with:
 *   java -cp target/classes:target/test-classes org.rinha.KnnSearchTest
 *
 * No framework needed — plain assertions with descriptive failure messages.
 * Each test is isolated: fresh VectorStore, fresh scratch arrays.
 */
public final class KnnSearchTest {

    private static int PASS = 0;
    private static int FAIL = 0;

    public static void main(String[] args) {

        // ── insert() edge cases ───────────────────────────────────────────────

        run("insert: new element becomes worst (pos=4)",         KnnSearchTest::checkInsertWorst);
        run("insert: new element becomes best (pos=0)",          KnnSearchTest::checkInsertBest);
        run("insert: new element lands in middle (pos=2)",       KnnSearchTest::checkInsertMiddle);
        run("insert: first call populates slot 0",               KnnSearchTest::checkInsertFirstSlot);
        run("insert: equal distance goes after equal (stable)",  KnnSearchTest::checkInsertEqual);

        // ── search() (brute-force) correctness ───────────────────────────────

        run("search: exact match is always closest",             KnnSearchTest::checkExactMatch);
        run("search: top-5 sorted ascending",                    KnnSearchTest::checkSortedOrder);
        run("search: all 5 found in tiny store of 5",            KnnSearchTest::checkTinyStore);
        run("search: works correctly with store of size 1",      KnnSearchTest::checkSingleVector);
        run("search: worst is discarded when better found",      KnnSearchTest::checkWorstDiscarded);

        // ── searchIndexed() correctness (same oracle, indexed path) ──────────

        run("searchIndexed: exact match",                        KnnSearchTest::checkIdxExactMatch);
        run("searchIndexed: top-5 sorted ascending",             KnnSearchTest::checkIdxSortedOrder);
        run("searchIndexed: all 5 found in tiny store",          KnnSearchTest::checkIdxTinyStore);
        run("searchIndexed: store of size 1",                    KnnSearchTest::checkIdxSingleVector);
        run("searchIndexed: worst discarded",                    KnnSearchTest::checkIdxWorstDiscarded);
        run("searchIndexed: matches brute-force on 100 vectors", KnnSearchTest::checkIdxMatchesBrute);
        run("buildDim0Index: sortedDim0 is non-decreasing",      KnnSearchTest::checkIndexSorted);
        run("buildDim0Index: sortedByDim0 maps back correctly",  KnnSearchTest::checkIndexMapping);

        // ── fraudCount() ─────────────────────────────────────────────────────────

        run("fraudCount: all legit  → 0",                        KnnSearchTest::checkFraudCountZero);
        run("fraudCount: all fraud  → 5",                        KnnSearchTest::checkFraudCountFive);
        run("fraudCount: mixed 2/5  → 2",                        KnnSearchTest::checkFraudCountMixed);
        run("fraudCount: -1 indices ignored",                    KnnSearchTest::checkFraudCountUnfilled);

        // ── fraudScore() ──────────────────────────────────────────────────────

        run("fraudScore: 0 fraud  → 0.0",                        KnnSearchTest::checkScoreZero);
        run("fraudScore: 5 fraud  → 1.0",                        KnnSearchTest::checkScoreFive);
        run("fraudScore: 1 fraud  → 0.2",                        KnnSearchTest::checkScoreOne);
        run("fraudScore: 2 fraud  → 0.4",                        KnnSearchTest::checkScoreTwo);
        run("fraudScore: 3 fraud  → 0.6",                        KnnSearchTest::checkScoreThree);
        run("fraudScore: 4 fraud  → 0.8",                        KnnSearchTest::checkScoreFour);
        run("fraudScore: -1 slots ignored",                      KnnSearchTest::checkScoreUnfilled);
        run("fraudScore: consistent with fraudCount",            KnnSearchTest::checkScoreConsistency);

        // ── spec examples ─────────────────────────────────────────────────────

        run("spec: legit tx → top-5 all legit → score 0.0",     KnnSearchTest::checkSpecLegit);
        run("spec: fraud tx → top-5 all fraud → score 1.0",     KnnSearchTest::checkSpecFraud);
        run("spec: threshold 0.6 → approved boundary",          KnnSearchTest::checkThreshold);

        // ── summary ──────────────────────────────────────────────────────────

        System.out.printf("%n%d passed, %d failed%n", PASS, FAIL);
        if (FAIL > 0) System.exit(1);
    }

    // ── insert edge cases ─────────────────────────────────────────────────────

    private static void checkInsertWorst() {
        float[] d = {1f, 2f, 3f, 4f, 5f};
        int[]   i = {10, 20, 30, 40, 50};
        KnnSearch.insert(d, i, 4.5f, 99);
        assertEq("dist[4]", 4.5f, d[4]);
        assertEq("idx[4]",  99,   i[4]);
        assertEq("dist[3]", 4f,   d[3]);
    }

    private static void checkInsertBest() {
        float[] d = {2f, 3f, 4f, 5f, 6f};
        int[]   i = {10, 20, 30, 40, 50};
        KnnSearch.insert(d, i, 0.5f, 99);
        assertEq("dist[0]", 0.5f, d[0]);
        assertEq("idx[0]",  99,   i[0]);
        assertEq("dist[1]", 2f,   d[1]);
        assertEq("idx[1]",  10,   i[1]);
        assertEq("dist[4]", 5f,   d[4]);
    }

    private static void checkInsertMiddle() {
        float[] d = {1f, 2f, 4f, 5f, 6f};
        int[]   i = {10, 20, 30, 40, 50};
        KnnSearch.insert(d, i, 3f, 99);
        assertEq("dist[2]", 3f,   d[2]);
        assertEq("idx[2]",  99,   i[2]);
        assertEq("dist[3]", 4f,   d[3]);
        assertEq("dist[4]", 5f,   d[4]);
    }

    private static void checkInsertFirstSlot() {
        float[] d = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE};
        int[]   i = {-1, -1, -1, -1, -1};
        KnnSearch.insert(d, i, 0.1f, 7);
        assertEq("dist[0]", 0.1f, d[0]);
        assertEq("idx[0]",  7,    i[0]);
        assertTrue("rest still MAX", d[1] == Float.MAX_VALUE);
    }

    private static void checkInsertEqual() {
        float[] d = {1f, 2f, 3f, 4f, 5f};
        int[]   i = {10, 20, 30, 40, 50};
        KnnSearch.insert(d, i, 3f, 99);
        assertEq("dist[2]", 3f,  d[2]);
        assertEq("dist[3]", 3f,  d[3]);
        assertTrue("one of them is 99", i[2] == 30 && i[3] == 99 || i[2] == 99 && i[3] == 30);
    }

    // ── search() (brute-force) ────────────────────────────────────────────────

    private static void checkExactMatch() {
        VectorStore store = storeOf(
                new float[]{1f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT,
                new float[]{2f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT,
                new float[]{3f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT
        );
        float[] query = {2f,0,0,0,0,0,0,0,0,0,0,0,0,0};
        int[] idx = runSearch(store, query);
        assertEq("closest idx", 1, idx[0]);
        assertEq("closest dist", 0f, VectorStore.distSq(query, store.vectors[idx[0]]));
    }

    private static void checkSortedOrder() {
        VectorStore store = new VectorStore(10);
        float[] scratch = new float[VectorStore.DIMS];
        for (int i = 0; i < 10; i++) { scratch[0] = (float)(9 - i); store.add(scratch, VectorStore.LEGIT); }
        float[] query = new float[VectorStore.DIMS];
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.search(store, query, topDist, topIdx);
        for (int i = 0; i < KnnSearch.K - 1; i++) assertTrue("sorted at " + i, topDist[i] <= topDist[i + 1]);
        assertEq("idx[0] is vector 9 (dist=0)", 9, topIdx[0]);
        assertEq("idx[1] is vector 8 (dist=1)", 8, topIdx[1]);
    }

    private static void checkTinyStore() {
        VectorStore store = new VectorStore(5);
        float[] scratch = new float[VectorStore.DIMS];
        for (int i = 0; i < 5; i++) { scratch[0] = i; store.add(scratch, VectorStore.LEGIT); }
        int[] idx = runSearch(store, new float[VectorStore.DIMS]);
        for (int i = 0; i < KnnSearch.K; i++) assertTrue("idx[" + i + "] >= 0", idx[i] >= 0);
    }

    private static void checkSingleVector() {
        VectorStore store = storeOf(new float[]{0.5f,0.5f,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.FRAUD);
        int[] idx = runSearch(store, new float[VectorStore.DIMS]);
        assertEq("only vector found", 0, idx[0]);
        for (int i = 1; i < KnnSearch.K; i++) assertEq("slot " + i + " is -1", -1, idx[i]);
    }

    private static void checkWorstDiscarded() {
        VectorStore store = new VectorStore(6);
        float[] scratch = new float[VectorStore.DIMS];
        float[] dists = {10f, 9f, 8f, 7f, 6f, 1f};
        for (int i = 0; i < 6; i++) { scratch[0] = dists[i]; store.add(scratch, VectorStore.LEGIT); }
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.search(store, new float[VectorStore.DIMS], topDist, topIdx);
        assertEq("best  = vector 5 (d²=1)",  5, topIdx[0]);
        assertEq("worst = vector 1 (d²=81)", 1, topIdx[4]);
        for (int i = 0; i < KnnSearch.K; i++) assertTrue("vector 0 excluded", topIdx[i] != 0);
    }

    // ── searchIndexed() ───────────────────────────────────────────────────────

    private static void checkIdxExactMatch() {
        VectorStore store = storeOf(
                new float[]{1f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT,
                new float[]{2f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT,
                new float[]{3f,0,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.LEGIT
        );
        store.buildDim0Index();
        float[] query = {2f,0,0,0,0,0,0,0,0,0,0,0,0,0};
        int[] idx = runIndexed(store, query);
        assertEq("closest idx", 1, idx[0]);
        assertEq("closest dist", 0f, VectorStore.distSq(query, store.vectors[idx[0]]));
    }

    private static void checkIdxSortedOrder() {
        VectorStore store = new VectorStore(10);
        float[] scratch = new float[VectorStore.DIMS];
        for (int i = 0; i < 10; i++) { scratch[0] = (float)(9 - i); store.add(scratch, VectorStore.LEGIT); }
        store.buildDim0Index();
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.searchIndexed(store, new float[VectorStore.DIMS], topDist, topIdx);
        for (int i = 0; i < KnnSearch.K - 1; i++) assertTrue("sorted at " + i, topDist[i] <= topDist[i + 1]);
        assertEq("idx[0] is vector 9", 9, topIdx[0]);
        assertEq("idx[1] is vector 8", 8, topIdx[1]);
    }

    private static void checkIdxTinyStore() {
        VectorStore store = new VectorStore(5);
        float[] scratch = new float[VectorStore.DIMS];
        for (int i = 0; i < 5; i++) { scratch[0] = i; store.add(scratch, VectorStore.LEGIT); }
        store.buildDim0Index();
        int[] idx = runIndexed(store, new float[VectorStore.DIMS]);
        for (int i = 0; i < KnnSearch.K; i++) assertTrue("idx[" + i + "] >= 0", idx[i] >= 0);
    }

    private static void checkIdxSingleVector() {
        VectorStore store = storeOf(new float[]{0.5f,0.5f,0,0,0,0,0,0,0,0,0,0,0,0}, VectorStore.FRAUD);
        store.buildDim0Index();
        int[] idx = runIndexed(store, new float[VectorStore.DIMS]);
        assertEq("only vector found", 0, idx[0]);
        for (int i = 1; i < KnnSearch.K; i++) assertEq("slot " + i + " is -1", -1, idx[i]);
    }

    private static void checkIdxWorstDiscarded() {
        VectorStore store = new VectorStore(6);
        float[] scratch = new float[VectorStore.DIMS];
        float[] dists = {10f, 9f, 8f, 7f, 6f, 1f};
        for (int i = 0; i < 6; i++) { scratch[0] = dists[i]; store.add(scratch, VectorStore.LEGIT); }
        store.buildDim0Index();
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.searchIndexed(store, new float[VectorStore.DIMS], topDist, topIdx);
        assertEq("best  = vector 5 (d²=1)",  5, topIdx[0]);
        assertEq("worst = vector 1 (d²=81)", 1, topIdx[4]);
        for (int i = 0; i < KnnSearch.K; i++) assertTrue("vector 0 excluded", topIdx[i] != 0);
    }

    /** searchIndexed must return identical results to brute-force search. */
    private static void checkIdxMatchesBrute() {
        // 100 randomly-ish spaced vectors
        VectorStore store = new VectorStore(100);
        float[] v = new float[VectorStore.DIMS];
        for (int i = 0; i < 100; i++) {
            // Use a pseudo-random spread across all 14 dims
            for (int d = 0; d < VectorStore.DIMS; d++) v[d] = ((i * 37 + d * 13) % 100) / 100f;
            store.add(v, (i % 3 == 0) ? VectorStore.FRAUD : VectorStore.LEGIT);
        }
        store.buildDim0Index();

        float[] query = new float[VectorStore.DIMS];
        for (int d = 0; d < VectorStore.DIMS; d++) query[d] = 0.5f;

        float[] bDist = new float[KnnSearch.K]; int[] bIdx = new int[KnnSearch.K];
        float[] iDist = new float[KnnSearch.K]; int[] iIdx = new int[KnnSearch.K];
        KnnSearch.search(store, query, bDist, bIdx);
        KnnSearch.searchIndexed(store, query, iDist, iIdx);

        for (int i = 0; i < KnnSearch.K; i++) {
            assertEq("dist[" + i + "] matches", bDist[i], iDist[i]);
            assertEq("idx["  + i + "] matches", bIdx[i],  iIdx[i]);
        }
    }

    private static void checkIndexSorted() {
        VectorStore store = new VectorStore(20);
        float[] v = new float[VectorStore.DIMS];
        // Add vectors with dim-0 in reverse order: 19, 18, ..., 0
        for (int i = 19; i >= 0; i--) { v[0] = i; store.add(v, VectorStore.LEGIT); }
        store.buildDim0Index();
        for (int i = 0; i < store.size() - 1; i++) {
            assertTrue("sortedDim0[" + i + "] <= sortedDim0[" + (i+1) + "]",
                    store.sortedDim0[i] <= store.sortedDim0[i + 1]);
        }
    }

    private static void checkIndexMapping() {
        VectorStore store = new VectorStore(5);
        float[] v = new float[VectorStore.DIMS];
        float[] dim0vals = {3f, 1f, 4f, 1f, 5f};
        for (int i = 0; i < 5; i++) { v[0] = dim0vals[i]; store.add(v, VectorStore.LEGIT); }
        store.buildDim0Index();
        // Verify that sortedDim0[i] == vectors[sortedByDim0[i]][0]
        for (int i = 0; i < store.size(); i++) {
            assertEq("mapping at " + i,
                    store.sortedDim0[i],
                    store.vectors[store.sortedByDim0[i]][0]);
        }
    }

    // ── fraudScore() ──────────────────────────────────────────────────────────

    private static void checkScoreZero() {
        byte[] labels = {VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT};
        assertEq("score=0.0", 0.0f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreFive() {
        byte[] labels = {VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD};
        assertEq("score=1.0", 1.0f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreOne() {
        byte[] labels = {VectorStore.FRAUD, VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT};
        assertEq("score=0.2", 0.2f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreTwo() {
        byte[] labels = {VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.LEGIT, VectorStore.LEGIT, VectorStore.LEGIT};
        assertEq("score=0.4", 0.4f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreThree() {
        byte[] labels = {VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.LEGIT, VectorStore.LEGIT};
        assertEq("score=0.6", 0.6f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreFour() {
        byte[] labels = {VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.FRAUD, VectorStore.LEGIT};
        assertEq("score=0.8", 0.8f, KnnSearch.fraudScore(labels, new int[]{0,1,2,3,4}));
    }

    private static void checkScoreUnfilled() {
        // Only slot 0 is filled; that vector is FRAUD → score = 1/5 = 0.2
        byte[] labels = {VectorStore.FRAUD};
        assertEq("score with unfilled", 0.2f, KnnSearch.fraudScore(labels, new int[]{0,-1,-1,-1,-1}));
    }

    /** fraudScore(labels, idx) must equal fraudCount(store, idx) / K for every N. */
    private static void checkScoreConsistency() {
        VectorStore store = new VectorStore(5);
        store.add(v(0), VectorStore.FRAUD);
        store.add(v(1), VectorStore.LEGIT);
        store.add(v(2), VectorStore.FRAUD);
        store.add(v(3), VectorStore.LEGIT);
        store.add(v(4), VectorStore.FRAUD);
        int[] idx = {0,1,2,3,4};
        float expected = KnnSearch.fraudCount(store, idx) / (float) KnnSearch.K;
        float actual   = KnnSearch.fraudScore(store.labels, idx);
        assertEq("fraudScore == fraudCount/K", expected, actual);
    }

    // ── fraudCount() — int version used by Main.RESPONSES[] lookup ───────────

    private static void checkFraudCountZero() {
        VectorStore store = new VectorStore(5);
        for (int i = 0; i < 5; i++) store.add(v(i), VectorStore.LEGIT);
        assertEq("fraudCount", 0, KnnSearch.fraudCount(store, new int[]{0,1,2,3,4}));
    }

    private static void checkFraudCountFive() {
        VectorStore store = new VectorStore(5);
        for (int i = 0; i < 5; i++) store.add(v(i), VectorStore.FRAUD);
        assertEq("fraudCount", 5, KnnSearch.fraudCount(store, new int[]{0,1,2,3,4}));
    }

    private static void checkFraudCountMixed() {
        VectorStore store = new VectorStore(5);
        store.add(v(0), VectorStore.FRAUD);
        store.add(v(1), VectorStore.LEGIT);
        store.add(v(2), VectorStore.FRAUD);
        store.add(v(3), VectorStore.LEGIT);
        store.add(v(4), VectorStore.LEGIT);
        assertEq("fraudCount", 2, KnnSearch.fraudCount(store, new int[]{0,1,2,3,4}));
    }

    private static void checkFraudCountUnfilled() {
        VectorStore store = new VectorStore(1);
        store.add(v(0), VectorStore.FRAUD);
        assertEq("fraudCount with -1 slots", 1,
                KnnSearch.fraudCount(store, new int[]{0,-1,-1,-1,-1}));
    }

    // ── spec examples ─────────────────────────────────────────────────────────

    private static void checkSpecLegit() {
        float[] legit = {0.0041f,0.1667f,0.05f,0.7826f,0.3333f,-1f,-1f,0.0292f,0.15f,0f,1f,0f,0.15f,0.006f};
        float[] fraud = {0.9506f,0.8333f,1.0f,0.2174f,0.8333f,-1f,-1f,0.9523f,1.0f,0f,1f,1f,0.75f,0.0055f};
        VectorStore store = new VectorStore(10);
        for (int i = 0; i < 5; i++) { float[] v = legit.clone(); v[0] += i*0.001f; store.add(v, VectorStore.LEGIT); }
        for (int i = 0; i < 5; i++) { float[] v = fraud.clone(); v[0] += i*0.001f; store.add(v, VectorStore.FRAUD); }
        assertEq("legit query → fraudCount=0", 0, KnnSearch.fraudCount(store, runSearch(store, legit)));
    }

    private static void checkSpecFraud() {
        float[] legit = {0.0041f,0.1667f,0.05f,0.7826f,0.3333f,-1f,-1f,0.0292f,0.15f,0f,1f,0f,0.15f,0.006f};
        float[] fraud = {0.9506f,0.8333f,1.0f,0.2174f,0.8333f,-1f,-1f,0.9523f,1.0f,0f,1f,1f,0.75f,0.0055f};
        VectorStore store = new VectorStore(10);
        for (int i = 0; i < 5; i++) { float[] v = legit.clone(); v[0] += i*0.001f; store.add(v, VectorStore.LEGIT); }
        for (int i = 0; i < 5; i++) { float[] v = fraud.clone(); v[0] += i*0.001f; store.add(v, VectorStore.FRAUD); }
        assertEq("fraud query → fraudCount=5", 5, KnnSearch.fraudCount(store, runSearch(store, fraud)));
    }

    private static void checkThreshold() {
        VectorStore store = new VectorStore(5);
        for (int i = 0; i < 5; i++) store.add(v(i), VectorStore.LEGIT);
        store.labels[0] = VectorStore.FRAUD;
        store.labels[1] = VectorStore.FRAUD;
        store.labels[2] = VectorStore.FRAUD;
        int[] idx = {0,1,2,3,4};
        float score = KnnSearch.fraudCount(store, idx) / (float) KnnSearch.K;
        assertEq("3 fraud → score=0.6", 0.6f, score);
        assertTrue("0.6 is NOT approved", !(score < 0.6f));
        store.labels[2] = VectorStore.LEGIT;
        score = KnnSearch.fraudCount(store, idx) / (float) KnnSearch.K;
        assertEq("2 fraud → score=0.4", 0.4f, score);
        assertTrue("0.4 IS approved", score < 0.6f);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static VectorStore storeOf(Object... args) {
        int n = args.length / 2;
        VectorStore store = new VectorStore(n);
        for (int i = 0; i < args.length; i += 2)
            store.add((float[]) args[i], ((Number) args[i + 1]).byteValue());
        return store;
    }

    private static float[] v(int seed) {
        float[] a = new float[VectorStore.DIMS];
        a[0] = seed * 0.1f;
        return a;
    }

    private static int[] runSearch(VectorStore store, float[] query) {
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.search(store, query, topDist, topIdx);
        return topIdx;
    }

    private static int[] runIndexed(VectorStore store, float[] query) {
        float[] topDist = new float[KnnSearch.K]; int[] topIdx = new int[KnnSearch.K];
        KnnSearch.searchIndexed(store, query, topDist, topIdx);
        return topIdx;
    }

    // ── Test runner ───────────────────────────────────────────────────────────

    private static void run(String name, Runnable body) {
        try {
            body.run();
            System.out.printf("  [PASS] %s%n", name);
            PASS++;
        } catch (AssertionError e) {
            System.out.printf("  [FAIL] %s — %s%n", name, e.getMessage());
            FAIL++;
        }
    }

    private static void assertEq(String label, float expected, float actual) {
        if (Math.abs(expected - actual) > 1e-5f)
            throw new AssertionError(label + ": expected=" + expected + " actual=" + actual);
    }

    private static void assertEq(String label, int expected, int actual) {
        if (expected != actual)
            throw new AssertionError(label + ": expected=" + expected + " actual=" + actual);
    }

    private static void assertTrue(String msg, boolean cond) {
        if (!cond) throw new AssertionError(msg);
    }
}

