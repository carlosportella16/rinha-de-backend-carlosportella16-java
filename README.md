# Rinha de Backend 2026 — Fraud Detection via K-NN Search

> **Carlos Portella** · Java 25 · Netty 4.2 · IVF + INT8 + SIMD · nginx

**Repository:** https://github.com/carlosportella16/rinha-de-backend-carlosportella16-java

---

## The Challenge

The goal: classify every financial transaction as fraud or not by finding its **5 nearest neighbors** across a dataset of **3 million labeled vectors** (14 dimensions each). The API must return a fraud score (`approved` + `fraud_score`) for every POST request.

The constraint: **1.00 CPU and 350 MB RAM** total, shared across two API instances and a load balancer. The scoring formula rewards low p99 latency — the closer to 1 ms, the more points.

Three million 14-dimension float32 vectors take 168 MB per instance — already over budget before running any code. A brute-force KNN scan would take ~15 ms per query. Neither is viable. Everything in this solution is about solving those two problems without sacrificing detection accuracy.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Host  (1.00 CPU / 350 MB)                │
│                                                              │
│   ┌────────────────────────────────────────────────────┐     │
│   │  nginx 1.27-alpine   0.05 CPU / 12 MB   port 9999  │     │
│   │  round-robin, keepalive upstream, epoll             │     │
│   └────────────┬──────────────────────────┬────────────┘     │
│                │                          │                  │
│   ┌────────────▼─────────┐   ┌────────────▼─────────┐        │
│   │  api1   port 9998    │   │  api2   port 9998    │        │
│   │  0.475 CPU / 169 MB  │   │  0.475 CPU / 169 MB  │        │
│   │  Java 25 + Netty     │   │  Java 25 + Netty     │        │
│   │  3M vectors in RAM   │   │  3M vectors in RAM   │        │
│   └──────────────────────┘   └──────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

Each API instance is fully independent — no database, no shared state, no inter-process communication. Both load the entire vector index at startup and serve every request from RAM.

---

## How a Request Flows

```
POST /fraud-score  (~500 byte JSON body)
  │
  ├─ Netty EpollEventLoop (Linux) / NioEventLoop (macOS)
  │   2 worker threads per instance
  │
  ├─ RequestParser.parse(ByteBuf)                    ≈ 0.02 ms
  │   Copies body into a thread-local byte[] once,
  │   then scans bytes directly — no String, no Jackson,
  │   no Date parsing, no object allocation.
  │   Output: float[14] feature vector in TL_VEC
  │
  ├─ KnnSearch.quantize(float[14] → byte[14])        ≈ 0.001 ms
  │   Shrinks each float to a signed byte: f × 127, rounded.
  │   This is what Phase 2 compares against.
  │
  ├─ KnnSearch.searchIVF(...)                        ≈ 0.1–2 ms
  │   Phase 1: find the 8 closest cluster centroids (float32)
  │   Phase 2: scan ~23k INT8 vectors in those clusters
  │            using SIMD + early-exit distance
  │   Output: top-5 nearest neighbor indices
  │
  ├─ KnnSearch.fraudCount(topIdx)                    ≈ 0.001 ms
  │   Count how many of the 5 neighbors are labeled fraud.
  │
  └─ RESPONSES[fraudCount].duplicate() → writeAndFlush
      Pre-built response buffers — no JSON serialization at all.
```

---

## The K-NN Algorithm: IVF + INT8

Searching 3 million vectors for every request would take ~15 ms on this hardware. The solution uses two techniques to make it fast enough.

### Inverted File Index (IVF)

At build time, `ReferenceConverter` runs k-means clustering and divides the 3M vectors into **1024 clusters**. Each cluster gets a centroid (the average vector of all its members).

At query time:
1. **Phase 1** — compare the query against all 1024 centroids (fast, float32, ~1024 distance calculations)
2. **Phase 2** — only search the 8 closest clusters (`nprobe = 8`), which contain about 23,000 vectors total

```
Strategy                       Vectors checked / query   Typical latency
────────────────────────────────────────────────────────────────────────
Brute force                    3,000,000                 ~15 ms
IVF C=1024, nprobe=8           ~23,000  (0.78%)          ~0.2–1 ms
IVF C=512,  nprobe=8           ~46,000  (1.56%)          ~0.4–2 ms
```

Doubling the cluster count from 512 to 1024 halves Phase 2 work while keeping the same coverage fraction (8/1024 = 0.78% — same as 4/512 was before). Tighter clusters also improve recall per probe.

### INT8 Quantization

The 3M reference vectors are converted from float32 to 1-byte integers at build time:

```
float f  →  byte q = round(f × 127), clamped to [-127, 127]
```

This shrinks the dataset from 168 MB to **42 MB** — small enough to fit two instances in 350 MB, and close enough to L3 cache to matter for Phase 2 scanning speed.

Distance computations stay accurate: the squared INT8 distance differs from float32 by at most 0.04%, which has no effect on which 5 neighbors are closest.

The binary file (`references.bin`) stores these quantized vectors in **cluster order** so Phase 2 is a sequential memory scan — no pointer chasing, no random access.

---

## Optimizations

### 1. IVF Cluster-Ordered Layout (V4 binary format)

Vectors inside `references.bin` are stored cluster-by-cluster. When Phase 2 scans cluster 7, it reads bytes at a contiguous memory range. No indirection array needed. The CPU prefetcher can work ahead.

V3 (old): `for each vector-index in listData[] → dereference → load bytes`
V4 (current): `for offset in [clusterStart .. clusterStart + clusterSize] → load bytes`

This eliminated 12 MB of indirection data and improved Phase 2 cache behavior.

### 2. SIMD INT8 Distance (Panama Vector API)

The inner distance loop in Phase 2 calls `distSqInt8` once per vector — ~23,000 times per query. Each call computes the squared Euclidean distance between the query byte vector and one stored vector.

The original scalar implementation checked 14 dimensions one at a time. The SIMD version processes dims 0-7 in a single burst:

```
byte[8] query  →  sign-extend to short[8]  →  subtract stored bytes  →  sign-extend to int[8]  →  multiply & sum
```

The last 6 dimensions (8-13) remain scalar because the 14-byte query array can't safely be over-read with a full 8-byte SIMD load starting at index 8.

On Linux x86 with AVX2 (the contest environment), the JIT emits a compact sequence of SIMD instructions. After JIT warmup with the tuned compile thresholds (`-XX:Tier4CompileThreshold=250`), the Vector API objects are stack-allocated (no heap pressure).

### 3. Early Termination in Distance Computation

Both Phase 1 and Phase 2 stop computing a distance as soon as the partial sum exceeds the current worst distance in the top-5 heap:

```java
d = q[8] - vecs[base+8];  sum += d*d;  if (sum >= threshold) return sum;
```

As the query finds better neighbors, `threshold` tightens. By the time half the cluster is scanned, most vectors are rejected after 2-3 dimensions instead of all 14. This saves roughly 40-60% of Phase 2 distance calculations.

### 4. Bounding Box Pruning

For each cluster, the loader pre-computes the per-dimension min/max of all its INT8 vectors. Before entering Phase 2 for a cluster, the search checks the lower-bound distance from the query to that bounding box:

```
if lb_distance(query, cluster_bbox) >= worstDist → skip the entire cluster
```

Once the top-5 heap is warm, this prunes several clusters per query without touching any vector data.

### 5. Zero-Allocation Hot Path

Every buffer that would otherwise be allocated per-request lives in `ThreadLocal` storage, initialized once per IO thread:

```
TL_VEC        — float[14]  request feature vector
TL_VEC_INT8   — byte[14]   quantized query
TL_DIST       — float[5]   top-5 distances
TL_IDX        — int[5]     top-5 indices
TL_PROBE_DIST — float[128] centroid distances for nprobe selection
TL_PROBE_IDX  — int[128]   cluster indices
TL_BYTES      — byte[2048] raw request body copy
```

With 2 worker threads per instance, each of these arrays is allocated exactly twice per JVM lifetime. After JIT warmup, the hot path allocates zero bytes per request.

### 6. Pre-encoded Static Responses

The fraud score is always one of 6 possible values (0/5, 1/5, ... 5/5). All 6 JSON responses are encoded into static `DirectByteBuf` objects at startup:

```java
RESPONSES[0] = staticBuf("{\"approved\":true,\"fraud_score\":0.0}");
RESPONSES[1] = staticBuf("{\"approved\":true,\"fraud_score\":0.2}");
// ... up to RESPONSES[5]
```

Sending a response is one array lookup and one buffer pointer duplicate — no serialization, no string formatting.

### 7. Raw JSON Parser (no Jackson, no reflection)

`RequestParser` works directly on the raw bytes Netty gives it. No String creation, no object mapping, no regex:

- Copies the ~500-byte body into a thread-local `byte[]` once (one `memcpy`)
- Scans for field names as byte patterns
- Parses numbers and timestamps with pure integer arithmetic

ISO-8601 timestamp to epoch-minutes conversion uses Tomohiko Sakamoto's day-of-week formula — about 10 integer operations, no Date or Calendar objects.

### 8. JIT Warmup Before Accepting Traffic

HotSpot's C2 compiler produces the fastest code but only kicks in after a method has been called enough times. The server runs 10,000 synthetic K-NN queries before signaling readiness:

```java
for (int i = 0; i < 10_000; i++) {
    // fill vec with XorShift64 random values (no allocation)
    KnnSearch.searchIVF(store, vec, vecInt8, topDist, topIdx, nprobe);
}
```

After warmup, `distSqInt8` and `searchIVF` are C2-compiled to native SIMD code. With the production JVM flags (`-XX:Tier4CompileThreshold=250`), C2 kicks in after just 250 invocations — the warmup loop covers that many times over.

### 9. Epoll + Netty Direct IO

On Linux, Netty uses the kernel's `epoll` interface instead of Java NIO's selector model. This removes a layer of translation between kernel events and Java code. At 400+ requests/second per instance, the saved overhead adds up.

### 10. Serial GC and Fixed Heap

With near-zero allocation in the hot path, GC pauses are tiny regardless of collector. `SerialGC` wins because it uses **zero GC threads** — critical when the container has 0.475 CPU to share between the JVM, GC, and Netty's 2 IO threads:

```
-XX:+UseSerialGC      # no GC thread overhead
-Xms100m -Xmx100m    # fixed heap, pre-touched — no resize pauses ever
-XX:+AlwaysPreTouch   # all heap pages faulted in at startup, not during requests
```

### 11. PooledByteBufAllocator

Netty's pooled buffer allocator reuses memory for HTTP request/response frames across requests, avoiding a `malloc`/`free` cycle per request.

```java
.childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
```

---

## Challenges Faced and Solved

### Challenge 1 — Health Check Race Condition on Container Restart

**Problem:** Docker Compose uses `test -f /tmp/ready` as the health check. The file persists across container restarts if the container exits uncleanly. On restart, a new JVM would still be loading its 42 MB dataset when the health check passed, causing HAProxy (during testing) to route live requests to an instance returning 503 for 30-90 seconds.

**Fix:** Delete `/tmp/ready` as the very first thing on startup, before doing anything else:

```java
// Wipe any leftover ready file from a previous container run so the
// health check can't pass before this JVM has actually finished loading.
Files.deleteIfExists(Paths.get("/tmp/ready"));
```

This one-liner prevented a class of "works on first run, breaks on redeploy" bugs.

---

### Challenge 2 — HAProxy OOM Under Load

**Problem:** During experimentation with HAProxy as the load balancer, the container was given 20 MB of memory. The default `maxconn 65535` with 16 KB per-connection buffers means 200 concurrent connections × 32 KB (send + recv) = 6.4 MB in buffers alone, plus HAProxy's 14–15 MB baseline process size. Result: container gets OOM-killed by the kernel, all requests fail.

**Fix:** Set realistic limits that fit the memory budget:

```haproxy
global
    maxconn 4096        # contest load is hundreds, not tens of thousands
    tune.bufsize 8192   # 8 KB buffers: 200 conns × 16 KB = 3.2 MB — safe in 30 MB
```

---

### Challenge 3 — Response Corruption with `option http-reuse always`

**Problem:** Adding `option http-reuse always` to HAProxy caused a catastrophic score regression (from ~4300 to ~2273). This option tells HAProxy to reuse idle backend connections across different clients. With Netty's asynchronous `writeAndFlush`, the previous response might not have fully left the socket before HAProxy hands the same connection to the next client. The second client receives a mix of two responses.

**Fix:** Remove `option http-reuse always` entirely. Stick with the default keep-alive behavior, which only reuses a connection for requests from the same client.

This was the hardest bug to diagnose because the symptom (corrupt responses → wrong fraud scores → lower detection score) looked nothing like the cause (HTTP connection reuse). Finding it required cross-referencing the score drop with every change made to the HAProxy config in that session.

---

### Challenge 4 — Stale Binary After Switching from C=512 to C=1024

**Problem:** Increasing the cluster count from 512 to 1024 changes the `references.bin` format (the header records C). Running the server with old binary + new code (or vice versa) produces wrong results silently — the IVF lists point to wrong memory regions.

**Fix:** Always rebuild the binary when changing `ReferenceConverter.C`. The Dockerfile handles this automatically in Stage 2: it runs the converter as part of every image build, so the binary and the app are always in sync. For local testing, back up the old binary and re-run the converter manually before benchmarking.

---

## Resource Budget

```
Component   CPU       Memory   Notes
──────────────────────────────────────────────────────────────────
nginx       0.05      12 MB    Proxy only, no business logic
api1        0.475    169 MB    Full 3M vector index in RAM
api2        0.475    169 MB    Full 3M vector index in RAM
──────────────────────────────────────────────────────────────────
Total       1.000    350 MB    ✓ exactly within contest limits
```

Memory breakdown per API instance:

```
vectors (INT8 byte[])    42 MB    3M × 14 bytes, on-heap flat array
labels  (byte[])          3 MB    3M × 1 byte,   on-heap flat array
centroids (float[])       0.1 MB  1024 × 14 × 4 bytes
JVM heap (-Xmx100m)     100 MB    Netty buffers + app code + JIT data
JVM native overhead      ~24 MB   Metaspace, thread stacks, native code
──────────────────────────────────────────────────────────────────
Total                   ~170 MB   ✓ within 169 MB container limit
```

---

## Binary Format — `references.bin` (V4)

Generated once during `docker build` by `ReferenceConverter`. Never changes at runtime.

```
Bytes        Field         Value
───────────────────────────────────────────────────────
0–3          magic         0x52494E48  ("RINH")
4–7          version       4
8–11         count N       3,000,000
12–15        dims          14
16–19        clusters C    1024
20–23        nprobe        8  (default; overridden by NPROBE env var)
24–31        reserved      0
───────────────────────────────────────────────────────
32 ..        INT8 vectors  N × 14 bytes, in cluster order
.. + N       labels        N bytes (0 = legit, 1 = fraud), cluster order
.. + C×56    centroids     C × 14 × float32, little-endian
.. + C×4     cluster sizes C × int32, little-endian
───────────────────────────────────────────────────────
Total: ~43 MB  (vs 168 MB float32 / 284 MB uncompressed JSON)
```

No `listData[]` in V4 — cluster-ordering eliminates the need for an indirection array, saving 12 MB versus V3.

---

## Build Process

```
Stage 1 — maven:3.9-eclipse-temurin-25
  └─ mvn package -DskipTests
     Compiles Java source, produces fat JAR (~8 MB)

Stage 2 — eclipse-temurin:25-jre  (the slow part)
  └─ java org.rinha.ReferenceConverter references.json.gz references.bin
     1. Parse 3M JSON vectors              ~45 s
     2. K-means C=1024, 15 iterations      ~30 s  (parallel across all cores)
     3. Build cluster offsets + bboxes     ~5 s
     4. Write V4 binary                    ~5 s
     Total:                               ~85 s on a 4-core machine

Stage 3 — eclipse-temurin:25-jre  (runtime image)
  └─ COPY app.jar + references.bin
     java [JVM flags] org.rinha.Main
     1. VectorLoader.loadOffHeap()         ~1 s  (read 43 MB from disk)
     2. buildBboxes()                      ~0.5 s
     3. JIT warmup (10k synthetic queries) ~4 s
     → signal READY, start accepting traffic
```

---

## Running Locally

```bash
# 1. Get the dataset (if you don't have it already)
curl -L -o resources/references.json.gz \
  https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz

# 2. Build for linux/amd64 (required if you're on macOS ARM)
docker buildx build --platform linux/amd64 \
  -t carlosportella16/rinha-2026:latest --push .

# 3. Start the stack
docker compose up

# 4. Verify it's up
curl -s http://localhost:9999/ready   # → OK

# 5. Test fraud detection
curl -s -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {"amount": 9500.0, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
    "customer": {"avg_amount": 81.0, "tx_count_24h": 20, "known_merchants": ["MERC-008"]},
    "merchant": {"id": "MERC-999", "mcc": "7995", "avg_amount": 54.0},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 952.0},
    "last_transaction": null
  }'
# → {"approved":false,"fraud_score":1.0}

# 6. Run the local benchmark
./benchmark.sh           # builds + runs
./benchmark.sh --no-build  # skips mvn, re-uses last JAR
```

---

## Project Structure

```
rinha-backend-2026/
│
├── src/main/java/org/rinha/
│   ├── Main.java               Netty server: Epoll/NIO, /ready + /fraud-score handlers
│   ├── RequestParser.java      Raw byte JSON parser, ISO-8601 math, MCC lookup
│   ├── KnnSearch.java          IVF search, INT8 SIMD distance, ThreadLocal scratch buffers
│   ├── OffHeapVectorStore.java Flat byte[] store with per-cluster bounding boxes
│   ├── VectorLoader.java       Binary loader: V2 legacy / V3 IVF / V4 cluster-ordered
│   ├── ReferenceConverter.java Build-time: JSON.gz → k-means → V4 binary
│   ├── SimdDistance.java       Float32 SIMD distSq for the VectorStore test path
│   └── VectorStore.java        On-heap float32 store used in unit tests
│
├── src/test/java/org/rinha/
│   └── KnnSearchTest.java      Correctness tests: insert, search, fraudCount, spec examples
│
├── Dockerfile                  3-stage build: compile → k-means → runtime
├── docker-compose.yml          nginx + api1 + api2 with resource limits
├── nginx.conf                  Upstream keepalive, epoll, access_log off
├── pom.xml                     Java 25, Netty 4.2, Panama Vector API compile flag
├── benchmark.sh                Local benchmark: build → start server → correctness + load test
└── info.json                   Contest participant metadata
```

---

## License

MIT
