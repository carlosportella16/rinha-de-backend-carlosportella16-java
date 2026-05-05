# Rinha de Backend 2026 — Fraud Detection via Vector Search

> **Carlos Portella** · Java 25 · Netty · IVF + INT8 · Off-Heap

A high-performance fraud detection API built for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).  
The goal: serve KNN-based fraud scores with **p99 ≤ 1 ms** under 1 CPU / 350 MB RAM constraints.

---

## Benchmark Results

| Scenario | p99 | score_p99 | score_det | final_score |
|----------|-----|-----------|-----------|-------------|
| Baseline (1D index, float32) | 15.2 ms | 1818 | 3000 | **4818** |
| + JVM flags + Epoll + warmup | ~8 ms | ~2097 | 3000 | **~5097** |
| + INT8 quantization + IVF V4 | ~2 ms | ~2699 | 3000 | **~5699** |

---

## Architecture

```
Client
  └─ nginx (round-robin load balancer, port 9999)
        ├─ api1 (port 9998) — JVM, Netty, 0.475 CPU, 167 MB
        └─ api2 (port 9998) — JVM, Netty, 0.475 CPU, 167 MB
```

Each API instance runs the full pipeline independently — no shared state, no database.

### Request Pipeline

```
POST /fraud-score
    └─ Netty Epoll EventLoop (Linux) / NIO (macOS)
        └─ HttpObjectAggregator (64 KB)
            └─ RequestParser          → float[14]  (zero-alloc, raw ByteBuf scan)
                └─ KnnSearch.quantize → byte[14]   (INT8, ThreadLocal)
                    └─ searchIVF      → top-5 idx  (IVF V4, distSqInt8 + early-exit)
                        └─ fraudCount → RESPONSES[n].duplicate()
                            └─ writeAndFlush
```

---

## Stack

| Component | Technology |
|-----------|------------|
| Runtime | Java 25 (eclipse-temurin:25-jre) |
| HTTP Server | Netty 4.2 (no Spring, no Servlet) |
| Vector Index | Custom IVF (K-Means C=512, offline) |
| Vector Storage | Off-heap `ByteBuffer.allocateDirect` (INT8) |
| Load Balancer | nginx 1.27-alpine |
| Build | Maven 3.9 + maven-shade-plugin |
| SIMD | JDK Vector API (`jdk.incubator.vector`, JEP 508) |

---

## Optimizations

### 1. IVF Index — Inverted File Index (biggest impact)

Instead of scanning all 3M vectors per query, a **K-Means index** partitions the dataset into **C = 512 clusters** at build time. At query time, only the `nprobe = 32` nearest clusters are scanned.

```
Brute force:  3,000,000 distSq calls/query  →  ~15 ms worst case
IVF (C=512):  ~185,000 distSq calls/query   →  ~2 ms
```

The index is built **once** during `docker build` by `ReferenceConverter`, which:
1. Parses `references.json.gz` (3M JSON vectors → flat `float[]`)
2. Runs parallel k-means (15 iterations, `IntStream.parallel()`)
3. Writes a **V4 cluster-ordered binary** (`references.bin`, ~46 MB)

**V4 format** stores vectors physically in cluster order — no indirection array needed. Phase 2 scanning is a pure sequential memory read, maximizing CPU cache efficiency.

### 2. INT8 Quantization — 42 MB instead of 168 MB

All 3M reference vectors are quantized from `float32` to `int8` at build time:

```
float f → byte = round(f × 127), clamped to [-127, 127]
```

| Format | Memory per instance | Notes |
|--------|--------------------|----|
| float32 | 168 MB | 3M × 14 × 4 bytes |
| INT8 | **42 MB** | 3M × 14 × 1 byte |

This frees ~126 MB per instance, making two instances fit within the 350 MB total limit.  
Distance is computed with integer arithmetic — faster than float for this range.

### 3. Early Termination in `distSqInt8`

The inner distance loop short-circuits as soon as the partial sum exceeds the current worst distance (threshold):

```java
int distSqInt8(byte[] query, int idx, int threshold) {
    long base = vectorsAddr + (long)idx * 14;
    int sum = 0, d;
    d = query[0] - UNSAFE.getByte(base);    sum += d*d; if (sum >= threshold) return sum;
    d = query[1] - UNSAFE.getByte(base+1);  sum += d*d; if (sum >= threshold) return sum;
    // ... all 14 dims
}
```

As the top-5 heap fills up, the threshold tightens and more vectors are rejected after only 2–3 dimensions. This saves 30–50% of the total work in a typical query.

### 4. Epoll Transport on Linux

In production (Linux containers), Netty uses `EpollIoHandler` instead of the Java NIO selector:

```java
final boolean useEpoll = Epoll.isAvailable();  // true on Linux
EventLoopGroup boss   = new MultiThreadIoEventLoopGroup(1, useEpoll ? EpollIoHandler.newFactory() : NioIoHandler.newFactory());
```

Epoll reduces syscall overhead by ~15% compared to NIO at high request rates.

### 5. Zero-Allocation Hot Path

Every object that would be allocated per-request is instead stored in `ThreadLocal` buffers, allocated once per I/O thread:

```java
// Per-thread scratch buffers — zero heap allocation after warmup
static final ThreadLocal<float[]> TL_VEC      = ThreadLocal.withInitial(() -> new float[14]);
static final ThreadLocal<byte[]>  TL_VEC_INT8 = ThreadLocal.withInitial(() -> new byte[14]);
static final ThreadLocal<float[]> TL_DIST     = ThreadLocal.withInitial(() -> new float[5]);
static final ThreadLocal<int[]>   TL_IDX      = ThreadLocal.withInitial(() -> new int[5]);
```

The only unavoidable per-request allocations are HTTP framing objects from Netty (`DefaultFullHttpResponse`).

### 6. Pre-encoded Responses

All 6 possible fraud scores (`0/5` to `5/5`) are encoded as static `DirectByteBuf` buffers at startup. The response path does a single array lookup and `ByteBuf.duplicate()` — no JSON building at query time:

```java
private static final ByteBuf[] RESPONSES = {
    staticBuf("{\"approved\":true,\"fraud_score\":0.0}"),   // 0 frauds
    staticBuf("{\"approved\":true,\"fraud_score\":0.2}"),   // 1 fraud
    staticBuf("{\"approved\":true,\"fraud_score\":0.4}"),   // 2 frauds
    staticBuf("{\"approved\":false,\"fraud_score\":0.6}"),  // 3 frauds
    staticBuf("{\"approved\":false,\"fraud_score\":0.8}"),  // 4 frauds
    staticBuf("{\"approved\":false,\"fraud_score\":1.0}"),  // 5 frauds
};
```

### 7. Zero-Copy JSON Parsing

`RequestParser` operates directly on the raw Netty `ByteBuf` — no `String` creation, no Jackson, no reflection:

- Scans bytes with hand-written pattern matchers
- Parses ISO-8601 timestamps to epoch-minutes via pure integer arithmetic (Tomohiko Sakamoto's algorithm)
- `merchantKnown()` comparison is byte-by-byte against the raw buffer — no string allocation

### 8. JIT Warmup (10,000 queries)

HotSpot's C2 compiler reaches peak optimization after ~10,000 invocations of a method. The server runs 10,000 synthetic IVF searches before setting `READY = true`, using an XorShift64 PRNG (no allocation):

```java
for (int i = 0; i < 10_000; i++) {
    // XorShift64 RNG — fills vec[] with pseudo-random values
    KnnSearch.searchIVF(store, vec, vecInt8, topDist, topIdx, nprobe);
}
```

### 9. Aggressive JVM Flags

```
-XX:+UseSerialGC              # single-threaded GC — no GC threads competing for CPU
-Xms64m -Xmx64m              # pre-allocate full heap — no GC expansion overhead
-XX:+AlwaysPreTouch           # touch all heap pages at startup — no page faults at runtime
-XX:+TieredCompilation        # enable C1 → C2 tiered compilation
-XX:CompileThreshold=1000     # reach C2 faster (default is 10,000)
-XX:Tier4CompileThreshold=500 # C2 compilation even sooner for hot methods
-XX:+DoEscapeAnalysis         # stack-allocate short-lived objects
-XX:+EliminateAllocations     # eliminate scalar replaceable allocations
-XX:+DisableExplicitGC        # ignore System.gc() calls
```

### 10. SIMD Distance (float32 on heap)

`SimdDistance` uses the JDK Vector API (`jdk.incubator.vector`) to compute float32 squared Euclidean distance with AVX2/AVX-512 instructions:

```java
// For 14-dim vectors on AVX2 (8 lanes):
//   1 full SIMD pass for dims [0-7]  →  1 VMOVUPS + 1 VSUBPS + 1 VFMADD
//   6 scalar iterations for dims [8-13]
var acc = FloatVector.zero(SPECIES);
for (int i = 0; i <= BOUND; i += LANES) {
    var d = FloatVector.fromArray(SPECIES, a, i).sub(FloatVector.fromArray(SPECIES, b, i));
    acc = d.fma(d, acc);
}
```

Used for centroid distance computation in IVF Phase 1 (float32 centroids, 512 × 14 = 7,168 ops per query). The main Phase 2 distance uses INT8 off-heap (`distSqInt8`).

### 11. PooledByteBufAllocator

```java
.childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
```

Reuses ByteBuf memory across requests, avoiding repeated `malloc`/`free` for response buffers.

---

## Memory Layout

```
Per instance (167 MB limit):
  Off-heap vectors:   42 MB   (3M × 14 INT8 bytes, ByteBuffer.allocateDirect)
  Off-heap labels:     3 MB   (3M × 1 byte)
  On-heap centroids:  28 KB   (512 × 14 × 4 bytes, float32)
  JVM heap (-Xmx64m): 64 MB
  JVM overhead:      ~40 MB   (JIT code cache, class metadata, threads)
  ─────────────────────────
  Total:            ~149 MB   ✓ within 167 MB limit
```

---

## Binary Format (V4)

`ReferenceConverter` generates a compact binary at build time. The loader (`VectorLoader`) reads it directly into off-heap memory at startup — no JSON parsing at runtime.

```
Header (32 bytes, little-endian):
  [0..3]   magic    = 0x52494E48  ("RINH")
  [4..7]   version  = 4
  [8..11]  count    = N (3,000,000)
  [12..15] dims     = 14
  [16..19] clusters = 512
  [20..23] nprobe   = 32
  [24..31] reserved = 0

Data:
  INT8 vectors:  N × 14 bytes  (cluster-ordered)
  Labels:        N × 1 byte    (cluster-ordered)
  Centroids:     C × 14 × 4 bytes (float32 LE)
  Cluster sizes: C × 4 bytes   (int32 LE)
```

Total file size: ~46 MB (vs ~284 MB for the raw `references.json.gz` uncompressed).

---

## Resource Distribution

```yaml
nginx:  0.05 CPU  /  16 MB   # just a proxy — minimal footprint
api1:   0.475 CPU / 167 MB
api2:   0.475 CPU / 167 MB
────────────────────────────
Total:  1.00 CPU  / 350 MB   ✓
```

---

## Running Locally

```bash
# 1. Place the dataset (download from the official contest repo)
cp /path/to/references.json.gz resources/

# 2. Build the Docker image (runs k-means C=512 during build — takes ~5 min)
docker buildx build \
  --platform linux/amd64 \
  -t carlosportella16/rinha-2026:latest \
  --push \
  .

# 3. Start the stack
docker compose up

# 4. Test
curl -s http://localhost:9999/ready
curl -s -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d @resources/example-payloads.json | head -c 200
```

---

## Project Structure

```
src/main/java/org/rinha/
├── Main.java              # Netty server bootstrap, /ready + /fraud-score handlers
├── RequestParser.java     # Zero-alloc JSON parser (raw ByteBuf scan)
├── KnnSearch.java         # IVF search (searchIVF) + brute-force fallback (search)
├── OffHeapVectorStore.java# Off-heap INT8 vector store + distSqInt8 with early-exit
├── VectorLoader.java      # Binary loader for V2/V3/V4 formats
├── ReferenceConverter.java# Build-time: JSON.gz → V4 binary (k-means + quantization)
├── SimdDistance.java      # JDK Vector API float32 distance (AVX2/AVX-512)
├── VectorStore.java       # On-heap float32 store (used in tests)
└── DistanceBenchmark.java # Micro-benchmark: SIMD vs scalar
```

---

## License

MIT

