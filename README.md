# Rinha de Backend 2026 — Fraud Detection via Vector Search

> **Carlos Portella** · Java 25 · Netty 4.2 · IVF + INT8 · Off-Heap · Epoll

A **high-performance fraud detection API** built for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).  
**Repository:** https://github.com/carlosportella16/rinha-de-backend-carlosportella16-java

The challenge: serve KNN-based fraud scores over 3 million vectors with **p99 ≤ 1 ms**, using only **1 CPU and 350 MB RAM** across two API instances + one load balancer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host (1 CPU / 350 MB)                    │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  nginx 1.27-alpine  (0.05 CPU / 16 MB)   :9999          │  │
│   │  round-robin upstream, keepalive 64, epoll, tcp_nodelay  │  │
│   └────────────────────┬─────────────────────┬───────────────┘  │
│                        │                     │                  │
│          ┌─────────────▼──────┐   ┌──────────▼─────────┐       │
│          │  api1  :9998       │   │  api2  :9998        │       │
│          │  0.475 CPU / 167MB │   │  0.475 CPU / 167MB  │       │
│          │  Java 25 + Netty   │   │  Java 25 + Netty    │       │
│          └────────────────────┘   └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

Each API instance is **fully independent** — no database, no shared cache, no inter-process communication. The entire 3M-vector index is loaded in off-heap memory at startup.

### Request Pipeline

```
POST /fraud-score
  │
  ├─ Netty EpollEventLoop (Linux) / NioEventLoop (macOS/dev)
  │   └─ 1 boss thread + 2 worker threads per instance
  │
  ├─ HttpServerCodec + HttpObjectAggregator (64 KB max body)
  │
  ├─ RequestParser.parse(ByteBuf)                    ~0.02 ms
  │   ├─ Raw byte scan — no String, no Jackson
  │   ├─ ISO-8601 → epoch-minutes (pure integer math)
  │   └─ Writes float[14] into ThreadLocal TL_VEC
  │
  ├─ KnnSearch.quantize(float[14] → byte[14])        ~0.001 ms
  │   └─ f × 127 + round, clamped [-127, 127]
  │
  ├─ KnnSearch.searchIVF(store, vec, vecInt8, ...)   ~1–5 ms
  │   ├─ Phase 1: scan 512 float32 centroids → top 32 (nprobe)
  │   └─ Phase 2: scan ~185k INT8 vectors with early-exit distSq
  │
  ├─ KnnSearch.fraudCount(topIdx[5])                 ~0.001 ms
  │   └─ count labels == FRAUD among top-5
  │
  └─ RESPONSES[fraudCount].duplicate() + writeAndFlush
      └─ Pre-built DirectByteBuf — zero JSON encoding
```

---

## Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Runtime | Java 25 (eclipse-temurin:25-jre) | `sun.misc.Unsafe` + `jdk.incubator.vector` + Panama |
| HTTP Server | Netty 4.2 (no Spring) | Direct ByteBuf, Epoll, zero-copy |
| Vector Index | Custom IVF V4 (K-Means C=512) | No dependency, full control over memory layout |
| Off-heap Storage | `ByteBuffer.allocateDirect` + `sun.misc.Unsafe` | Bypasses GC, raw `getByte` with no bounds check |
| Quantization | INT8 (1 byte/dim vs 4 for float32) | 4× smaller → fits L2 cache better |
| Load Balancer | nginx 1.27-alpine | Minimal CPU footprint, native Epoll, keepalive upstream |
| Build | Maven 3.9 + maven-shade-plugin | Fat JAR with no external files at runtime |
| SIMD | JDK Vector API (`jdk.incubator.vector`, JEP 508) | AVX2/AVX-512 for centroid distance in Phase 1 |

---

## Optimizations

### 1. IVF Index — Inverted File Index ⭐⭐⭐⭐⭐

**The biggest single optimization.** Instead of computing distance to all 3M vectors per query:

1. **Offline (build time):** `ReferenceConverter` runs k-means on all 3M float32 vectors → `C = 512` centroids
2. **Query time Phase 1:** Compute distance from query to all 512 centroids (float32) → select top `nprobe = 32` closest clusters
3. **Query time Phase 2:** Scan only the vectors in those 32 clusters → ~185,000 INT8 distance computations

```
Strategy           Vectors evaluated/query   Worst-case latency
──────────────────────────────────────────────────────────────
Brute force        3,000,000                 ~15 ms
IVF C=512 nprobe=32  ~185,000 (~6%)          ~2 ms
IVF C=512 nprobe=5     ~30,000 (~1%)         < 1 ms  (lower accuracy)
```

The index is persisted in **V4 binary format** (cluster-ordered) — loaded at container startup in milliseconds, no k-means at runtime.

#### V4 Cluster-Ordered Format

In V4, vectors are stored in **cluster order** in the binary file. Cluster 0's vectors come first, then cluster 1's, and so on. This eliminates the need for an indirection array (`listData[]` in V3), making Phase 2 a pure sequential memory scan:

```
V3 (indirect):   for each cluster c → for each idx in listData[off..off+sz] → distSqInt8(idx)
V4 (direct):     for each cluster c → for each li in [off..off+sz]          → distSqInt8(off+li)
```
Sequential access = better CPU prefetch = fewer cache misses.

---

### 2. INT8 Quantization — 42 MB instead of 168 MB ⭐⭐⭐⭐

Converting float32 → int8 at build time shrinks the vector dataset 4×:

```
float f → byte q = round(f × 127), clamped to [-127, 127]

Special case: sentinel -1.0 → -127 (for "no previous transaction" dims 5 and 6)
```

| Format | Size (3M vectors × 14 dims) | Fits L3 cache? |
|--------|----------------------------|----------------|
| float32 | **168 MB** | No (typical L3: 8–32 MB) |
| INT8 | **42 MB** | Better cache locality |

This also enables **two instances** to fit within the 350 MB total limit:

```
Per instance:    42 MB vectors + 3 MB labels + 64 MB JVM heap + ~40 MB JVM overhead ≈ 149 MB
Two instances:   149 × 2 = 298 MB + nginx 16 MB = 314 MB  ✓ under 350 MB
```

The `ReferenceConverter` performs quantization in a single pass when writing the binary — zero overhead at runtime.

---

### 3. Early Termination in `distSqInt8` ⭐⭐⭐

The inner distance loop exits as soon as the partial sum exceeds the current K-th worst distance (`threshold`):

```java
int distSqInt8(final byte[] query, final int idx, final int threshold) {
    final long base = vectorsAddr + (long)idx * 14;
    int sum = 0, d;
    d = (int)query[0]  - (int)UNSAFE.getByte(base);    sum += d*d; if (sum >= threshold) return sum;
    d = (int)query[1]  - (int)UNSAFE.getByte(base+1);  sum += d*d; if (sum >= threshold) return sum;
    d = (int)query[2]  - (int)UNSAFE.getByte(base+2);  sum += d*d; if (sum >= threshold) return sum;
    // ... dims 3–12 same pattern ...
    d = (int)query[13] - (int)UNSAFE.getByte(base+13); sum += d*d;
    return sum;
}
```

**How it improves over time within a query:**
- First K=5 vectors: no threshold yet → full 14-dim scan
- After 5 vectors found: `threshold = worstDist` → vectors rejected after avg ~7 dims
- As better neighbors are found: `threshold` narrows → vectors rejected after avg ~3 dims

Savings: **30–50% fewer dimension comparisons** in a typical query once the top-5 heap is reasonably full.

---

### 4. `sun.misc.Unsafe` for Off-Heap Access

Vectors and labels live in native memory allocated with `ByteBuffer.allocateDirect()`. Access goes through `Unsafe.getByte(address)` — **no Java bounds check, no virtual dispatch**:

```java
// Address computed once per vector — single pointer arithmetic
final long base = vectorsAddr + (long)idx * VEC_STRIDE;  // VEC_STRIDE = 14
UNSAFE.getByte(base + d)  // raw C-like memory read
```

Compared to a heap `byte[][]` approach:
- No GC scanning of 3M objects
- Better CPU cache behaviour (contiguous memory)
- No object header overhead (12–16 bytes per array)

---

### 5. Epoll Transport on Linux ⭐⭐

In production, Netty automatically switches to the Linux Epoll transport instead of Java NIO's selector:

```java
final boolean useEpoll = Epoll.isAvailable(); // true on Linux containers
final IoHandlerFactory ioFactory = useEpoll
    ? EpollIoHandler.newFactory()
    : NioIoHandler.newFactory();
final Class<? extends ServerChannel> channelClass = useEpoll
    ? EpollServerSocketChannel.class
    : NioServerSocketChannel.class;
```

Epoll eliminates the `epoll_wait` → `selector.select()` → demultiplex overhead of Java NIO. At high concurrency (~400 rps per instance), this saves ~15% in I/O latency.

---

### 6. Zero-Allocation Hot Path ⭐⭐⭐

Every scratch buffer that would be allocated per-request is stored in `ThreadLocal`s, initialized **once** per I/O thread during the first request:

```java
// KnnSearch.java
static final ThreadLocal<float[]> TL_DIST       = ThreadLocal.withInitial(() -> new float[5]);
static final ThreadLocal<int[]>   TL_IDX        = ThreadLocal.withInitial(() -> new int[5]);
static final ThreadLocal<byte[]>  TL_VEC_INT8   = ThreadLocal.withInitial(() -> new byte[14]);
static final ThreadLocal<float[]> TL_PROBE_DIST = ThreadLocal.withInitial(() -> new float[128]);
static final ThreadLocal<int[]>   TL_PROBE_IDX  = ThreadLocal.withInitial(() -> new int[128]);

// RequestParser.java
static final ThreadLocal<float[]> TL_VEC = ThreadLocal.withInitial(() -> new float[14]);
```

With 2 worker threads per instance, these arrays are allocated exactly **twice** per JVM — not once per request. Heap allocation in the hot path after warmup: **zero bytes**.

The only unavoidable per-request object: `DefaultFullHttpResponse` (Netty HTTP/1.1 framing). Its cost is amortized by `PooledByteBufAllocator`.

---

### 7. Pre-encoded Static Responses

The fraud score is always `n/5` for `n ∈ {0,1,2,3,4,5}`. All 6 JSON responses are encoded into static `Unpooled.directBuffer` objects at startup:

```java
private static final ByteBuf[] RESPONSES = {
    staticBuf("{\"approved\":true,\"fraud_score\":0.0}"),   // fraudCount = 0
    staticBuf("{\"approved\":true,\"fraud_score\":0.2}"),   // fraudCount = 1
    staticBuf("{\"approved\":true,\"fraud_score\":0.4}"),   // fraudCount = 2
    staticBuf("{\"approved\":false,\"fraud_score\":0.6}"),  // fraudCount = 3
    staticBuf("{\"approved\":false,\"fraud_score\":0.8}"),  // fraudCount = 4
    staticBuf("{\"approved\":false,\"fraud_score\":1.0}"),  // fraudCount = 5
};
```

Response encoding cost per request: **1 array lookup + 1 `ByteBuf.duplicate()`** (shallow copy of pointer/length — no byte copying).

---

### 8. Zero-Copy JSON Parser

`RequestParser` scans the raw Netty `ByteBuf` directly — no `String` creation, no Jackson, no POJO mapping:

```
Technique                      Allocation?   Cost
─────────────────────────────────────────────────
buf.getByte(i)                 none          ~1 ns
Pattern scan (indexOf)         none          O(n) byte compare
readFloat() from raw bytes     none          ~5 ns per number
read2()/read4Digits()          none          2–4 getByte + arithmetic
isoToEpochMin() timestamp      none          ~10 integer ops
merchantKnown() comparison     none          byte-by-byte in buffer
```

**ISO-8601 timestamp parsing** without `DateTimeFormatter` or any Date object:
```java
// "2026-03-11T20:23:35Z" → epoch minutes, pure integer arithmetic
private static long isoToEpochMin(ByteBuf buf, int pos) {
    int y  = read2(buf, pos) * 100 + read2(buf, pos+2);   // 2026
    int mo = read2(buf, pos+5);                            // 03
    int d  = read2(buf, pos+8);                            // 11
    int h  = read2(buf, pos+11);                           // 20
    int mi = read2(buf, pos+14);                           // 23
    return epochMin(y, mo, d, h, mi);   // Gregorian calendar arithmetic
}
```

**Day-of-week** uses Tomohiko Sakamoto's formula — no Calendar, no lookup table beyond a 12-entry switch.

---

### 9. JIT Warmup — 10,000 Queries Before `READY = true`

HotSpot's C2 compiler triggers at `~10,000` method invocations (with `CompileThreshold=1000`). The server runs a synthetic workload before accepting real traffic, ensuring all hot paths are at peak C2 optimization:

```java
private static void warmup(final OffHeapVectorStore store) {
    final float[] vec     = new float[14];
    final byte[]  vecInt8 = new byte[14];
    final float[] topDist = new float[5];
    final int[]   topIdx  = new int[5];

    long seed = 0x9E3779B97F4A7C15L;  // XorShift64 — no allocation, deterministic

    for (int i = 0; i < 10_000; i++) {
        for (int d = 0; d < 14; d++) {
            seed ^= seed << 13; seed ^= seed >>> 7; seed ^= seed << 17;
            vec[d] = Float.intBitsToFloat(((int)(seed >>> 41) & 0x007FFFFF) | 0x3F800000) - 1f;
        }
        KnnSearch.quantize(vec, vecInt8);
        KnnSearch.searchIVF(store, vec, vecInt8, topDist, topIdx, store.defaultNprobe);
    }
}
```

Methods warmed up: `distSqInt8`, `centDistSq`, `insertProbe`, `insert`, `searchIVF`, `fraudCount` — all the hot path.

---

### 10. Aggressive JVM Flags

```bash
-server                          # enable server JIT (default in JDK 25, explicit for clarity)
-XX:+UseSerialGC                 # no GC threads — single-threaded stop-the-world
                                 # correct for: small heap + low allocation rate + 0.5 CPU
-Xms64m -Xmx64m                 # heap pre-allocated at startup — no GC expansion pause ever
-XX:+AlwaysPreTouch              # touch all heap pages at JVM init — no page fault at request time
-XX:+TieredCompilation           # C1 → C2 compilation pipeline
-XX:CompileThreshold=1000        # reach C2 after 1000 invocations (default: 10000)
-XX:Tier4CompileThreshold=500    # C2 (tier 4) compiled sooner for the hottest methods
-XX:+DoEscapeAnalysis            # scalar-replace short-lived objects onto the stack
-XX:+EliminateAllocations        # elide allocations that escape analysis confirms safe
-XX:+DisableExplicitGC           # ignore any System.gc() calls from libraries
-Dio.netty.allocator.type=pooled # force pool allocator even if heuristics say unpooled
-Dio.netty.recycler.maxCapacityPerThread=4096  # increase object recycler capacity
```

Why `SerialGC` over `G1` or `ZGC`? With `-Xmx64m` and near-zero allocation in the hot path, GC pauses are negligible regardless. `SerialGC` saves ~3 threads of CPU time vs `G1GC` — critical at 0.5 CPU per instance.

---

### 11. SIMD Distance — JDK Vector API

`SimdDistance` uses `jdk.incubator.vector` (JEP 508, 10th incubator in JDK 25) for AVX2/AVX-512 float32 distance. Used in **Phase 1** (centroid scanning, float32 on heap):

```java
static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
// SPECIES_PREFERRED selects widest SIMD width at runtime:
//   SSE4:   128-bit → 4 lanes
//   AVX2:   256-bit → 8 lanes  ← typical on x86 since 2013
//   AVX-512: 512-bit → 16 lanes

static float distSq(final float[] a, final float[] b) {
    var acc = FloatVector.zero(SPECIES);
    // Full-width passes (no masked load — uses JIT fast intrinsic path)
    for (int i = 0; i <= BOUND; i += LANES) {
        var d = FloatVector.fromArray(SPECIES, a, i)
                           .sub(FloatVector.fromArray(SPECIES, b, i));
        acc = d.fma(d, acc);   // fused: d² + acc — 1 instruction, 1 rounding
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    // Scalar tail for remaining dims (DIMS=14, LANES=8 → 6 scalar ops)
    for (int i = TAIL_START; i < DIMS; i++) { float d = a[i]-b[i]; sum += d*d; }
    return sum;
}
```

For 512 centroids × 14 dims: **7,168 float ops per query** in Phase 1 — negligible compared to Phase 2.

---

### 12. PooledByteBufAllocator

```java
.childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
```

Netty's pooled allocator reuses `ByteBuf` memory across requests via thread-local pool arenas. Without it, each HTTP response body triggers a `malloc`/`free` cycle, adding GC pressure.

---

## Memory Layout per Instance

```
────────────────────────────────────────────────────────
 Region               Location    Size      Notes
────────────────────────────────────────────────────────
 INT8 vectors         off-heap    ~40 MB    3M × 14 bytes, ByteBuffer.allocateDirect
 Labels               off-heap     ~3 MB    3M × 1 byte,  ByteBuffer.allocateDirect
 Float32 centroids    on-heap     ~28 KB    512 × 14 × 4 bytes
 IVF offsets/sizes    on-heap      ~4 KB    512 ints each
 JVM heap (-Xmx64m)  on-heap      64 MB    app + Netty + JIT data structures
 JVM native overhead  native      ~40 MB    JIT code cache, class metadata, threads
────────────────────────────────────────────────────────
 Total                           ~149 MB    ✓ within 167 MB container limit
────────────────────────────────────────────────────────
```

---

## Binary Format V4 — `references.bin`

Generated once during `docker build` by `ReferenceConverter`. Loaded at startup by `VectorLoader` in ~1 second.

```
Offset   Size    Field
──────────────────────────────────────────────────────────
0        4       magic    = 0x52494E48  ("RINH", little-endian)
4        4       version  = 4
8        4       count    = N           (3,000,000)
12       4       dims     = 14
16       4       clusters = C           (512)
20       4       nprobe   = 32          (default at query time)
24       8       reserved = 0
──────────────────────────────────────────────────────────
32       N×14    INT8 vectors           (cluster-ordered)
32+N×14  N×1     Labels                 (cluster-ordered, 0=legit 1=fraud)
+N×C×56  C×56    Centroids              (C × 14 × float32 LE)
+C×4     C×4     Cluster sizes          (int32 LE)
──────────────────────────────────────────────────────────
Total: ~46 MB (vs 284 MB uncompressed JSON / 16 MB gzipped)
```

No `listData[]` array in V4 (eliminated by cluster-ordering) → saves `N × 4 = 12 MB` vs V3.

---

## Build Process

The Docker build has **3 stages**:

```
Stage 1 — maven:3.9-eclipse-temurin-25 (builder)
  └─ mvn package -DskipTests
     └─ Compiles all Java, produces fat JAR (~8 MB)

Stage 2 — eclipse-temurin:25-jre (converter)
  └─ java org.rinha.ReferenceConverter references.json.gz references.bin
     ├─ Parse 3M JSON vectors           (~30s)
     ├─ K-means C=512, 15 iterations    (~3–8 min depending on CPU)
     ├─ Build IVF lists + quantize      (~10s)
     └─ Write V4 binary (~46 MB)        (~5s)

Stage 3 — eclipse-temurin:25-jre (runtime)
  └─ COPY app.jar + references.bin
     └─ java [JVM flags] org.rinha.Main
        ├─ VectorLoader.loadOffHeap()   (~1s, mmap-speed read)
        ├─ JIT warmup 10k queries       (~3s)
        └─ READY → accept traffic
```

---

## Running Locally

```bash
# 1. Get the dataset (from the official contest repo)
curl -L -o resources/references.json.gz \
  https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz

# 2. Build for linux/amd64 (required if you're on macOS ARM)
docker buildx build \
  --platform linux/amd64 \
  -t carlosportella16/rinha-2026:latest \
  --push \
  .

# 3. Start the stack (nginx + api1 + api2)
docker compose up

# 4. Health check
curl -s http://localhost:9999/ready
# → OK

# 5. Test fraud detection
curl -s -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-test",
    "transaction": {"amount": 9500.0, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
    "customer": {"avg_amount": 81.0, "tx_count_24h": 20, "known_merchants": ["MERC-008"]},
    "merchant": {"id": "MERC-999", "mcc": "7995", "avg_amount": 54.0},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 952.0},
    "last_transaction": null
  }'
# → {"approved":false,"fraud_score":1.0}
```

---

## Project Structure

```
rinha-backend-2026/
│
├── src/main/java/org/rinha/
│   ├── Main.java                # Netty bootstrap: Epoll/NIO, boss+2 workers, /ready + /fraud-score
│   ├── RequestParser.java       # Zero-alloc JSON parser: raw ByteBuf byte scan, ISO-8601 arithmetic
│   ├── KnnSearch.java           # IVF search (searchIVF), quantize(), fraudCount(), ThreadLocals
│   ├── OffHeapVectorStore.java  # Off-heap INT8 store: distSqInt8 with early-exit, Unsafe access
│   ├── VectorLoader.java        # Binary loader: V2 (float32 legacy), V3 (IVF+INT8), V4 (cluster-ordered)
│   ├── ReferenceConverter.java  # Build-time: JSON.gz → V4 binary (parallel k-means + quantization)
│   ├── SimdDistance.java        # JDK Vector API: float32 DistSq with AVX2/AVX-512 intrinsics
│   ├── VectorStore.java         # On-heap float32 store + dim-0 sorted index (used in tests)
│   └── DistanceBenchmark.java   # Micro-benchmark: SIMD vs scalar distSq
│
├── src/test/java/org/rinha/
│   └── KnnSearchTest.java       # Correctness tests: IVF recall vs brute-force
│
├── Dockerfile                   # 3-stage build: compile → k-means → runtime
├── docker-compose.yml           # nginx + api1 + api2 with resource limits
├── nginx.conf                   # Upstream keepalive, epoll, tcp_nodelay, access_log off
├── pom.xml                      # Java 25, Netty 4.2, maven-shade, Vector API compile flag
├── info.json                    # Participant metadata
├── participants/
│   └── carlosportella16.json    # Registration file for the official contest repo
└── benchmark-results/           # Local benchmark runs (gitignored in submission)
```

---

## Resource Distribution

```yaml
# docker-compose.yml
nginx:  cpus: "0.05"   memory: "16MB"   # proxy only — no business logic
api1:   cpus: "0.475"  memory: "167MB"
api2:   cpus: "0.475"  memory: "167MB"
# ──────────────────────────────────────
# Total: 1.00 CPU / 350 MB  ✓
```

---

## Pending Optimizations

The following improvements are identified but not yet implemented:

| # | Optimization | Expected gain | Complexity |
|---|-------------|--------------|-----------|
| A | Reorder dims by variance (early-exit fires earlier) | +10–30% Phase 2 speed | Medium |
| B | `ByteVector` SIMD for INT8 off-heap distance (Phase 2) | +30–50% Phase 2 speed | High |
| C | `mmap` shared between instances (save ~45 MB RAM) | −45 MB physical RAM | Medium |
| D | Eliminate `try/catch` in `RequestParser.parse()` (C2 inlining) | +5–10% parse | Low |
| E | Warmup using real centroid vectors instead of XorShift random | Better JIT branch prediction | Low |

---

## License

MIT

