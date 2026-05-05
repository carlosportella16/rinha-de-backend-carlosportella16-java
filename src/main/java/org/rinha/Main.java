package org.rinha;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.channel.MultiThreadIoEventLoopGroup;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollIoHandler;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.epoll.EpollServerSocketChannel;
import io.netty.channel.nio.NioIoHandler;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.http.*;
import io.netty.util.CharsetUtil;
import java.nio.file.Files;
import java.nio.file.Paths;

public final class Main {

    private static final int PORT = Integer.parseInt(
            System.getenv().getOrDefault("SERVER_PORT", "9999"));

    // ── Pre-encoded complete HTTP/1.1 responses (headers + body, raw bytes) ──
    // Writing raw ByteBuf bypasses HttpResponseEncoder (which only intercepts
    // HttpObject types) — zero allocation on the hot path.
    // fraud_score = n/5 for n ∈ {0,1,2,3,4,5}
    private static final ByteBuf[] HTTP_OK = buildOkResponses();

    private static ByteBuf[] buildOkResponses() {
        final String[] bodies = {
            "{\"approved\":true,\"fraud_score\":0.0}",
            "{\"approved\":true,\"fraud_score\":0.2}",
            "{\"approved\":true,\"fraud_score\":0.4}",
            "{\"approved\":false,\"fraud_score\":0.6}",
            "{\"approved\":false,\"fraud_score\":0.8}",
            "{\"approved\":false,\"fraud_score\":1.0}",
        };
        final ByteBuf[] r = new ByteBuf[bodies.length];
        for (int i = 0; i < bodies.length; i++) {
            r[i] = encodeResponse(200, "OK", "application/json", bodies[i]);
        }
        return r;
    }

    // Non-hot-path responses (error / status)
    private static final ByteBuf HTTP_BAD_REQUEST         = encodeResponse(400, "Bad Request",         "text/plain", "Bad Request");
    private static final ByteBuf HTTP_NOT_FOUND           = encodeResponse(404, "Not Found",           "text/plain", "Not Found");
    private static final ByteBuf HTTP_METHOD_NOT_ALLOWED  = encodeResponse(405, "Method Not Allowed",  "text/plain", "Method Not Allowed");
    private static final ByteBuf HTTP_SERVICE_UNAVAILABLE = encodeResponse(503, "Service Unavailable", "text/plain", "Starting");
    private static final ByteBuf HTTP_READY_OK            = encodeResponse(200, "OK",                  "text/plain", "OK");

    /**
     * Builds a pre-encoded, unreleasable raw HTTP/1.1 response buffer.
     * Written directly to the channel — HttpResponseEncoder passes ByteBuf
     * through unchanged (it only intercepts HttpObject messages).
     */
    private static ByteBuf encodeResponse(int status, String statusText,
                                          String contentType, String body) {
        final String raw = "HTTP/1.1 " + status + " " + statusText + "\r\n"
                + "Content-Type: " + contentType + "\r\n"
                + "Content-Length: " + body.length() + "\r\n"
                + "Connection: keep-alive\r\n\r\n"
                + body;
        final byte[] bytes = raw.getBytes(CharsetUtil.US_ASCII);
        return Unpooled.unreleasableBuffer(
                Unpooled.directBuffer(bytes.length).writeBytes(bytes));
    }

    static OffHeapVectorStore STORE;
    static volatile boolean   READY = false;

    public static void main(String[] args) throws Exception {
        Files.deleteIfExists(Paths.get("/tmp/ready"));

        // Use Epoll when available (Linux in production) — fewer syscalls, ~15% I/O improvement
        final boolean useEpoll = Epoll.isAvailable();

        final IoHandlerFactory ioFactory = useEpoll
                ? EpollIoHandler.newFactory()
                : NioIoHandler.newFactory();
        final Class<? extends io.netty.channel.ServerChannel> channelClass = useEpoll
                ? EpollServerSocketChannel.class
                : NioServerSocketChannel.class;
        System.out.println("Transport: " + (useEpoll ? "epoll" : "nio"));

        // 1 boss + 2 workers per instance.
        // 2 EventLoops allow HAProxy connections to be distributed across threads:
        // while worker-1 is running searchIVF for connection A, worker-2 can
        // read/dispatch connection B — overlapping I/O dispatch with computation.
        // With 0.475 vCPU, 2 threads also increase scheduling opportunities within
        // the CFS period (both threads are runnable while quota remains).
        final MultiThreadIoEventLoopGroup boss   = new MultiThreadIoEventLoopGroup(1, ioFactory);
        final MultiThreadIoEventLoopGroup worker = new MultiThreadIoEventLoopGroup(2, ioFactory);

        // TCP listener — always active; used by health checks (test -f /tmp/ready)
        final Channel channel = new ServerBootstrap()
                .group(boss, worker)
                .channel(channelClass)
                .option(ChannelOption.SO_BACKLOG, 2048)
                .option(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.TCP_NODELAY, true)
                .childOption(ChannelOption.SO_KEEPALIVE, false)
                .childOption(ChannelOption.SO_RCVBUF, 32768)
                .childOption(ChannelOption.SO_SNDBUF, 32768)
                .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ch.pipeline()
                                // HttpServerCodec = HttpRequestDecoder + HttpResponseEncoder.
                                // The encoder is required for correct keep-alive framing;
                                // it silently passes our raw ByteBuf responses unchanged
                                // (encoder only intercepts HttpObject types).
                                .addLast(new HttpServerCodec(4096, 8192, 8192))
                                .addLast(new HttpObjectAggregator(8192))
                                .addLast(RequestHandler.INSTANCE);
                    }
                })
                .bind(PORT)
                .sync()
                .channel();

        // Unix Domain Socket listener (optional, when SOCKET_PATH is set)
        final String socketPath = System.getenv("SOCKET_PATH");
        if (socketPath != null && !socketPath.isEmpty() && useEpoll) {
            Files.deleteIfExists(Paths.get(socketPath));
            new ServerBootstrap()
                    .group(boss, worker)
                    .channel(EpollServerDomainSocketChannel.class)
                    .option(ChannelOption.SO_BACKLOG, 2048)
                    .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                    .childHandler(new ChannelInitializer<Channel>() {
                        @Override
                        protected void initChannel(Channel ch) {
                            ch.pipeline()
                                    .addLast(new HttpServerCodec(4096, 8192, 8192))
                                    .addLast(new HttpObjectAggregator(8192))
                                    .addLast(RequestHandler.INSTANCE);
                        }
                    })
                    .bind(new DomainSocketAddress(socketPath))
                    .sync();
            System.out.println("Unix socket: " + socketPath);
        }

        try {
            final String binPath = System.getenv().getOrDefault("VECTORS_BIN", "/data/references.bin");
            long t = System.currentTimeMillis();
            System.out.println("Loading vectors from: " + binPath);
            STORE = VectorLoader.loadOffHeap(binPath);

            // Allow runtime override of nprobe via NPROBE env var
            final String nprobeEnv = System.getenv("NPROBE");
            if (nprobeEnv != null && !nprobeEnv.isEmpty()) {
                final int overrideNprobe = Integer.parseInt(nprobeEnv.trim());
                System.out.printf("NPROBE env override: %d → %d%n", STORE.defaultNprobe, overrideNprobe);
                STORE.defaultNprobe = overrideNprobe;
            }
            System.out.printf("Loaded %,d vectors (C=%d, nprobe=%d) in %d ms%n",
                    STORE.size(), STORE.numClusters, STORE.defaultNprobe,
                    System.currentTimeMillis() - t);

            // ── JIT warmup: submit 3 × 10k queries via the worker EventLoops ──
            // Running warmup ON the actual worker threads (not main thread) ensures:
            //   1. ThreadLocal buffers (TL_VEC, TL_VEC_INT8, etc.) are allocated in
            //      the threads that will serve real traffic — no first-request penalty.
            //   2. JIT compiles the hot path in the exact execution context that will
            //      be used during the benchmark (same thread, same cache layout).
            t = System.currentTimeMillis();
            System.out.println("Running JIT warmup (3 × 10k queries per worker)…");
            final java.util.concurrent.CountDownLatch latch =
                    new java.util.concurrent.CountDownLatch(2);
            for (int w = 0; w < 2; w++) {
                worker.next().submit(() -> {
                    warmup(STORE);
                    warmup(STORE);
                    warmup(STORE);
                    warmupParser();
                    latch.countDown();
                });
            }
            latch.await();
            System.out.printf("Warmup done in %d ms%n", System.currentTimeMillis() - t);

            READY = true;
            Files.writeString(Paths.get("/tmp/ready"), "OK");
            System.out.println("Ready.");

            channel.closeFuture().sync();
        } finally {
            boss.shutdownGracefully();
            worker.shutdownGracefully();
        }
    }

    /**
     * Runs 10,000 synthetic searches to trigger C2 JIT compilation of the hot path.
     * Uses XorShift64 for random vectors: no allocation, no I/O, deterministic.
     */
    private static void warmup(final OffHeapVectorStore store) {
        final float[] vec      = new float[OffHeapVectorStore.DIMS];
        final byte[]  vecInt8  = new byte[OffHeapVectorStore.DIMS];
        final float[] topDist  = new float[KnnSearch.K];
        final int[]   topIdx   = new int[KnnSearch.K];

        long seed = 0x9E3779B97F4A7C15L;
        final int nprobe = store.defaultNprobe;

        for (int i = 0; i < 10_000; i++) {
            for (int d = 0; d < OffHeapVectorStore.DIMS; d++) {
                seed ^= seed << 13;
                seed ^= seed >>> 7;
                seed ^= seed << 17;
                vec[d] = Float.intBitsToFloat(((int)(seed >>> 41) & 0x007FFFFF) | 0x3F800000) - 1f;
            }
            KnnSearch.quantize(vec, vecInt8);
            KnnSearch.searchIVF(store, vec, vecInt8, topDist, topIdx, nprobe);
        }
    }

    /**
     * Warms up RequestParser so it is JIT-compiled before traffic arrives.
     */
    private static void warmupParser() {
        final String body =
            "{\"transaction\":{\"amount\":100.0,\"installments\":1,"
            + "\"requested_at\":\"2025-01-15T14:30:00Z\"},"
            + "\"customer\":{\"avg_amount\":200.0,\"tx_count_24h\":3,"
            + "\"known_merchants\":[\"merch-abc\"]},"
            + "\"merchant\":{\"id\":\"merch-abc\",\"mcc\":\"5812\","
            + "\"avg_amount\":150.0},"
            + "\"terminal\":{\"is_online\":false,\"card_present\":true,"
            + "\"km_from_home\":5.0},"
            + "\"last_transaction\":{\"timestamp\":\"2025-01-15T12:00:00Z\","
            + "\"km_from_current\":3.0}}";
        final byte[] bytes = body.getBytes(java.nio.charset.StandardCharsets.ISO_8859_1);
        final io.netty.buffer.ByteBuf buf =
                io.netty.buffer.Unpooled.wrappedBuffer(bytes);
        final float[] scratch = new float[VectorStore.DIMS];
        for (int i = 0; i < 10_000; i++) {
            buf.resetReaderIndex();
            RequestParser.parse(buf, scratch);
        }
        buf.release();
    }

    @ChannelHandler.Sharable
    static final class RequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

        static final RequestHandler INSTANCE = new RequestHandler();

        @Override
        protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest req) {
            if (!READY) {
                ctx.writeAndFlush(HTTP_SERVICE_UNAVAILABLE.duplicate(), ctx.voidPromise());
                return;
            }

            final String     uri = req.uri();
            final HttpMethod m   = req.method();

            if ("/fraud-score".equals(uri)) {
                if (HttpMethod.POST.equals(m)) {
                    handleFraudScore(ctx, req);
                } else {
                    ctx.writeAndFlush(HTTP_METHOD_NOT_ALLOWED.duplicate(), ctx.voidPromise());
                }
                return;
            }

            if ("/ready".equals(uri)) {
                if (HttpMethod.GET.equals(m)) {
                    ctx.writeAndFlush(HTTP_READY_OK.duplicate(), ctx.voidPromise());
                } else {
                    ctx.writeAndFlush(HTTP_METHOD_NOT_ALLOWED.duplicate(), ctx.voidPromise());
                }
                return;
            }

            ctx.writeAndFlush(HTTP_NOT_FOUND.duplicate(), ctx.voidPromise());
        }

        /**
         * POST /fraud-score hot path — ZERO heap allocations per request.
         *   - ThreadLocal scratch buffers (vec, vecInt8, topDist, topIdx)
         *   - Pre-encoded response ByteBuf (.duplicate() shares bytes, no copy)
         *   - HttpResponseEncoder passes raw ByteBuf through unchanged
         */
        private static void handleFraudScore(ChannelHandlerContext ctx, FullHttpRequest req) {
            final float[] vec     = RequestParser.TL_VEC.get();
            final byte[]  vecInt8 = KnnSearch.TL_VEC_INT8.get();
            final float[] topDist = KnnSearch.TL_DIST.get();
            final int[]   topIdx  = KnnSearch.TL_IDX.get();

            if (!RequestParser.parse(req.content(), vec)) {
                ctx.writeAndFlush(HTTP_BAD_REQUEST.duplicate(), ctx.voidPromise());
                return;
            }

            KnnSearch.quantize(vec, vecInt8);
            KnnSearch.searchIVF(STORE, vec, vecInt8, topDist, topIdx, STORE.defaultNprobe);

            ctx.writeAndFlush(
                    HTTP_OK[KnnSearch.fraudCount(STORE, topIdx)].duplicate(),
                    ctx.voidPromise());
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            ctx.close();
        }
    }
}

