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

    /*
     * Pre-build complete HTTP/1.1 200 OK responses (headers + body) for each of the
     * 6 possible fraud scores. Responses are written as raw ByteBuf — no
     * DefaultFullHttpResponse allocation per request. Zero GC pressure on hot path.
     * fraud_score = n/5 for n ∈ {0,1,2,3,4,5}
     */
    private static final ByteBuf[] HTTP_OK_RESPONSES = buildHttpResponses();

    private static ByteBuf[] buildHttpResponses() {
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
            r[i] = rawHttpResponse(200, "OK", "application/json", bodies[i]);
        }
        return r;
    }

    private static final ByteBuf HTTP_BAD_REQUEST         = rawHttpResponse(400, "Bad Request",
            "text/plain", "Bad Request");
    private static final ByteBuf HTTP_NOT_FOUND           = rawHttpResponse(404, "Not Found",
            "text/plain", "Not Found");
    private static final ByteBuf HTTP_METHOD_NOT_ALLOWED  = rawHttpResponse(405, "Method Not Allowed",
            "text/plain", "Method Not Allowed");
    private static final ByteBuf HTTP_SERVICE_UNAVAILABLE = rawHttpResponse(503, "Service Unavailable",
            "text/plain", "Starting");
    private static final ByteBuf HTTP_READY_OK            = rawHttpResponse(200, "OK",
            "text/plain", "OK");

    private static ByteBuf rawHttpResponse(int status, String statusText,
                                            String contentType, String body) {
        final String http = "HTTP/1.1 " + status + " " + statusText + "\r\n"
                + "Content-Type: " + contentType + "\r\n"
                + "Content-Length: " + body.length() + "\r\n"
                + "Connection: keep-alive\r\n\r\n"
                + body;
        final byte[] bytes = http.getBytes(CharsetUtil.US_ASCII);
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

        // 1 boss + 1 worker per instance.
        // With 0.475 CPU and CFS throttling, a single event-loop worker is more
        // efficient: no context-switch overhead between 2 workers fighting for the same
        // CPU quota. Netty is event-driven — 1 thread handles all I/O without blocking.
        final MultiThreadIoEventLoopGroup boss   = new MultiThreadIoEventLoopGroup(1, ioFactory);
        final MultiThreadIoEventLoopGroup worker = new MultiThreadIoEventLoopGroup(1, ioFactory);

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
                                // Decode only — responses are raw ByteBuf, no encoder needed
                                .addLast(new HttpRequestDecoder(4096, 8192, 8192))
                                // 8 KB max body size — real payloads are ~500 bytes
                                .addLast(new HttpObjectAggregator(8192))
                                .addLast(RequestHandler.INSTANCE);
                    }
                })
                .bind(PORT)
                .sync()
                .channel();

        // Unix Domain Socket listener — HAProxy routes real traffic here when SOCKET_PATH is set.
        // UDS bypasses the kernel TCP stack entirely, saving ~1–2 µs per request on the
        // LB→backend hop. TCP (above) stays alive for health checks only.
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
                                    .addLast(new HttpRequestDecoder(4096, 8192, 8192))
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

            // ── JIT warmup: 20k queries (2 × 10k) to reach stable C2 compilation ──
            // HotSpot C2 threshold ≈ 10k invocations. Two passes ensure the SIMD path
            // (scanClusterSoA) and all branch paths are fully compiled before traffic.
            t = System.currentTimeMillis();
            System.out.println("Running JIT warmup (20 000 queries)…");
            warmup(STORE);
            warmup(STORE);
            warmupParser();
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
     * Runs 10,000 synthetic searches before opening for traffic.
     * Lets the JIT compile the hot path (distSqInt8, searchIVF, scanClusterSoA)
     * to native code — including SIMD — before the first real request arrives.
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
                // [1,2) − 1 → [0,1); dims 5,6 occasionally get sentinel -1
                vec[d] = Float.intBitsToFloat(((int)(seed >>> 41) & 0x007FFFFF) | 0x3F800000) - 1f;
            }
            KnnSearch.quantize(vec, vecInt8);
            KnnSearch.searchIVF(store, vec, vecInt8, topDist, topIdx, nprobe);
        }
    }

    /**
     * Warms up RequestParser so it is JIT-compiled before traffic arrives.
     * Uses a minimal but complete JSON body covering both branches
     * (with and without last_transaction).
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
         * POST /fraud-score handler — ZERO heap allocations per request.
         * All scratch buffers are ThreadLocal; response is a pre-built static ByteBuf.
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
                    HTTP_OK_RESPONSES[KnnSearch.fraudCount(STORE, topIdx)].duplicate(),
                    ctx.voidPromise());
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            ctx.close();
        }
    }
}

