package org.rinha;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.channel.MultiThreadIoEventLoopGroup;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollIoHandler;
import io.netty.channel.epoll.EpollServerSocketChannel;
import io.netty.channel.nio.NioIoHandler;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.http.*;
import io.netty.util.AsciiString;
import io.netty.util.CharsetUtil;

public final class Main {

    private static final int PORT = Integer.parseInt(
            System.getenv().getOrDefault("SERVER_PORT", "9999"));

    private static final AsciiString CONTENT_TYPE_JSON = AsciiString.cached("application/json");
    private static final AsciiString CONNECTION        = AsciiString.cached("connection");
    private static final AsciiString KEEP_ALIVE_VALUE  = AsciiString.cached("keep-alive");

    private static final HttpHeadersFactory HEADERS_FACTORY  =
            DefaultHttpHeadersFactory.headersFactory().withValidation(false);
    private static final HttpHeadersFactory TRAILERS_FACTORY =
            DefaultHttpHeadersFactory.trailersFactory().withValidation(false);

    private static final ByteBuf BUF_OK          = staticBuf("OK");
    private static final ByteBuf BUF_NOT_FOUND   = staticBuf("Not Found");
    private static final ByteBuf BUF_METHOD_ERR  = staticBuf("Method Not Allowed");
    private static final ByteBuf BUF_BAD_REQUEST = staticBuf("Bad Request");
    private static final ByteBuf BUF_STARTING    = staticBuf("Starting");

    /*
     * fraud_score is always n/5 for n ∈ {0,1,2,3,4,5}.
     * Pre-encode all 6 responses — zero JSON building at query time.
     */
    private static final ByteBuf[] RESPONSES = {
            staticBuf("{\"approved\":true,\"fraud_score\":0.0}"),
            staticBuf("{\"approved\":true,\"fraud_score\":0.2}"),
            staticBuf("{\"approved\":true,\"fraud_score\":0.4}"),
            staticBuf("{\"approved\":false,\"fraud_score\":0.6}"),
            staticBuf("{\"approved\":false,\"fraud_score\":0.8}"),
            staticBuf("{\"approved\":false,\"fraud_score\":1.0}"),
    };

    private static ByteBuf staticBuf(String s) {
        byte[] b = s.getBytes(CharsetUtil.UTF_8);
        return Unpooled.unreleasableBuffer(
                Unpooled.directBuffer(b.length).writeBytes(b));
    }

    static OffHeapVectorStore STORE;
    static volatile boolean   READY = false;

    public static void main(String[] args) throws Exception {
        // Use Epoll when available (Linux in production) — fewer syscalls, ~15% I/O improvement
        final boolean useEpoll = Epoll.isAvailable();

        final IoHandlerFactory ioFactory = useEpoll
                ? EpollIoHandler.newFactory()
                : NioIoHandler.newFactory();
        final Class<? extends io.netty.channel.ServerChannel> channelClass = useEpoll
                ? EpollServerSocketChannel.class
                : NioServerSocketChannel.class;
        System.out.println("Transport: " + (useEpoll ? "epoll" : "nio"));

        // 1 boss + 2 workers per 0.5-CPU instance (contest: 1 CPU / 2 instances)
        final MultiThreadIoEventLoopGroup boss   = new MultiThreadIoEventLoopGroup(1, ioFactory);
        final MultiThreadIoEventLoopGroup worker = new MultiThreadIoEventLoopGroup(2, ioFactory);

        final Channel channel = new ServerBootstrap()
                .group(boss, worker)
                .channel(channelClass)
                .option(ChannelOption.SO_BACKLOG, 1024)
                .option(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.TCP_NODELAY, true)
                .childOption(ChannelOption.SO_KEEPALIVE, false)
                .childOption(ChannelOption.SO_RCVBUF, 16384)
                .childOption(ChannelOption.SO_SNDBUF, 16384)
                // PooledByteBufAllocator: reuses buffer memory → less GC pressure
                .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ch.pipeline()
                                .addLast(new HttpServerCodec())
                                // 8 KB max body size — real payloads are ~500 bytes;
                                // smaller limit means less zeroing on buffer allocation
                                .addLast(new HttpObjectAggregator(8192))
                                .addLast(RequestHandler.INSTANCE);
                    }
                })
                .bind(PORT)
                .sync()
                .channel();

        try {
            final String binPath = System.getenv().getOrDefault("VECTORS_BIN", "/data/references.bin");
            long t = System.currentTimeMillis();
            System.out.println("Loading vectors from: " + binPath);
            STORE = VectorLoader.loadOffHeap(binPath);
            // Allow runtime override of nprobe via NPROBE env var (default: value baked into .bin header)
            final String nprobeEnv = System.getenv("NPROBE");
            if (nprobeEnv != null && !nprobeEnv.isEmpty()) {
                final int overrideNprobe = Integer.parseInt(nprobeEnv.trim());
                System.out.printf("NPROBE env override: %d → %d%n", STORE.defaultNprobe, overrideNprobe);
                STORE.defaultNprobe = overrideNprobe;
            }
            System.out.printf("Loaded %,d vectors (C=%d, nprobe=%d) in %d ms%n",
                    STORE.size(), STORE.numClusters, STORE.defaultNprobe,
                    System.currentTimeMillis() - t);

            // ── JIT warmup: 10k queries to reach C2 compilation ───────────────
            // HotSpot C2 threshold is ~10k invocations; 1k wasn't enough.
            t = System.currentTimeMillis();
            System.out.println("Running JIT warmup (10 000 queries)…");
            warmup(STORE);
            warmupParser();
            System.out.printf("Warmup done in %d ms%n", System.currentTimeMillis() - t);

            READY = true;
            System.out.println("Ready.");

            channel.closeFuture().sync();
        } finally {
            boss.shutdownGracefully();
            worker.shutdownGracefully();
        }
    }

    /**
     * Runs 10,000 IVF searches with pseudo-random query vectors.
     * Uses XorShift64 — no allocation, deterministic, covers full [-1,1] range.
     * Results discarded; side effect is JIT compilation of the hot path.
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
     * Warm up the RequestParser's byte[] hot path so it is C2-compiled before
     * the first real request arrives.  Uses a minimal but valid JSON body that
     * exercises every code branch (hasLast=true and hasLast=false).
     */
    private static void warmupParser() {
        // A minimal fraud-score body with last_transaction present
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
            final boolean ka = HttpUtil.isKeepAlive(req);
            if (!READY) {
                send(ctx, HttpResponseStatus.SERVICE_UNAVAILABLE, BUF_STARTING, ka);
                return;
            }

            final String     uri = req.uri();
            final HttpMethod m   = req.method();

            if ("/fraud-score".equals(uri)) {
                if (HttpMethod.POST.equals(m)) {
                    handleFraudScore(ctx, req, ka);
                } else {
                    send(ctx, HttpResponseStatus.METHOD_NOT_ALLOWED, BUF_METHOD_ERR, ka);
                }
                return;
            }

            if ("/ready".equals(uri)) {
                if (HttpMethod.GET.equals(m)) {
                    send(ctx, HttpResponseStatus.OK, BUF_OK, ka);
                } else {
                    send(ctx, HttpResponseStatus.METHOD_NOT_ALLOWED, BUF_METHOD_ERR, ka);
                }
                return;
            }

            send(ctx, HttpResponseStatus.NOT_FOUND, BUF_NOT_FOUND, ka);
        }

        /**
         * Hot path — POST /fraud-score.
         * Zero heap allocations after warmup (all scratch arrays are ThreadLocal).
         *
         * Per-call unavoidable allocations (HTTP/1.1 framing):
         *   1. body.duplicate()        — DuplicatedByteBuf (shared bytes)
         *   2. DefaultFullHttpResponse — per-response headers
         */
        private static void handleFraudScore(ChannelHandlerContext ctx,
                                             FullHttpRequest req, boolean ka) {
            final float[] vec     = RequestParser.TL_VEC.get();
            final byte[]  vecInt8 = KnnSearch.TL_VEC_INT8.get();
            final float[] topDist = KnnSearch.TL_DIST.get();
            final int[]   topIdx  = KnnSearch.TL_IDX.get();

            if (!RequestParser.parse(req.content(), vec)) {
                send(ctx, HttpResponseStatus.BAD_REQUEST, BUF_BAD_REQUEST, ka);
                return;
            }

            KnnSearch.quantize(vec, vecInt8);
            KnnSearch.searchIVF(STORE, vec, vecInt8, topDist, topIdx, STORE.defaultNprobe);

            send(ctx, HttpResponseStatus.OK,
                    RESPONSES[KnnSearch.fraudCount(STORE, topIdx)], ka);
        }

        private static void send(ChannelHandlerContext ctx, HttpResponseStatus status,
                                 ByteBuf body, boolean keepAlive) {
            final ByteBuf slice = body.duplicate();
            final FullHttpResponse res = new DefaultFullHttpResponse(
                    HttpVersion.HTTP_1_1, status, slice, HEADERS_FACTORY, TRAILERS_FACTORY);

            res.headers()
                    .set(HttpHeaderNames.CONTENT_TYPE,     CONTENT_TYPE_JSON)
                    .setInt(HttpHeaderNames.CONTENT_LENGTH, slice.readableBytes());

            if (keepAlive) {
                res.headers().set(CONNECTION, KEEP_ALIVE_VALUE);
                ctx.writeAndFlush(res, ctx.voidPromise());
            } else {
                ctx.writeAndFlush(res).addListener(ChannelFutureListener.CLOSE);
            }
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            ctx.close();
        }
    }
}