# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-25 AS builder
WORKDIR /build
COPY pom.xml .
COPY src ./src
RUN mvn -q -T 1C package -DskipTests

# ── Stage 2: Preprocess dataset (k-means + INT8 + IVF) ───────────────────────
FROM eclipse-temurin:25-jre AS converter
WORKDIR /build
COPY --from=builder /build/target/rinha-backend-2026-1.0-SNAPSHOT.jar /app/app.jar
COPY resources/references.json.gz /data/references.json.gz

# Build references.bin: cluster-ordered INT8 + IVF index (v4 format, ~45 MB)
RUN java \
    --add-modules jdk.incubator.vector \
    --add-opens java.base/sun.misc=ALL-UNNAMED \
    --sun-misc-unsafe-memory-access=allow \
    -Xmx512m \
    -cp /app/app.jar org.rinha.ReferenceConverter \
    /data/references.json.gz /data/references.bin && \
    rm /data/references.json.gz

# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM eclipse-temurin:25-jre
WORKDIR /app

COPY --from=builder /build/target/rinha-backend-2026-1.0-SNAPSHOT.jar /app/app.jar
COPY --from=converter /data/references.bin /data/references.bin

EXPOSE 9999

CMD java \
    -server \
    --add-modules jdk.incubator.vector \
    --add-opens java.base/sun.misc=ALL-UNNAMED \
    --sun-misc-unsafe-memory-access=allow \
    -XX:+UseSerialGC \
    -Xms64m -Xmx64m \
    -XX:+AlwaysPreTouch \
    -XX:+TieredCompilation \
    -XX:CompileThreshold=1000 \
    -XX:Tier4CompileThreshold=500 \
    -XX:+OptimizeStringConcat \
    -XX:+DoEscapeAnalysis \
    -XX:+EliminateAllocations \
    -XX:+DisableExplicitGC \
    -Dio.netty.allocator.type=pooled \
    -Dio.netty.recycler.maxCapacityPerThread=4096 \
    -cp /app/app.jar org.rinha.Main
