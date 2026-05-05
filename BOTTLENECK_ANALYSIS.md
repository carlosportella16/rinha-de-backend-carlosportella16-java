# Análise de Gargalos — Rinha de Backend 2026

Score atual: **2184.94** → Meta: **≥ 4500**

---

## 1. Diagnóstico do Score Atual

```
p99 = 267.09ms  →  p99_score  =  573.34   (de 3000 possíveis)
FP  =  53       →
FN  =  66       →  detection  = 1611.60   (de ~3000 possíveis)
E   = 251       →
ε   = 0.00466   →
final_score = 2184.94
```

### Onde estão os pontos perdidos?

| Componente | Atual | Máximo | Pontos perdidos |
|---|---|---|---|
| `p99_score` | 573 | 3000 | **−2427** |
| `detection_score` | 1611 | ~2800 | −1189 |
| **Total** | **2184** | **~5800** | ~3616 |

**p99 = 267ms é o gargalo crítico.** Sozinho representa 73% dos pontos perdidos.

---

## 2. Causa Raiz do p99 = 267ms — CPU Throttling por CFS

O ambiente de teste é um Mac Mini 2014 (Ubuntu 24.04), onde o Docker usa o **CFS (Completely Fair Scheduler)** para impor limites de CPU. Este é o mecanismo:

```
CFS period = 100ms
API instance quota = 0.45 CPU × 100ms = 45ms de CPU por período
```

**O que acontece em carga alta:**

1. k6 aumenta VUs progressivamente, chegando a pico de ~1500–2000 req/s total.
2. Cada instância precisa lidar com ~750–1000 req/s.
3. Com ~1ms de CPU por request (IVF search com nprobe=8), cada instância precisa de 750–1000ms de CPU por segundo.
4. Mas o CFS só fornece 450ms de CPU por segundo (0.45 CPU).
5. **Resultado: a instância fica em throttle ~35–55% do tempo.** Pedidos na fila esperam o próximo período de 100ms, podendo acumular latência de 100–300ms.

**Evidência**: O p99 = 267ms é condizente com requests esperando 1–3 períodos de CFS (100–300ms) enquanto o container está throttled.

### Por que as boas métricas de detecção não ajudam?

Com detection_score = 1611 e o máximo teórico de ~2800:
- **FP = 53** contribui com peso 1: 53 pontos de penalidade
- **FN = 66** contribui com peso 3: 198 pontos de penalidade (FN é 3× pior)
- Total ponderado E = 251, ε = 0.00466

A detecção já está boa (0.22% failure rate). O trabalho está no p99.

---

## 3. Mapeamento Completo de Gargalos (por impacto)

### 🔴 CRÍTICO — Latência (p99)

#### 3.1 NPROBE=8 → 4 (mais fácil, maior impacto imediato)

**Problema**: Com C=1024 clusters e nprobe=8:
- Fase 2 escaneia ~8 × 2929 = **23.432 vetores** por request
- Custo estimado de CPU: ~0.6–0.8ms por request
- Essa é a maior fatia do budget de CPU por request

**Solução**: Mudar `NPROBE=8` para `NPROBE=4` no `docker-compose.yml`.

Impacto esperado:
- Fase 2 escaneia ~11.716 vetores → **metade do trabalho computacional**
- CPU por request cai para ~0.3–0.4ms
- Throughput máximo sobe de 450 para ~900 req/s por instância
- p99 esperado: 267ms → **~30–60ms**

Risco: Redução na qualidade do recall. Monitorar FP/FN no próximo teste. Se FP+FN aumentar muito, tentar NPROBE=6.

**Mudança no `docker-compose.yml`:**
```yaml
environment:
  - NPROBE=4   # era 8
```

#### 3.2 Worker Threads: 2 → 1 por instância

**Problema**: A instância tem 0.45 CPU. Com 2 worker threads + 1 boss thread:
- 3 threads competindo por 0.45 CPU via CFS
- Context switching entre threads consome CPU do budget
- A fila de requests distribui entre 2 workers, cada um executando com 0.225 CPU efetivo

**Solução**: `new MultiThreadIoEventLoopGroup(1, ioFactory)` para workers em `Main.java`.

Com 1 worker thread:
- Sem context switching entre workers
- O único thread worker usa 100% do budget de CPU disponível
- Netty é event-driven — 1 thread é suficiente para I/O não-bloqueante

**Mudança em `Main.java`:**
```java
// Era: new MultiThreadIoEventLoopGroup(2, ioFactory)
final MultiThreadIoEventLoopGroup worker = new MultiThreadIoEventLoopGroup(1, ioFactory);
```

#### 3.3 Respostas HTTP pré-construídas (eliminação de alocações por request)

**Problema**: Em cada request, o código cria:
```java
// Aloca um DefaultFullHttpResponse (+ DefaultHttpHeaders internamente)
final FullHttpResponse res = new DefaultFullHttpResponse(
    HttpVersion.HTTP_1_1, status, slice, HEADERS_FACTORY, TRAILERS_FACTORY);
res.headers()
    .set(HttpHeaderNames.CONTENT_TYPE, CONTENT_TYPE_JSON)
    .setInt(HttpHeaderNames.CONTENT_LENGTH, slice.readableBytes());
```

Isso cria 1–2 objetos por request. Com SerialGC e 900 req/s por instância, isso contribui para pausas de GC que adicionam jitter ao p99.

**Solução**: Pré-construir as 6 respostas completas HTTP/1.1 (header + body) como `ByteBuf` estáticos e escrever diretamente no canal, bypassing o encoder do `HttpServerCodec`.

```java
// 6 respostas HTTP/1.1 completas pré-construídas
private static final ByteBuf[] HTTP_RESPONSES = buildHttpResponses();

private static ByteBuf[] buildHttpResponses() {
    String[] bodies = {
        "{\"approved\":true,\"fraud_score\":0.0}",
        "{\"approved\":true,\"fraud_score\":0.2}",
        "{\"approved\":true,\"fraud_score\":0.4}",
        "{\"approved\":false,\"fraud_score\":0.6}",
        "{\"approved\":false,\"fraud_score\":0.8}",
        "{\"approved\":false,\"fraud_score\":1.0}",
    };
    ByteBuf[] r = new ByteBuf[bodies.length];
    for (int i = 0; i < bodies.length; i++) {
        String b = bodies[i];
        String http = "HTTP/1.1 200 OK\r\n" +
            "Content-Type: application/json\r\n" +
            "Content-Length: " + b.length() + "\r\n" +
            "Connection: keep-alive\r\n\r\n" + b;
        byte[] bytes = http.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        r[i] = Unpooled.unreleasableBuffer(
            Unpooled.directBuffer(bytes.length).writeBytes(bytes));
    }
    return r;
}
```

Isso exige adaptar o pipeline para não passar pelo `HttpServerCodec` encoder na direção de saída (ou processar os bytes brutos diretamente).

**Alternativa mais simples**: No `send()`, usar `PooledByteBufAllocator` para o `DefaultFullHttpResponse` e retorná-lo ao pool após flush:
```java
// ctx.writeAndFlush já libera o ByteBuf após envio via SimpleChannelInboundHandler
// Mas DefaultFullHttpResponse não vai para pool — é o problema
```

A solução mais segura é remover `HttpServerCodec` do pipeline e escrever as respostas pré-construídas com `ctx.writeAndFlush(HTTP_RESPONSES[idx].retainedDuplicate(), ctx.voidPromise())`.

#### 3.4 io_uring (segundo maior ganho de I/O)

**Problema**: Epoll usa `epoll_wait` + `recv`/`send` syscalls individuais. Em alta carga, o overhead de syscall acumula.

**Solução**: Netty Incubator com io_uring batcha syscalls, reduzindo context switches em ~30–40%.

**Dependência no `pom.xml`:**
```xml
<dependency>
    <groupId>io.netty.incubator</groupId>
    <artifactId>netty-incubator-transport-native-io_uring</artifactId>
    <version>0.0.26.Final</version>
    <classifier>linux-x86_64</classifier>
</dependency>
```

**Mudança em `Main.java`:**
```java
final boolean useIoUring = io.netty.incubator.channel.uring.IOUring.isAvailable();
final boolean useEpoll   = !useIoUring && Epoll.isAvailable();

final IoHandlerFactory ioFactory = useIoUring
    ? io.netty.incubator.channel.uring.IOUringIoHandler.newFactory()
    : useEpoll ? EpollIoHandler.newFactory() : NioIoHandler.newFactory();

final Class<? extends ServerChannel> channelClass = useIoUring
    ? io.netty.incubator.channel.uring.IOUringServerSocketChannel.class
    : useEpoll ? EpollServerSocketChannel.class
    : NioServerSocketChannel.class;
```

#### 3.5 HAProxy: Unix Domain Socket (UDS) entre LB e APIs

**Problema**: A comunicação HAProxy → API1/API2 passa pelo stack TCP (loopback). Cada request tem overhead de:
- TCP connection (com `http-reuse always` isso é amortizado)
- Kernel TCP receive buffer copy

**Solução**: Configurar UDS entre HAProxy e cada API. O código já suporta via `SOCKET_PATH`. Precisa de volume compartilhado no docker-compose.

```yaml
# docker-compose.yml
volumes:
  sockets:

services:
  haproxy:
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - sockets:/run/sockets

  api1:
    environment:
      - SOCKET_PATH=/run/sockets/api1.sock
    volumes:
      - sockets:/run/sockets

  api2:
    environment:
      - SOCKET_PATH=/run/sockets/api2.sock
    volumes:
      - sockets:/run/sockets
```

```cfg
# haproxy.cfg
backend apis
    balance roundrobin
    http-reuse always
    server api1 /run/sockets/api1.sock check inter 2s fall 3 rise 1
    server api2 /run/sockets/api2.sock check inter 2s fall 3 rise 1
```

Ganho estimado: ~1–5μs por request na latência de transporte LB→API. Pequeno mas consistente.

#### 3.6 HAProxy: Otimizações de configuração

**Problemas no `haproxy.cfg` atual:**
- `maxconn 4096` global sem `nbthread 1` (com 0.10 CPU, threads extras desperdiçam CPU quota)
- Sem `option http-server-close` — conexões keep-alive para as APIs podem monopolizar workers
- Sem `timeout http-keep-alive` — conexões ociosas ficam abertas, ocupando slots

**Configuração otimizada:**
```cfg
global
    maxconn 8192
    nbthread 1        # 0.10 CPU → 1 thread é suficiente
    log stderr local0 notice

defaults
    mode http
    timeout connect 200ms
    timeout client  4s
    timeout server  4s
    timeout http-keep-alive 500ms
    option dontlognull
    option http-server-close   # fecha conexão API após response, libera workers

frontend front
    bind :9999
    default_backend apis

backend apis
    balance roundrobin
    http-reuse safe            # era 'always' — 'safe' evita HOL blocking
    option httpchk GET /ready
    server api1 api1:9998 check inter 2s fall 3 rise 1
    server api2 api2:9998 check inter 2s fall 3 rise 1
```

#### 3.7 JVM: Tamanho do Heap e GC

**Problema**: `-Xms100m -Xmx100m` com SerialGC:
- A store de vetores usa ~45MB no heap (3M × 14 bytes + labels + metadata)
- Sobram apenas ~55MB para o young gen
- SerialGC é single-threaded e para todos os threads durante coleta
- Com allocação de resposta HTTP por request, pode triggerar GC na fase de carga máxima

**Solução 1 (sem risco)**: Aumentar heap para usar mais da RAM disponível.
```dockerfile
# Container limit: 160MB → JVM heap pode ser 120MB
-Xms120m -Xmx120m   # era 100m/100m
```

**Solução 2 (para menor jitter de GC)**: Mudar para G1GC com meta de pause baixa:
```dockerfile
-XX:+UseG1GC
-XX:MaxGCPauseMillis=10    # GC concurrent, pauses curtas
-XX:G1NewSizePercent=30
-XX:G1MaxNewSizePercent=50
# Remover: -XX:+UseSerialGC
```

Risco: G1GC usa ~2 background threads que competem com o worker thread pelo CPU quota. Em testes, comparar p99 com SerialGC vs G1GC.

**Solução 3 (alvo de zero-allocation)**: Se o `send()` não alocar por request (ver item 3.3), SerialGC + heap maior é suficiente.

---

### 🟡 ALTO — Detecção (FP/FN)

#### 3.8 k-means++ para melhor qualidade dos clusters

**Problema**: `ReferenceConverter.java` usa inicialização aleatória (`rng.nextInt(N)`) para os centroides iniciais do k-means.

**Impacto**: Inicialização aleatória pode produzir clusters desiguais ou localmente ótimos, aumentando a distância média centroide→vetor e reduzindo o recall do IVF.

**k-means++ garante**: centroides iniciais bem espalhados → clusters mais uniformes → melhor recall para o mesmo nprobe.

```java
// Substituir initCentroids() em ReferenceConverter.java
private static float[] initCentroids(float[] allVecs, int N) {
    final float[] centroids = new float[C * VectorStore.DIMS];
    final Random rng = new Random(0x42424242L);
    final int DIMS = VectorStore.DIMS;

    // 1. Escolher o primeiro centroide aleatoriamente
    int first = rng.nextInt(N);
    System.arraycopy(allVecs, first * DIMS, centroids, 0, DIMS);

    // 2. Para cada centroide seguinte: probabilidade proporcional a distância²
    final float[] minDist = new float[N];
    Arrays.fill(minDist, Float.MAX_VALUE);

    for (int c = 1; c < C; c++) {
        // Atualizar distâncias mínimas ao novo centroide (c-1)
        final int prevBase = (c - 1) * DIMS;
        double totalDist = 0;
        for (int i = 0; i < N; i++) {
            final float d = vecDistSq14(allVecs, i * DIMS, centroids, prevBase);
            if (d < minDist[i]) minDist[i] = d;
            totalDist += minDist[i];
        }
        // Amostra proporcional a minDist²
        double target = rng.nextDouble() * totalDist;
        int chosen = 0;
        for (int i = 0; i < N; i++) {
            target -= minDist[i];
            if (target <= 0) { chosen = i; break; }
        }
        System.arraycopy(allVecs, chosen * DIMS, centroids, c * DIMS, DIMS);
    }
    return centroids;
}
```

**Resultado esperado**: Redução de ~20–40% nos FP+FN. Com 119 erros totais → ~70–95 erros → detection_score pode subir de 1611 para ~1800–2000.

**Custo**: Rebuild completo da imagem Docker (~10–20 min).

#### 3.9 Mais iterações k-means (15 → 25)

**Problema**: 15 iterações pode não ser suficiente para C=1024. Com k-means++, a convergência é mais rápida, mas aumentar para 20–25 iterações garante clusters mais estáveis.

```java
static final int MAX_ITER = 25;   // era 15
```

#### 3.10 Investigar edge cases no RequestParser

**Problema potencial**: `v[2] = clamp((txAmount / custAvg) / AMT_AVG_RATIO)`

Se `custAvg = 0`:
- Java: `100.0f / 0.0f = Infinity` → clamp retorna 1.0 ✓
- Java: `0.0f / 0.0f = NaN` → clamp retorna NaN ✗

`clamp(NaN)` em Java: `NaN < 0f` é `false`, `NaN > 1f` é `false` → retorna NaN!

Um vetor com NaN em qualquer dimensão vai gerar distâncias NaN, que são tratadas como não-matches, potencialmente causando FN em transações edge case.

**Fix em `RequestParser.java`:**
```java
// Linha 169 — substituir
v[2]  = clamp((txAmount / custAvg) / AMT_AVG_RATIO);
// Por:
v[2]  = (custAvg == 0f) ? 0f : clamp((txAmount / custAvg) / AMT_AVG_RATIO);
```

O dataset tem `"edge_case_count":797` — alguns destes podem ser transações com custAvg=0.

#### 3.11 NPROBE adaptativo (boost para edge cases)

**Observação**: 13 do total de edge cases (797) contribuem desproporcionalmente para os FP/FN. Edge cases são requests com características extremas onde o nprobe=4 pode não ser suficiente.

Uma abordagem é detectar edge cases (e.g., sentinela -1 em v[5] e v[6], com v[0] alto) e aumentar nprobe apenas para eles:

```java
// Em handleFraudScore, após parse:
final int nprobe = (vec[5] < 0 && vec[0] > 0.8f) ? 12 : STORE.defaultNprobe;
KnnSearch.searchIVF(STORE, vec, vecInt8, topDist, topIdx, nprobe);
```

Isso mantém nprobe=4 para 98% dos requests e usa nprobe=12 para potenciais fraudes extremas.

---

### 🟢 MÉDIO — Infraestrutura

#### 3.12 Distribuição de CPU (marginal mas simples)

Atual: HAProxy=0.10, API1=0.45, API2=0.45 (total=1.0)

HAProxy só roteia, não processa. Pode funcionar com 0.05 CPU:

```yaml
haproxy:
  deploy:
    resources:
      limits:
        cpus: "0.05"   # era 0.10
        memory: "30MB"

api1:
  deploy:
    resources:
      limits:
        cpus: "0.475"  # era 0.45
        memory: "160MB"

api2:
  deploy:
    resources:
      limits:
        cpus: "0.475"  # era 0.45
        memory: "160MB"
```

Ganho: +5.5% de CPU para cada API. Pequeno mas grátis.

#### 3.13 Warmup mais agressivo

A `warmup()` atual faz 10k queries. O JIT C2 compila a ~10k invocações, mas o SIMD path (`scanClusterSoA`) pode se beneficiar de mais iterações para que o JIT inlinie tudo corretamente.

```java
// Main.java — warmup mais agressivo
warmup(STORE);       // 10k queries existente
warmup(STORE);       // 2ª rodada para garantir estabilidade C2
warmupParser();      // existente
```

Custo: +~1s no startup. Benefício: menor variância no p99 inicial.

---

## 4. Plano de Ação por Prioridade

```
┌────┬─────────────────────────────────────┬────────┬──────────────────────┬──────────┐
│ #  │ Mudança                             │ Esforço│ Impacto no Score     │ Rebuild? │
├────┼─────────────────────────────────────┼────────┼──────────────────────┼──────────┤
│ 1  │ NPROBE=4 no docker-compose          │ 2 min  │ p99: 267→~50ms (+700)│ Não      │
│ 2  │ Workers: 2→1 em Main.java           │ 5 min  │ p99 -20–40ms         │ Sim      │
│ 3  │ Heap: 100→120MB no Dockerfile       │ 2 min  │ Menos GC jitter      │ Sim      │
│ 4  │ HAProxy config (nbthread,timeouts)  │ 5 min  │ Reduz overhead LB    │ Não      │
│ 5  │ io_uring no pom.xml + Main.java     │ 1h     │ I/O -30% CPU (+200?) │ Sim      │
│ 6  │ Fix NaN em custAvg=0 (RequestParser)│ 5 min  │ Reduz FP/FN edge     │ Sim      │
│ 7  │ k-means++ (ReferenceConverter)      │ 30min  │ detection +200–400   │ Sim      │
│ 8  │ Respostas HTTP pré-construídas      │ 1–2h   │ -100μs/req, GC ↓     │ Sim      │
│ 9  │ UDS HAProxy→APIs                    │ 30min  │ -1–5μs/req           │ Não      │
│ 10 │ NPROBE adaptativo edge cases        │ 30min  │ FP/FN -10–20%        │ Sim      │
└────┴─────────────────────────────────────┴────────┴──────────────────────┴──────────┘
```

---

## 5. Score Projetado por Cenário

### Cenário A: Mudanças rápidas (sem rebuild)
- NPROBE=4 + HAProxy config otimizado
- p99 estimado: ~50ms → p99_score = 1000 × log₁₀(1000/50) = **1301**
- detection_score: 1611 (inalterado)
- **Score total estimado: ~2912**

### Cenário B: Rebuild básico (NPROBE=4 + workers=1 + heap=120m + custAvg fix)
- p99 estimado: ~20–30ms → p99_score ≈ 1523–1699
- FP/FN ligeiramente reduzidos: detection ≈ 1650
- **Score total estimado: ~3200–3350**

### Cenário C: Rebuild completo (B + io_uring + k-means++ + respostas pré-construídas)
- p99 estimado: ~5–10ms → p99_score ≈ 2000–2301
- detection melhorado com k-means++: score ≈ 1900–2100
- **Score total estimado: ~4000–4400**

### Cenário D: Ótimo (C + UDS + NPROBE adaptativo + mais iterações k-means)
- p99 estimado: ~2–5ms → p99_score ≈ 2301–2699
- detection otimizado: score ≈ 2100–2300
- **Score total estimado: ~4400–5000** ✅

---

## 6. Ordem de Implementação Recomendada

```
Passo 1 (5 min, sem rebuild):
  → docker-compose.yml: NPROBE=4, HAProxy: 0.05 CPU, APIs: 0.475 CPU
  → haproxy.cfg: nbthread 1, http-server-close, timeouts otimizados
  → Testar: score esperado ~2900

Passo 2 (30 min, rebuild):
  → Main.java: workers 1 thread
  → Dockerfile: -Xms120m -Xmx120m
  → RequestParser: fix NaN custAvg=0
  → Testar: score esperado ~3300

Passo 3 (2h, rebuild):
  → pom.xml + Main.java: adicionar io_uring
  → Main.java: respostas HTTP pré-construídas
  → Testar: score esperado ~3800–4200

Passo 4 (45 min, rebuild completo com conversão):
  → ReferenceConverter.java: k-means++ + 25 iterações
  → Testar: score esperado ~4500+
```

---

## 7. Observações sobre o Ambiente de Teste

- **Mac Mini 2014, 2.6 GHz**: Core i5 (2 cores físicos + HT). Ubuntu 24.04.
- **Docker CFS throttling**: É o maior vilão. `cfs_period_us=100000` (100ms), throttle quando quota de 45ms se esgota por período.
- **Resultado local ≠ resultado no servidor**: Em Mac local (M1/M2/M3), não há throttling CFS real — o contenedor usa CPU real sem limite de kernel. Por isso testes locais mostram p99 < 5ms mas o servidor mostra 267ms.
- **Soluções que ajudam APENAS no servidor**: Reduzir CPU por request (NPROBE menor, 1 worker, io_uring). No Mac local, o efeito é imperceptível.

---

## 8. Monitoramento Pós-Mudança

Para cada rebuild/mudança, verificar no JSON de resultado:

```json
{
  "p99": "< 10ms",           ← Principal indicador
  "scoring": {
    "breakdown": {
      "false_positive_detections": "< 30",
      "false_negative_detections": "< 40"
    },
    "failure_rate": "< 0.5%",
    "p99_score": { "value": "> 2000", "cut_triggered": false },
    "detection_score": { "value": "> 1800" },
    "final_score":       "> 4500"
  }
}
```

Se `p99_score.cut_triggered = true` ou `detection_score.cut_triggered = true`, há problema grave — investigar antes de continuar.

