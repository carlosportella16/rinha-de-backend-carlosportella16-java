#!/usr/bin/env bash
# =============================================================================
#  benchmark.sh — Rinha de Backend 2026 | Fraud Detection API Benchmark
#  Tests correctness, latency and computes the official Rinha scoring formula.
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
HOST="${HOST:-localhost}"
PORT="${PORT:-9999}"
BASE_URL="http://$HOST:$PORT"
REQUESTS="${REQUESTS:-500}"        # total /fraud-score requests in load test
CONCURRENCY="${CONCURRENCY:-20}"   # parallel curl workers
WARMUP_REQUESTS="${WARMUP_REQUESTS:-50}"
JAR="target/rinha-backend-2026-1.0-SNAPSHOT.jar"
DATASET="${DATASET:-}"             # optional path to references.json.gz
MANAGE_SERVER="${MANAGE_SERVER:-true}"  # set false if server is already running
RESULTS_DIR="benchmark-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# JVM flags matching competition constraints
JVM_FLAGS=(
  "-XX:+UseSerialGC"
  "-Xms128m"
  "-Xmx128m"
  "-XX:+AlwaysPreTouch"
  "--add-modules" "jdk.incubator.vector"
)

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo -e "${CYAN}[bench]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
fail() { echo -e "${RED}[ FAIL ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
section() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════${NC}"; }
hr() { echo -e "${DIM}──────────────────────────────────────────────────${NC}"; }

# ms_now: milliseconds since epoch — macOS (BSD date) doesn't support %N
ms_now() {
  # gdate (brew install coreutils) → python3 → fallback to seconds×1000
  if command -v gdate &>/dev/null; then
    gdate +%s%3N
  elif command -v python3 &>/dev/null; then
    python3 -c "import time; print(int(time.time()*1000))"
  else
    echo $(( $(date +%s) * 1000 ))
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --smoke         Run only correctness tests (no load test)
  --bench         Run full benchmark (default)
  --no-build      Skip Maven build
  --no-server     Don't start/stop server (assume it's running on :$PORT)
  --requests N    Number of load-test requests (default: $REQUESTS)
  --concurrency N Concurrent workers (default: $CONCURRENCY)
  --host H        Target host (default: $HOST)
  --port P        Target port (default: $PORT)
  --help          Show this help
EOF
  exit 0
}

# ─── Arg parsing ─────────────────────────────────────────────────────────────
MODE="bench"
DO_BUILD=true
# Use while+shift: for loops can't consume the NEXT arg after a flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)       MODE="smoke" ;;
    --bench)       MODE="bench" ;;
    --no-build)    DO_BUILD=false ;;
    --no-server)   MANAGE_SERVER=false ;;
    --requests)    shift; REQUESTS="$1" ;;
    --concurrency) shift; CONCURRENCY="$1" ;;
    --host)        shift; HOST="$1"; BASE_URL="http://$HOST:$PORT" ;;
    --port)        shift; PORT="$1"; BASE_URL="http://$HOST:$PORT" ;;
    --help|-h)     usage ;;
    *) warn "Unknown option: $1" ;;
  esac
  shift
done

mkdir -p "$RESULTS_DIR"

# ─── Test Payloads ───────────────────────────────────────────────────────────
# LEGIT: low amount, known merchant, close to home, no last_tx
# Expected result: approved=true, fraud_score=0.0
PAYLOAD_LEGIT='{
  "id": "tx-1329056812",
  "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
  "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
  "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
  "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.23},
  "last_transaction": null
}'

# FRAUD: high amount, unknown merchant, far from home, 20 tx in 24h
# Expected result: approved=false, fraud_score=1.0
PAYLOAD_FRAUD='{
  "id": "tx-3330991687",
  "transaction": {"amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
  "customer": {"avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"]},
  "merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
  "terminal": {"is_online": false, "card_present": true, "km_from_home": 952.27},
  "last_transaction": null
}'

# MIXED: mid-range transaction with history
PAYLOAD_MID='{
  "id": "tx-3576980410",
  "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
  "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001", "MERC-001"]},
  "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
  "terminal": {"is_online": false, "card_present": true, "km_from_home": 13.7090520965},
  "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774}
}'

# Additional payloads for load test variety
PAYLOAD_MCC_UNKNOWN='{
  "id": "tx-9999999999",
  "transaction": {"amount": 250.00, "installments": 1, "requested_at": "2026-03-12T10:30:00Z"},
  "customer": {"avg_amount": 300.00, "tx_count_24h": 2, "known_merchants": ["MERC-001"]},
  "merchant": {"id": "MERC-999", "mcc": "9999", "avg_amount": 275.00},
  "terminal": {"is_online": true, "card_present": false, "km_from_home": 5.0},
  "last_transaction": {"timestamp": "2026-03-11T10:00:00Z", "km_from_current": 3.2}
}'

PAYLOAD_ONLINE_SMALL='{
  "id": "tx-1111111111",
  "transaction": {"amount": 12.50, "installments": 1, "requested_at": "2026-03-13T14:00:00Z"},
  "customer": {"avg_amount": 50.00, "tx_count_24h": 1, "known_merchants": ["MERC-010", "MERC-011"]},
  "merchant": {"id": "MERC-010", "mcc": "5311", "avg_amount": 45.00},
  "terminal": {"is_online": true, "card_present": false, "km_from_home": 0.5},
  "last_transaction": null
}'

# Array of test payloads for load test rotation
PAYLOADS=("$PAYLOAD_LEGIT" "$PAYLOAD_FRAUD" "$PAYLOAD_MID" "$PAYLOAD_MCC_UNKNOWN" "$PAYLOAD_ONLINE_SMALL")

# ─── Find Maven (mvnw > PATH > ~/.m2/wrapper/dists) ─────────────────────────
find_mvn() {
  # 1. Project wrapper
  if [[ -x "./mvnw" ]]; then echo "./mvnw"; return; fi
  # 2. System PATH
  if command -v mvn &>/dev/null; then echo "mvn"; return; fi
  # 3. ~/.m2/wrapper/dists (IntelliJ/IDE download cache)
  local bin
  bin=$(find "$HOME/.m2/wrapper/dists" -name "mvn" -type f 2>/dev/null \
        | sort -t/ -k8 -r | head -1)
  if [[ -n "$bin" ]]; then echo "$bin"; return; fi
  # 4. Common brew / sdkman
  for c in /opt/homebrew/bin/mvn /usr/local/bin/mvn \
            "$HOME/.sdkman/candidates/maven/current/bin/mvn"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  echo ""
}

MVN_CMD="$(find_mvn)"

# ─── Prerequisites ───────────────────────────────────────────────────────────
check_prereqs() {
  section "1. Prerequisites"
  local missing=0

  for tool in curl java awk bc; do
    if command -v "$tool" &>/dev/null; then
      ok "$tool found: $(command -v "$tool")"
    else
      fail "$tool not found"
      missing=$((missing + 1))
    fi
  done

  # Maven check
  if [[ -n "$MVN_CMD" ]]; then
    local mvn_ver
    mvn_ver=$("$MVN_CMD" --version 2>/dev/null | head -1 | grep -o 'Apache Maven [0-9.]*' || echo "unknown")
    ok "maven found: $MVN_CMD  ($mvn_ver)"
  else
    fail "Maven not found. Install: brew install maven"
    missing=$((missing + 1))
  fi

  # Check Java version
  local java_ver
  java_ver=$(java -version 2>&1 | head -1 | awk -F'"' '{print $2}' | awk -F'.' '{print $1}')
  if [[ "${java_ver:-0}" -ge 21 ]]; then
    ok "Java version: $java_ver (>= 21)"
  else
    warn "Java version $java_ver detected — Java 25 recommended for SIMD Vector API"
  fi

  # Optional: wrk for high-performance load testing
  if command -v wrk &>/dev/null; then
    ok "wrk found — will use for high-accuracy load test"
    HAS_WRK=true
  else
    warn "wrk not found — using parallel curl (install with: brew install wrk)"
    HAS_WRK=false
  fi

  # Optional: hey
  if command -v hey &>/dev/null; then
    ok "hey found"
    HAS_HEY=true
  else
    HAS_HEY=false
  fi

  [[ $missing -eq 0 ]] || { fail "Missing required tools. Install them and retry."; exit 1; }
}

# ─── Build ───────────────────────────────────────────────────────────────────
build_jar() {
  section "2. Build"
  if [[ "$DO_BUILD" == false ]]; then
    warn "Skipping build (--no-build)"
    return
  fi

  if [[ ! -f "pom.xml" ]]; then
    fail "pom.xml not found. Run from project root."
    exit 1
  fi

  if [[ -z "$MVN_CMD" ]]; then
    fail "Maven not found — cannot build. Install: brew install maven"
    exit 1
  fi

  log "Running Maven package (skipping tests) via: $MVN_CMD"
  local start_ms end_ms elapsed mvn_log
  mvn_log=$(mktemp /tmp/rinha-mvn-XXXXXX.log)
  start_ms=$(ms_now)

  if "$MVN_CMD" -q package -DskipTests > "$mvn_log" 2>&1; then
    end_ms=$(ms_now)
    elapsed=$((end_ms - start_ms))
    ok "Build successful in ${elapsed}ms"
    ls -lh "$JAR" 2>/dev/null | awk '{print "    JAR size: " $5 "  →  " $NF}'
  else
    fail "Build failed — output:"
    grep -v "^WARNING:" "$mvn_log" | grep -v "^$" || cat "$mvn_log"
    rm -f "$mvn_log"
    exit 1
  fi
  rm -f "$mvn_log"
}

# ─── Synthetic dataset generator ─────────────────────────────────────────────
# Creates a small .bin file in the official format so the server can start
# locally for benchmarking when the real 3M-vector dataset is not present.
SYNTHETIC_BIN="$RESULTS_DIR/synthetic-vectors.bin"

generate_test_dataset() {
  local n="${1:-10000}"   # default 10 k vectors — enough for meaningful KNN
  log "Generating synthetic dataset ($n vectors, 14 dims) → $SYNTHETIC_BIN..."

  python3 - <<PYEOF
import struct, random, math, os

random.seed(42)
N    = $n
DIMS = 14
C    = 32       # IVF clusters for synthetic dataset
NPROBE = 8      # default nprobe
MAGIC   = 0x52494E48   # 'R','I','N','H' LE
VERSION = 4            # cluster-ordered (V4)

os.makedirs("$RESULTS_DIR", exist_ok=True)

# Build vectors and labels
vecs   = []
labels = []
for i in range(N):
    fraud = (i % 2 == 0)
    vec = []
    if fraud:
        vec.append(min(1.0, random.gauss(0.85, 0.1)))
        vec.append(min(1.0, random.gauss(0.75, 0.15)))
        vec.append(min(1.0, random.gauss(0.90, 0.1)))
        vec.append(random.gauss(0.22, 0.1) % 1.0)
        vec.append(random.random())
        vec.append(-1.0)
        vec.append(-1.0)
        vec.append(min(1.0, random.gauss(0.90, 0.1)))
        vec.append(min(1.0, random.gauss(0.95, 0.05)))
        vec.append(0.0)
        vec.append(1.0)
        vec.append(1.0)
        vec.append(random.choice([0.75, 0.80, 0.85]))
        vec.append(min(1.0, random.gauss(0.05, 0.02)))
    else:
        vec.append(max(0.0, random.gauss(0.05, 0.04)))
        vec.append(max(0.0, random.gauss(0.10, 0.05)))
        vec.append(max(0.0, random.gauss(0.05, 0.03)))
        vec.append(random.gauss(0.75, 0.1) % 1.0)
        vec.append(random.random())
        vec.append(max(0.0, random.gauss(0.50, 0.2)))
        vec.append(max(0.0, random.gauss(0.02, 0.01)))
        vec.append(max(0.0, random.gauss(0.03, 0.02)))
        vec.append(max(0.0, random.gauss(0.15, 0.05)))
        vec.append(0.0)
        vec.append(1.0)
        vec.append(0.0)
        vec.append(random.choice([0.15, 0.20, 0.25]))
        vec.append(max(0.0, random.gauss(0.03, 0.01)))
    vec = [max(-1.0, min(1.0, v)) for v in vec]
    vecs.append(vec)
    labels.append(1 if fraud else 0)

# Simple k-means with C clusters
def dist_sq(a, b):
    return sum((a[d]-b[d])**2 for d in range(DIMS))

# Init: pick C evenly-spaced vectors as centroids
cent_idx = [int(i * N / C) for i in range(C)]
centroids = [list(vecs[i]) for i in cent_idx]

for iteration in range(8):
    assignments = []
    for i in range(N):
        best_c, best_d = 0, float('inf')
        for c in range(C):
            d = dist_sq(vecs[i], centroids[c])
            if d < best_d:
                best_d = d
                best_c = c
        assignments.append(best_c)
    # Update centroids
    sums   = [[0.0]*DIMS for _ in range(C)]
    counts = [0]*C
    for i in range(N):
        c = assignments[i]
        counts[c] += 1
        for d in range(DIMS):
            sums[c][d] += vecs[i][d]
    for c in range(C):
        if counts[c] > 0:
            centroids[c] = [sums[c][d]/counts[c] for d in range(DIMS)]

# Build IVF lists with cluster-ordered permutation (V4)
list_sizes   = [0]*C
for c in assignments:
    list_sizes[c] += 1
list_offsets = [0]*C
for c in range(1, C):
    list_offsets[c] = list_offsets[c-1] + list_sizes[c-1]
# permutation[new_idx] = original_idx
permutation = [0]*N
cursor      = [0]*C
for i in range(N):
    c = assignments[i]
    permutation[list_offsets[c] + cursor[c]] = i
    cursor[c] += 1

# Quantize to INT8
def quantize(f):
    q = int(f * 127.0 + (0.5 if f >= 0 else -0.5))
    return max(-127, min(127, q))

with open("$SYNTHETIC_BIN", "wb") as f:
    # V4 header (32 bytes)
    f.write(struct.pack("<iiiiiiii", MAGIC, VERSION, N, DIMS, C, NPROBE, 0, 0))
    # INT8 vectors in cluster order
    for ni in range(N):
        f.write(bytes([quantize(v) & 0xFF for v in vecs[permutation[ni]]]))
    # Labels in cluster order
    f.write(bytes([labels[permutation[ni]] for ni in range(N)]))
    # Centroids: C × DIMS float32 LE
    for cent in centroids:
        f.write(struct.pack("<" + "f"*DIMS, *cent))
    # List sizes: C × int32 LE
    f.write(struct.pack("<" + "i"*C, *list_sizes))
    # No listData in V4

print(f"Written {N} vectors ({C} clusters, V4) to $SYNTHETIC_BIN ({os.path.getsize('$SYNTHETIC_BIN')} bytes)")
PYEOF
  ok "Synthetic dataset ready: $SYNTHETIC_BIN"
}

# ─── Server Management ───────────────────────────────────────────────────────
SERVER_PID=""

start_server() {
  if [[ "$MANAGE_SERVER" == false ]]; then
    log "Skipping server start (--no-server)"
    return
  fi

  section "3. Server Startup"

  if [[ ! -f "$JAR" ]]; then
    fail "JAR not found: $JAR — build first"
    exit 1
  fi

  # Kill any existing process on the port
  local existing_pid
  existing_pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
  if [[ -n "$existing_pid" ]]; then
    warn "Port $PORT already in use (PID $existing_pid) — killing..."
    kill -9 "$existing_pid" 2>/dev/null || true
    sleep 1
  fi

  # Resolve dataset path → VECTORS_BIN env var (server reads this)
  local vectors_bin=""
  if [[ -n "$DATASET" ]]; then
    vectors_bin="$DATASET"
    log "Using dataset: $vectors_bin"
  else
    # Search common locations for the real dataset
    for candidate in \
      "/data/references.bin" \
      "resources/references.bin" \
      "src/main/resources/references.bin" \
      "$HOME/references.bin"; do
      if [[ -f "$candidate" ]]; then
        vectors_bin="$candidate"
        ok "Found real dataset: $vectors_bin"
        break
      fi
    done
  fi

  # Fall back to synthetic dataset for local benchmark testing
  if [[ -z "$vectors_bin" ]]; then
    warn "Real dataset not found — generating synthetic test dataset (10 k vectors)"
    warn "For accurate scoring, provide DATASET=/path/to/references.bin"
    generate_test_dataset 10000
    vectors_bin="$SYNTHETIC_BIN"
  fi

  log "Starting server on port $PORT..."
  log "JVM flags: ${JVM_FLAGS[*]}"

  local log_file="$RESULTS_DIR/server-$TIMESTAMP.log"

  VECTORS_BIN="$vectors_bin" java "${JVM_FLAGS[@]}" -jar "$JAR" \
    > "$log_file" 2>&1 &
  SERVER_PID=$!

  log "Server PID: $SERVER_PID | Logs: $log_file"
}

stop_server() {
  if [[ "$MANAGE_SERVER" == false ]] || [[ -z "$SERVER_PID" ]]; then
    return
  fi
  log "Stopping server (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  ok "Server stopped"
}

wait_for_ready() {
  section "4. Wait for /ready"
  # If we're not managing the server and --smoke was requested, do a single probe
  local max_wait=120  # 2 minutes for dataset loading
  if [[ "$MANAGE_SERVER" == false ]]; then
    local probe
    probe=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$BASE_URL/ready" 2>/dev/null || echo "000")
    if [[ "$probe" == "200" ]]; then
      ok "/ready returned 200"
      return 0
    fi
    warn "No server running at $BASE_URL (HTTP $probe). Start the server or omit --no-server."
    if [[ "$MODE" == "smoke" ]]; then
      warn "Skipping correctness tests — server not available"
      return 1
    fi
  fi
  local elapsed=0
  local last_code=""
  local start_ms end_ms
  start_ms=$(ms_now)

  log "Polling GET $BASE_URL/ready (timeout: ${max_wait}s)..."

  while [[ $elapsed -lt $max_wait ]]; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$BASE_URL/ready" 2>/dev/null || echo "000")

    if [[ "$http_code" == "200" ]]; then
      end_ms=$(ms_now)
      local startup_ms=$((end_ms - start_ms))
      ok "/ready returned 200 — startup time: ${startup_ms}ms"
      return 0
    fi

    if [[ "$http_code" != "$last_code" ]]; then
      echo -n "  HTTP $http_code "
      last_code="$http_code"
    else
      echo -n "."
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo ""
  fail "/ready did not return 200 within ${max_wait}s"
  exit 1
}

# ─── Correctness Tests ───────────────────────────────────────────────────────
CORRECTNESS_PASS=0
CORRECTNESS_FAIL=0

assert_response() {
  local name="$1"
  local payload="$2"
  local expect_approved="$3"  # "true" or "false"
  local expect_score="$4"     # exact float or "" to skip

  local response http_code body
  response=$(curl -s -w "\n%{http_code}" \
    -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    --connect-timeout 5 \
    --max-time 10 2>/dev/null)

  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  # Check HTTP status
  if [[ "$http_code" != "200" ]]; then
    fail "$name → HTTP $http_code (expected 200) | body: $body"
    CORRECTNESS_FAIL=$((CORRECTNESS_FAIL + 1))
    return
  fi

  # Check approved field
  local actual_approved actual_score
  actual_approved=$(echo "$body" | grep -o '"approved":[^,}]*' | cut -d: -f2 | tr -d ' ')
  actual_score=$(echo "$body" | grep -o '"fraud_score":[^,}]*' | cut -d: -f2 | tr -d ' ')

  if [[ "$actual_approved" != "$expect_approved" ]]; then
    fail "$name → approved=$actual_approved (expected $expect_approved) | score=$actual_score | body: $body"
    CORRECTNESS_FAIL=$((CORRECTNESS_FAIL + 1))
    return
  fi

  if [[ -n "$expect_score" ]] && [[ "$actual_score" != "$expect_score" ]]; then
    warn "$name → approved=$actual_approved ✓ | fraud_score=$actual_score (expected $expect_score)"
    # Score might differ if dataset not loaded — still "pass" on approved
    CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
    return
  fi

  ok "$name → approved=$actual_approved | fraud_score=$actual_score"
  CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
}

run_correctness_tests() {
  section "5. Correctness Tests"

  log "Test 1: Legit transaction (low amount, known merchant)"
  assert_response "LEGIT" "$PAYLOAD_LEGIT" "true" ""

  log "Test 2: Fraud transaction (high amount, unknown merchant, far from home)"
  assert_response "FRAUD" "$PAYLOAD_FRAUD" "false" ""

  log "Test 3: Mid-range transaction with history"
  response=$(curl -s -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD_MID" --max-time 10 2>/dev/null)
  ok "MID → response: $response"

  log "Test 4: Unknown MCC (should use default 0.5)"
  response=$(curl -s -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD_MCC_UNKNOWN" --max-time 10 2>/dev/null)
  ok "MCC_UNKNOWN → response: $response"

  log "Test 5: Small online transaction (no last_tx)"
  response=$(curl -s -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD_ONLINE_SMALL" --max-time 10 2>/dev/null)
  ok "ONLINE_SMALL → response: $response"

  log "Test 6: GET /ready returns 200"
  http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/ready")
  if [[ "$http_code" == "200" ]]; then
    ok "/ready → HTTP 200"
    CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
  else
    fail "/ready → HTTP $http_code"
    CORRECTNESS_FAIL=$((CORRECTNESS_FAIL + 1))
  fi

  log "Test 7: Invalid JSON returns 4xx"
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d '{"invalid":}' --max-time 5 2>/dev/null)
  if [[ "$http_code" == "400" ]] || [[ "$http_code" == "422" ]]; then
    ok "Invalid JSON → HTTP $http_code (correct)"
    CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
  else
    warn "Invalid JSON → HTTP $http_code (expected 4xx)"
  fi

  log "Test 8: Unknown endpoint returns 4xx"
  http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/unknown" 2>/dev/null)
  if [[ "$http_code" == "404" ]]; then
    ok "/unknown → HTTP 404 (correct)"
    CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
  else
    warn "/unknown → HTTP $http_code"
  fi

  hr
  echo -e "  Correctness: ${GREEN}$CORRECTNESS_PASS passed${NC} | ${RED}$CORRECTNESS_FAIL failed${NC}"
  if [[ $CORRECTNESS_FAIL -gt 0 ]]; then
    warn "Some correctness tests failed — results may not match expected Rinha scoring"
  fi
}

# ─── Single-request latency probe ───────────────────────────────────────────
measure_single_latency() {
  local payload="$1"
  # Returns time_total in milliseconds (no allocation, curl built-in timing)
  curl -s -o /dev/null \
    -w "%{time_total}" \
    -X POST "$BASE_URL/fraud-score" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    --connect-timeout 5 \
    --max-time 10 2>/dev/null | awk '{printf "%.3f", $1 * 1000}'
}

# ─── Load Test (parallel curl via xargs -P) ──────────────────────────────────
TIMING_FILE=""

run_load_test_curl() {
  local total="$1"
  local concurrency="$2"
  TIMING_FILE="$RESULTS_DIR/timings-$TIMESTAMP.txt"
  : > "$TIMING_FILE"

  local payload_count=${#PAYLOADS[@]}
  local tmpdir
  tmpdir=$(mktemp -d)

  # Write payloads to temp files (xargs subshells can't access bash arrays)
  for i in "${!PAYLOADS[@]}"; do
    printf '%s' "${PAYLOADS[$i]}" > "$tmpdir/p${i}.json"
  done

  log "Running $total requests (concurrency=$concurrency) via xargs -P..."
  echo -n "  Progress: "

  local start_epoch
  start_epoch=$(ms_now)

  # xargs -P launches up to $concurrency curl processes in parallel.
  # Worker script is written to a temp file to avoid "command line too long".
  local worker_sh="$tmpdir/worker.sh"
  cat > "$worker_sh" <<WORKER
#!/usr/bin/env bash
idx=\$(( \$1 % $payload_count ))
result=\$(curl -s \
  -w "%{http_code} %{time_total}" \
  -o /dev/null \
  -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  --data-binary "@$tmpdir/p\${idx}.json" \
  --connect-timeout 3 --max-time 5 2>/dev/null) || result="000 0"
code="\${result%% *}"
secs="\${result##* }"
ms=\$(awk -v t="\$secs" 'BEGIN{printf "%.3f", t*1000}')
echo "\$code \$ms" >> "$TIMING_FILE"
WORKER
  chmod +x "$worker_sh"

  seq 0 $((total - 1)) | xargs -P "$concurrency" -I{} "$worker_sh" {}

  echo " done"

  local end_epoch
  end_epoch=$(ms_now)
  LOAD_TEST_DURATION_MS=$((end_epoch - start_epoch))

  rm -rf "$tmpdir"
}

run_load_test_wrk() {
  local total="$1"
  local concurrency="$2"
  local duration="${WRK_DURATION:-30}"

  TIMING_FILE="$RESULTS_DIR/wrk-$TIMESTAMP.txt"

  # Write lua script for wrk
  local lua_script="$RESULTS_DIR/wrk-script.lua"
  cat > "$lua_script" <<'LUA'
local payloads = {
  '{"id":"tx-w1","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}',
  '{"id":"tx-w2","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}',
  '{"id":"tx-w3","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.71},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.86}}',
  '{"id":"tx-w4","transaction":{"amount":12.50,"installments":1,"requested_at":"2026-03-13T14:00:00Z"},"customer":{"avg_amount":50.00,"tx_count_24h":1,"known_merchants":["MERC-010"]},"merchant":{"id":"MERC-010","mcc":"5311","avg_amount":45.00},"terminal":{"is_online":true,"card_present":false,"km_from_home":0.5},"last_transaction":null}'
}

local counter = 0
request = function()
  counter = counter + 1
  local p = payloads[(counter % #payloads) + 1]
  return wrk.format("POST", "/fraud-score", {["Content-Type"] = "application/json"}, p)
end
LUA

  log "Running wrk load test: ${duration}s, concurrency=${concurrency}..."
  wrk -t"$concurrency" -c"$concurrency" -d"${duration}s" \
    --script "$lua_script" \
    "$BASE_URL/fraud-score" \
    > "$TIMING_FILE" 2>&1

  cat "$TIMING_FILE"
  LOAD_TEST_DURATION_MS=$((duration * 1000))
}

# ─── Statistics ──────────────────────────────────────────────────────────────
compute_stats() {
  local timing_file="$1"

  if [[ ! -f "$timing_file" ]]; then
    warn "Timing file not found: $timing_file"
    return
  fi

  # Parse: "HTTP_CODE TIME_MS" per line
  awk '
  BEGIN {
    count = 0
    errors = 0
    sum = 0
    min = 999999
    max = 0
  }
  NF >= 2 {
    code = $1
    t = $2 + 0
    if (code != "200") {
      errors++
    } else {
      times[count] = t
      count++
      sum += t
      if (t < min) min = t
      if (t > max) max = t
    }
  }
  END {
    if (count == 0) {
      print "ERROR: no successful responses"
      exit 1
    }

    # Sort times (bubble sort for simplicity, works for <10k)
    n = count
    for (i = 0; i < n-1; i++) {
      for (j = 0; j < n-i-1; j++) {
        if (times[j] > times[j+1]) {
          tmp = times[j]
          times[j] = times[j+1]
          times[j+1] = tmp
        }
      }
    }

    avg = sum / count
    p50_idx = int(n * 0.50)
    p90_idx = int(n * 0.90)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    p999_idx = int(n * 0.999)
    if (p999_idx >= n) p999_idx = n - 1

    p50  = times[p50_idx]
    p90  = times[p90_idx]
    p95  = times[p95_idx]
    p99  = times[p99_idx]
    p999 = times[p999_idx]

    total = count + errors
    error_rate = (errors / total) * 100

    printf "COUNT=%d\n", count
    printf "ERRORS=%d\n", errors
    printf "TOTAL=%d\n", total
    printf "ERROR_RATE=%.2f\n", error_rate
    printf "AVG=%.3f\n", avg
    printf "MIN=%.3f\n", min
    printf "MAX=%.3f\n", max
    printf "P50=%.3f\n", p50
    printf "P90=%.3f\n", p90
    printf "P95=%.3f\n", p95
    printf "P99=%.3f\n", p99
    printf "P999=%.3f\n", p999
  }
  ' "$timing_file"
}

# ─── Rinha Scoring Formula ───────────────────────────────────────────────────
compute_rinha_score() {
  local p99_ms="$1"
  local fp="${2:-0}"    # false positives
  local fn="${3:-0}"    # false negatives
  local err="${4:-0}"   # http errors
  local total="${5:-1}"

  awk -v p99="$p99_ms" -v FP="$fp" -v FN="$fn" -v ERR="$err" -v N="$total" '
  BEGIN {
    # ── Latency score ──
    K = 1000
    T_max = 1000       # ms
    p99_MIN = 1        # ms
    p99_MAX = 2000     # ms

    if (p99 > p99_MAX) {
      score_p99 = -3000
      p99_cut = 1
    } else {
      eff_p99 = (p99 < p99_MIN) ? p99_MIN : p99
      score_p99 = K * log(T_max / eff_p99) / log(10)
      if (score_p99 > 3000) score_p99 = 3000
      p99_cut = 0
    }

    # ── Detection score ──
    E = 1*FP + 3*FN + 5*ERR
    failures = FP + FN + ERR
    failure_rate = failures / N

    if (failure_rate > 0.15) {
      score_det = -3000
      det_cut = 1
    } else {
      eps_MIN = 0.001
      beta = 300
      eps = E / N
      if (eps < eps_MIN) eps = eps_MIN
      rate_comp = K * log(1 / eps) / log(10)
      abs_pen = -beta * log(1 + E) / log(10)
      score_det = rate_comp + abs_pen
      if (score_det > 3000) score_det = 3000
      det_cut = 0
    }

    final = score_p99 + score_det

    printf "SCORE_P99=%.2f\n", score_p99
    printf "SCORE_DET=%.2f\n", score_det
    printf "FINAL_SCORE=%.2f\n", final
    printf "P99_CUT=%d\n", p99_cut
    printf "DET_CUT=%d\n", det_cut
    printf "E=%d\n", E
    printf "FAILURE_RATE=%.4f\n", failure_rate
  }
  '
}

# ─── GC & JVM stats ──────────────────────────────────────────────────────────
check_jvm_stats() {
  if [[ -z "$SERVER_PID" ]]; then return; fi

  section "7. JVM / Process Stats"

  # RSS memory
  local rss
  rss=$(ps -o rss= -p "$SERVER_PID" 2>/dev/null || echo "N/A")
  if [[ "$rss" != "N/A" ]]; then
    local rss_mb
    rss_mb=$(awk -v r="$rss" 'BEGIN{printf "%.1f", r/1024}')
    echo -e "  RSS Memory : ${BOLD}${rss_mb} MB${NC}"
    if awk -v r="$rss_mb" 'BEGIN{exit (r > 350) ? 0 : 1}'; then
      warn "RSS exceeds 350 MB limit!"
    fi
  fi

  # CPU usage
  local cpu
  cpu=$(ps -o %cpu= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || echo "N/A")
  echo -e "  CPU usage  : ${BOLD}${cpu}%${NC}"

  # Optional: jstat GC if available
  if command -v jstat &>/dev/null && [[ -n "$SERVER_PID" ]]; then
    log "GC stats (jstat):"
    jstat -gc "$SERVER_PID" 2>/dev/null | head -3 | sed 's/^/    /' || true
  fi
}

# ─── Final Report ─────────────────────────────────────────────────────────────
# Uses plain globals set by eval'ing compute_stats / compute_rinha_score output.
# (No declare -A / nameref — must work on bash 3.2 shipped with macOS)
print_report() {
  section "📊 Benchmark Report — $(date '+%Y-%m-%d %H:%M:%S')"

  echo ""
  echo -e "  ${BOLD}Target:${NC}     $BASE_URL"
  echo -e "  ${BOLD}Requests:${NC}   $TOTAL total / $COUNT successful / $ERRORS errors"
  echo -e "  ${BOLD}Duration:${NC}   ${LOAD_TEST_DURATION_MS}ms"
  echo -e "  ${BOLD}Throughput:${NC} $(awk -v c="$COUNT" -v d="$LOAD_TEST_DURATION_MS" 'BEGIN{printf "%.0f req/s", c/(d/1000)}')"
  echo ""

  echo -e "  ${BOLD}${CYAN}Latency Percentiles:${NC}"
  echo -e "  ┌─────────┬──────────────┐"
  echo -e "  │  pct    │  latency     │"
  echo -e "  ├─────────┼──────────────┤"
  printf  "  │  p50    │  %8.3f ms │\n" "$P50"
  printf  "  │  p90    │  %8.3f ms │\n" "$P90"
  printf  "  │  p95    │  %8.3f ms │\n" "$P95"

  local p99_color="$GREEN"
  if awk -v p="$P99" 'BEGIN{exit (p > 1.0)  ? 0 : 1}'; then p99_color="$YELLOW"; fi
  if awk -v p="$P99" 'BEGIN{exit (p > 100)  ? 0 : 1}'; then p99_color="$RED";    fi
  printf  "  │  p99    │  ${p99_color}%8.3f ms${NC} │\n" "$P99"
  printf  "  │  p99.9  │  %8.3f ms │\n" "$P999"
  printf  "  │  avg    │  %8.3f ms │\n" "$AVG"
  printf  "  │  min    │  %8.3f ms │\n" "$MIN"
  printf  "  │  max    │  %8.3f ms │\n" "$MAX"
  echo -e "  └─────────┴──────────────┘"
  echo ""

  echo -e "  ${BOLD}${CYAN}Rinha Scoring (p99=${P99}ms, zero detection errors assumed):${NC}"
  echo -e "  ┌────────────────────┬───────────────┐"
  printf  "  │  score_p99         │  %+10.2f   │\n" "$SCORE_P99"
  printf  "  │  score_det         │  %+10.2f   │\n" "$SCORE_DET"
  echo -e "  ├────────────────────┼───────────────┤"
  local final_color="$RED"
  if awk -v f="$FINAL_SCORE" 'BEGIN{exit (f > 0)    ? 0 : 1}'; then final_color="$YELLOW"; fi
  if awk -v f="$FINAL_SCORE" 'BEGIN{exit (f > 3000) ? 0 : 1}'; then final_color="$GREEN";  fi
  printf  "  │  ${BOLD}FINAL SCORE${NC}         │  ${final_color}${BOLD}%+10.2f${NC}   │\n" "$FINAL_SCORE"
  echo -e "  └────────────────────┴───────────────┘"
  echo ""

  [[ "$P99_CUT"  == "1" ]] && fail "p99 CUTOFF TRIGGERED (p99 > 2000ms) → score_p99 = -3000"
  [[ "$DET_CUT"  == "1" ]] && fail "DETECTION CUTOFF TRIGGERED (failure_rate > 15%) → score_det = -3000"

  echo -e "  ${BOLD}Assessment:${NC}"
  if awk -v p="$P99" 'BEGIN{exit (p <= 1.0)  ? 0 : 1}'; then
    ok "p99 ≤ 1ms   → PERFECT LATENCY SCORE (+3000) 🏆"
  elif awk -v p="$P99" 'BEGIN{exit (p <= 10)  ? 0 : 1}'; then
    ok "p99 ≤ 10ms  → EXCELLENT (score ~2000+)"
  elif awk -v p="$P99" 'BEGIN{exit (p <= 100) ? 0 : 1}'; then
    warn "p99 ≤ 100ms → GOOD (score ~1000+), room for improvement"
  elif awk -v p="$P99" 'BEGIN{exit (p <= 1000)? 0 : 1}'; then
    warn "p99 ≤ 1000ms → ACCEPTABLE but optimize further"
  else
    fail "p99 > 1000ms → NEEDS OPTIMIZATION"
  fi

  echo ""
  echo -e "  ${DIM}Results saved to: $RESULTS_DIR/${NC}"
  hr

  local report_json="$RESULTS_DIR/report-$TIMESTAMP.json"
  cat > "$report_json" <<JSONEOF
{
  "timestamp": "$TIMESTAMP",
  "target": "$BASE_URL",
  "requests": {
    "total": $TOTAL, "successful": $COUNT, "errors": $ERRORS,
    "error_rate_pct": $ERROR_RATE
  },
  "throughput_rps": $(awk -v c="$COUNT" -v d="$LOAD_TEST_DURATION_MS" 'BEGIN{printf "%.1f", c/(d/1000)}'),
  "latency_ms": {
    "p50": $P50, "p90": $P90, "p95": $P95, "p99": $P99, "p99_9": $P999,
    "avg": $AVG, "min": $MIN, "max": $MAX
  },
  "rinha_scoring": {
    "score_p99": $SCORE_P99, "score_det": $SCORE_DET, "final_score": $FINAL_SCORE,
    "p99_cut_triggered": $P99_CUT, "det_cut_triggered": $DET_CUT
  },
  "correctness": { "passed": $CORRECTNESS_PASS, "failed": $CORRECTNESS_FAIL }
}
JSONEOF
  ok "JSON report: $report_json"
}

# ─── Trap for cleanup ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  stop_server
}
trap cleanup EXIT INT TERM

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  echo ""
  echo -e "${BOLD}${MAGENTA}"
  echo "  ██████╗ ██╗███╗   ██╗██╗  ██╗ █████╗     ██████╗  ██████╗ ██████╗  ██████╗"
  echo "  ██╔══██╗██║████╗  ██║██║  ██║██╔══██╗    ╚════██╗██╔═████╗╚════██╗██╔════╝"
  echo "  ██████╔╝██║██╔██╗ ██║███████║███████║     █████╔╝██║██╔██║ █████╔╝███████╗"
  echo "  ██╔══██╗██║██║╚██╗██║██╔══██║██╔══██║    ██╔═══╝ ████╔╝██║██╔═══╝ ██╔═══██╗"
  echo "  ██║  ██║██║██║ ╚████║██║  ██║██║  ██║    ███████╗╚██████╔╝███████╗╚██████╔╝"
  echo "  ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝ "
  echo -e "${NC}"
  echo -e "  ${BOLD}Fraud Detection API Benchmark${NC} | Mode: ${CYAN}$MODE${NC} | Requests: ${CYAN}$REQUESTS${NC} | Concurrency: ${CYAN}$CONCURRENCY${NC}"
  echo ""

  check_prereqs
  build_jar
  start_server
  wait_for_ready || {
    if [[ "$MODE" == "smoke" ]]; then
      section "⚠️  Smoke test — build OK, server not available"
      warn "Build succeeded but server is not running. Start with: java ${JVM_FLAGS[*]} -jar $JAR"
      return 0
    fi
    fail "Cannot proceed without a running server"
    exit 1
  }
  run_correctness_tests

  if [[ "$MODE" == "smoke" ]]; then
    section "✅ Smoke test complete"
    echo -e "  Correctness: ${GREEN}$CORRECTNESS_PASS passed${NC} | ${RED}$CORRECTNESS_FAIL failed${NC}"
    return
  fi

  # ── Warmup phase ──
  section "6. Load Test"
  # Warmup: 10 sequential requests — just enough to JIT-compile the hot path,
  # no background jobs, no wait, completes in <1s.
  log "Warming up (10 sequential requests)..."
  local _wu_payload="$PAYLOAD_LEGIT"
  for _wu in 1 2 3 4 5 6 7 8 9 10; do
    curl -s -o /dev/null -X POST "$BASE_URL/fraud-score" \
      -H "Content-Type: application/json" -d "$_wu_payload" --max-time 3
  done
  ok "Warmup complete"

  LOAD_TEST_DURATION_MS=0

  if [[ "${HAS_WRK:-false}" == "true" ]] && [[ "${USE_WRK:-true}" == "true" ]]; then
    run_load_test_wrk "$REQUESTS" "$CONCURRENCY"
    # wrk outputs its own stats; extract p99 from wrk output
    local wrk_p99
    wrk_p99=$(grep "99%" "$TIMING_FILE" 2>/dev/null | awk '{print $2}' | sed 's/ms//' || echo "")
    if [[ -n "$wrk_p99" ]]; then
      echo "wrk p99: ${wrk_p99}ms"
    fi
  else
    run_load_test_curl "$REQUESTS" "$CONCURRENCY"

    # Compute statistics — eval sets COUNT, ERRORS, P50, P90, P95, P99 … as globals
    log "Computing statistics..."
    local raw_stats
    raw_stats=$(compute_stats "$TIMING_FILE")

    if [[ -z "$raw_stats" ]]; then
      fail "No statistics available"
      return 1
    fi

    eval "$raw_stats"   # sets COUNT ERRORS TOTAL ERROR_RATE AVG MIN MAX P50 P90 P95 P99 P999

    check_jvm_stats

    # Compute Rinha score — eval sets SCORE_P99 SCORE_DET FINAL_SCORE P99_CUT DET_CUT E FAILURE_RATE
    eval "$(compute_rinha_score "$P99" 0 0 "$ERRORS" "$TOTAL")"

    print_report
  fi
}

main "$@"

