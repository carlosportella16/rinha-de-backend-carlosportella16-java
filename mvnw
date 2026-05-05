com#!/usr/bin/env bash
# Minimal Maven Wrapper — finds Maven from ~/.m2/wrapper/dists or PATH
set -euo pipefail

find_mvn() {
  # 1. System PATH
  if command -v mvn &>/dev/null; then
    echo "mvn"
    return
  fi
  # 2. ~/.m2/wrapper/dists — pick the newest mvn binary
  local mvn_bin
  mvn_bin=$(find "$HOME/.m2/wrapper/dists" -name "mvn" -type f 2>/dev/null \
    | sort -t/ -k8 -r | head -1)
  if [[ -n "$mvn_bin" ]]; then
    echo "$mvn_bin"
    return
  fi
  # 3. Common brew/sdkman locations
  for candidate in \
    /opt/homebrew/bin/mvn \
    /usr/local/bin/mvn \
    "$HOME/.sdkman/candidates/maven/current/bin/mvn"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

MVN=$(find_mvn)
if [[ -z "$MVN" ]]; then
  echo "ERROR: Maven not found. Install via: brew install maven" >&2
  exit 1
fi

exec "$MVN" "$@"

