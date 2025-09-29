#!/usr/bin/env bash

# askme Reranking Script
# Usage: ./rerank.sh --model=qwen_local --take=8

set -euo pipefail

MODEL="qwen_local"   # qwen_local | bge_local
TAKE=8
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
VERBOSE=false

usage() {
  cat << EOF
askme Reranking Script

USAGE:
  $0 [OPTIONS]

OPTIONS:
  --model=NAME     Reranker: qwen_local | bge_local (default: qwen_local)
  --take=N         Number of passages to keep after rerank (default: 8)
  --api-url=URL    API base URL (default: http://localhost:8080)
  --verbose        Verbose output
  --help           Show this help

NOTE:
  This script demonstrates reranking by invoking /query/retrieve first and
  then reranking is performed server-side via /query.
EOF
}

log(){ echo "[rerank] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h) usage; exit 0 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --take=*) TAKE="${1#*=}"; shift ;;
    --api-url=*) API_BASE_URL="${1#*=}"; shift ;;
    --verbose|-v) VERBOSE=true; shift ;;
    *) log "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if $VERBOSE; then
  log "Model=$MODEL take=$TAKE api=$API_BASE_URL"
fi

# Minimal smoke test by submitting a query to /query with reranker selection
REQ='{"q":"rerank smoke test","topK":16,"alpha":0.5,"use_rrf":true,"reranker":"'"$MODEL"'","max_passages":'"$TAKE"'}'
curl -s -X POST -H 'Content-Type: application/json' \
  -d "$REQ" "$API_BASE_URL/query/" | jq .
