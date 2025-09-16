#!/usr/bin/env bash
# AskMe Query Test Script
# Usage: ./query_test.sh "your question here"

set -euo pipefail

# Positional arg 1 = question; env overrides for others
QUESTION="${1:-韩立是谁？}"
TOPK="${TOPK:-5}"
MAX_PASSAGES="${MAX_PASSAGES:-3}"
INCLUDE_DEBUG="${INCLUDE_DEBUG:-true}"
ENDPOINT="${ASKME_ENDPOINT:-http://localhost:8080}"

# Requirements check
command -v curl >/dev/null || { echo "curl 未安装"; exit 1; }
command -v jq >/dev/null || { echo "jq 未安装"; exit 1; }

echo "🔍 查询: $QUESTION"
echo "⏰ 开始时间: $(date '+%H:%M:%S')"
echo ""

# Build JSON safely (handles quotes and unicode)
PAYLOAD=$(jq -n \
  --arg q "$QUESTION" \
  --argjson topk "$TOPK" \
  --argjson max_passages "$MAX_PASSAGES" \
  --argjson include_debug "$INCLUDE_DEBUG" \
  '{q:$q, topk:$topk, max_passages:$max_passages, include_debug:$include_debug}')

TMP_RESP=$(mktemp)
HTTP_CODE=$(curl -sS -o "$TMP_RESP" -w "%{http_code}" \
  -X POST "$ENDPOINT/query/" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" || true)

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "❌ 请求失败 (HTTP $HTTP_CODE):"
  cat "$TMP_RESP"
  rm -f "$TMP_RESP"
  exit 1
fi

jq -r '
  "📝 答案:",
  "=" * 50,
  (.answer // "无答案"),
  "",
  "📚 引用来源:",
  "-" * 30,
  ((.citations // [])[] | "• [\(.doc_id)] \(.content[:100])..."),
  "",
  "⚡ 性能信息:",
  "-" * 30,
  ("检索延迟: " + ((.retrieval_debug.latency_ms // "N/A") | tostring) + "ms"),
  ("嵌入延迟: " + ((.retrieval_debug.embedding_latency_ms // "N/A") | tostring) + "ms"),
  ("搜索延迟: " + ((.retrieval_debug.search_latency_ms // "N/A") | tostring) + "ms"),
  ("重排序延迟: " + ((.retrieval_debug.rerank_latency_ms // "N/A") | tostring) + "ms"),
  ("RRF_k: " + ((.retrieval_debug.rrf_k // "N/A") | tostring)),
  ("融合方式: " + ((.retrieval_debug.fusion_method // "N/A") | tostring)),
  "",
  ("查询ID: \(.query_id)"),
  ("时间戳: \(.timestamp)")
' "$TMP_RESP"

rm -f "$TMP_RESP"

echo ""
echo "✅ 完成时间: $(date '+%H:%M:%S')"
