#!/usr/bin/env bash

# askme Evaluation Grid Runner
# Sweeps alpha/topk/topn over /eval/run and collects results.

set -euo pipefail

# Defaults (aligned with current recommendations)
ALPHAS=${ALPHAS:-"0.0,0.25,0.5"}
TOPKS=${TOPKS:-"30,50,80"}
TOPNS=${TOPNS:-"5,8,10"}
SUITE=${SUITE:-"baseline"}
METRICS=${METRICS:-"answer_relevancy,context_precision"}
SAMPLE_SIZE=${SAMPLE_SIZE:-""}  # empty = all
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
OUTDIR=${OUTDIR:-"data/eval_runs"}
OUTFILE=${OUTFILE:-"grid_$(date +%Y%m%d_%H%M%S).jsonl"}
VERBOSE=${VERBOSE:-"0"}

usage() {
  cat << EOF
askme Evaluation Grid Runner

USAGE:
  $0 [--alphas=a,b,c] [--topks=... ] [--topns=... ] [--suite=baseline] [--metrics=list] [--sample-size=N] [--outdir=dir]

ENV:
  ALPHAS, TOPKS, TOPNS, SUITE, METRICS, SAMPLE_SIZE, OUTDIR, OUTFILE, ASKME_API_URL

Requires: bash, curl, jq
EOF
}

log() {
  echo "[grid] $*" >&2
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --alphas=*) ALPHAS="${1#*=}"; shift ;;
    --topks=*) TOPKS="${1#*=}"; shift ;;
    --topns=*) TOPNS="${1#*=}"; shift ;;
    --suite=*) SUITE="${1#*=}"; shift ;;
    --metrics=*) METRICS="${1#*=}"; shift ;;
    --sample-size=*) SAMPLE_SIZE="${1#*=}"; shift ;;
    --outdir=*) OUTDIR="${1#*=}"; shift ;;
    --outfile=*) OUTFILE="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"
OUTPATH="$OUTDIR/$OUTFILE"

# Health check
if ! curl -s "$API_BASE_URL/health" >/dev/null; then
  log "API not reachable at $API_BASE_URL. Start the server first."
  exit 1
fi

IFS=',' read -ra ASET <<< "$ALPHAS"
IFS=',' read -ra KSET <<< "$TOPKS"
IFS=',' read -ra NSET <<< "$TOPNS"

log "Writing results to $OUTPATH"

for a in "${ASET[@]}"; do
  for k in "${KSET[@]}"; do
    for n in "${NSET[@]}"; do
      # Run evaluation in JSON mode to parse run_id and metrics
      if [[ -n "$SAMPLE_SIZE" ]]; then SSFLAG=(--sample-size="$SAMPLE_SIZE"); else SSFLAG=(); fi
      JSON=$(ASKME_API_URL="$API_BASE_URL" bash scripts/evaluate.sh \
        --suite="$SUITE" --metrics="$METRICS" "${SSFLAG[@]}" \
        --alpha="$a" --topk="$k" --topn="$n" --format=json)

      RUN_ID=$(echo "$JSON" | jq -r '.run_id // empty')
      if [[ -z "$RUN_ID" ]]; then
        log "Failed run for a=$a k=$k n=$n"; continue
      fi

      # Extract overall metrics as a map
      METRICS_JSON=$(echo "$JSON" | jq '[.overall_metrics[]] | map({(.name): .value}) | add')
      SUMMARY=$(echo "$JSON" | jq '.summary')

      # Emit a JSON line with parameters and scores
      echo "{\"run_id\":\"$RUN_ID\",\"alpha\":$a,\"topk\":$k,\"topn\":$n,\"metrics\":$METRICS_JSON,\"summary\":$SUMMARY}" >> "$OUTPATH"

      [[ "$VERBOSE" == "1" ]] && log "Done a=$a k=$k n=$n → $RUN_ID"
    done
  done
done

log "Grid complete. Results: $OUTPATH"
