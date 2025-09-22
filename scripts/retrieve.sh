#!/usr/bin/env bash

# askme Document Retrieval Script
# Usage: ./retrieve.sh "<query>" [options]

set -euo pipefail

# Default values
QUERY=""
TOPK=30
ALPHA=0.5
USE_RRF=true
RRF_K=60
USE_HYDE=false
USE_RAG_FUSION=false
RERANKER="bge_local"
MAX_PASSAGES=5
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
OUTPUT_FORMAT="json"
INCLUDE_DEBUG=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
askme Document Retrieval Script

USAGE:
    $0 "<query>" [OPTIONS]

ARGUMENTS:
    <query>          Search query (required)

OPTIONS:
    --topk=N         Number of candidates to retrieve (default: 50)
    --alpha=X        Hybrid search alpha (0=sparse, 1=dense, default: 0.5)
    --rrf            Use reciprocal rank fusion (default: true)
    --no-rrf         Disable RRF, use alpha fusion
    --rrf-k=N        RRF k parameter (default: 60)
    --hyde           Enable HyDE query expansion
    --rag-fusion     Enable RAG-Fusion multi-query
    --reranker=TYPE  Reranker model: bge_local, cohere (default: bge_local)
    --max-passages=N Max passages for generation (default: 8)
    --debug          Include debug information in output
    --format=FORMAT  Output format: json, text, table (default: json)
    --api-url=URL    API base URL (default: http://localhost:8080)
    --verbose        Enable verbose output
    --help          Show this help message

EXAMPLES:
    # Basic hybrid search
    $0 "What is machine learning?" --alpha=0.5

    # Sparse-focused search with RRF
    $0 "neural networks deep learning" --alpha=0.2 --rrf --rrf-k=60

    # Dense search with query enhancement
    $0 "explain transformers" --alpha=0.8 --hyde --rag-fusion

    # Use Cohere reranker
    $0 "best practices for RAG" --reranker=cohere --debug

ENVIRONMENT VARIABLES:
    ASKME_API_URL    API base URL (default: http://localhost:8080)
    ASKME_API_KEY    API key for authentication (if required)

EOF
}

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - ${message}" >&2
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}" >&2
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}" >&2
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" >&2
            ;;
    esac
}

validate_parameters() {
    # Validate alpha parameter
    if (( $(echo "$ALPHA < 0 || $ALPHA > 1" | bc -l) )); then
        log ERROR "Alpha must be between 0 and 1, got: $ALPHA"
        exit 1
    fi

    # Validate topk
    if [[ $TOPK -lt 1 || $TOPK -gt 100 ]]; then
        log ERROR "topk must be between 1 and 100, got: $TOPK"
        exit 1
    fi

    # Validate max_passages
    if [[ $MAX_PASSAGES -lt 1 || $MAX_PASSAGES -gt 20 ]]; then
        log ERROR "max-passages must be between 1 and 20, got: $MAX_PASSAGES"
        exit 1
    fi

    # Validate reranker
    if [[ "$RERANKER" != "bge_local" && "$RERANKER" != "cohere" ]]; then
        log ERROR "Invalid reranker: $RERANKER. Must be 'bge_local' or 'cohere'"
        exit 1
    fi

    # Validate output format
    if [[ "$OUTPUT_FORMAT" != "json" && "$OUTPUT_FORMAT" != "text" && "$OUTPUT_FORMAT" != "table" ]]; then
        log ERROR "Invalid output format: $OUTPUT_FORMAT. Must be 'json', 'text', or 'table'"
        exit 1
    fi
}

create_retrieval_request() {
    cat << EOF
{
    "q": "$QUERY",
    "topk": $TOPK,
    "alpha": $ALPHA,
    "use_rrf": $USE_RRF,
    "rrf_k": $RRF_K,
    "use_hyde": $USE_HYDE,
    "use_rag_fusion": $USE_RAG_FUSION,
    "reranker": "$RERANKER",
    "max_passages": $MAX_PASSAGES,
    "include_debug": $INCLUDE_DEBUG
}
EOF
}

submit_retrieval_request() {
    local request_body="$1"
    local endpoint="${2:-retrieve}"

    local curl_args=(
        -s
        -X POST
        -H "Content-Type: application/json"
        -d "$request_body"
        "${API_BASE_URL}/query/${endpoint}"
    )

    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi

    if $VERBOSE; then
        log INFO "Submitting retrieval request to /${endpoint}..."
        echo "Request body:" >&2
        echo "$request_body" | jq . >&2
    fi

    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to submit retrieval request"
        exit 1
    fi

    # Check for API errors
    local error_message
    error_message=$(echo "$response" | jq -r '.detail // empty' 2>/dev/null || echo "")
    if [[ -n "$error_message" ]]; then
        log ERROR "API Error: $error_message"
        exit 1
    fi

    echo "$response"
}

format_output() {
    local response="$1"
    local format="$2"

    case $format in
        "json")
            echo "$response" | jq .
            ;;
        "text")
            format_text_output "$response"
            ;;
        "table")
            format_table_output "$response"
            ;;
    esac
}

format_text_output() {
    local response="$1"

    # Extract basic info
    local query_id
    query_id=$(echo "$response" | jq -r '.query_id // "N/A"')
    local timestamp
    timestamp=$(echo "$response" | jq -r '.timestamp // "N/A"')

    echo "Query ID: $query_id"
    echo "Timestamp: $timestamp"
    echo "Query: \"$QUERY\""
    echo

    # Check if this is a full query response or retrieval-only
    local has_answer
    has_answer=$(echo "$response" | jq 'has("answer")')

    if [[ "$has_answer" == "true" ]]; then
        # Full query response with generated answer
        local answer
        answer=$(echo "$response" | jq -r '.answer // "No answer generated"')
        echo "Generated Answer:"
        echo "=================="
        echo "$answer"
        echo
    fi

    # Display retrieved documents/citations
    echo "Retrieved Documents:"
    echo "==================="

    local documents
    if [[ "$has_answer" == "true" ]]; then
        documents=$(echo "$response" | jq -c '.citations[]? // empty')
    else
        documents=$(echo "$response" | jq -c '.documents[]? // empty')
    fi

    local count=1
    while IFS= read -r doc; do
        if [[ -n "$doc" ]]; then
            local doc_id title content score
            doc_id=$(echo "$doc" | jq -r '.doc_id // "N/A"')
            title=$(echo "$doc" | jq -r '.title // "Untitled"')
            content=$(echo "$doc" | jq -r '.content // "No content"')
            score=$(echo "$doc" | jq -r '.score // 0')

            echo "[$count] Doc ID: $doc_id"
            echo "    Title: $title"
            echo "    Score: $score"
            echo "    Content: $(echo "$content" | cut -c1-200)..."
            echo
            ((count++))
        fi
    done <<< "$documents"

    # Display debug info if available
    if $INCLUDE_DEBUG; then
        local debug_info
        debug_info=$(echo "$response" | jq '.retrieval_debug // empty')
        if [[ "$debug_info" != "null" && -n "$debug_info" ]]; then
            echo "Debug Information:"
            echo "=================="
            echo "$debug_info" | jq .
        fi
    fi
}

format_table_output() {
    local response="$1"

    # Use column command for table formatting if available
    if ! command -v column &> /dev/null; then
        log WARN "column command not available, falling back to text format"
        format_text_output "$response"
        return
    fi

    echo "Query: \"$QUERY\""
    echo

    # Create table header
    printf "%-3s %-15s %-30s %-8s %-50s\n" "Idx" "Doc ID" "Title" "Score" "Content Preview"
    printf "%-3s %-15s %-30s %-8s %-50s\n" "---" "---------------" "------------------------------" "--------" "--------------------------------------------------"

    # Display documents in table format
    local documents
    local has_answer
    has_answer=$(echo "$response" | jq 'has("answer")')

    if [[ "$has_answer" == "true" ]]; then
        documents=$(echo "$response" | jq -c '.citations[]? // empty')
    else
        documents=$(echo "$response" | jq -c '.documents[]? // empty')
    fi

    local count=1
    while IFS= read -r doc; do
        if [[ -n "$doc" ]]; then
            local doc_id title content score
            doc_id=$(echo "$doc" | jq -r '.doc_id // "N/A"')
            title=$(echo "$doc" | jq -r '.title // "Untitled"')
            content=$(echo "$doc" | jq -r '.content // ""')
            score=$(echo "$doc" | jq -r '.score // 0')

            # Truncate for table display
            doc_id=$(echo "$doc_id" | cut -c1-13)
            title=$(echo "$title" | cut -c1-28)
            content=$(echo "$content" | tr '\n' ' ' | cut -c1-48)

            printf "%-3d %-15s %-30s %-8.3f %-50s\n" "$count" "$doc_id" "$title" "$score" "$content"
            ((count++))
        fi
    done <<< "$documents"
}

main() {
    # Check dependencies
    if ! command -v curl &> /dev/null; then
        log ERROR "curl is required but not installed"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log ERROR "jq is required but not installed"
        exit 1
    fi

    if ! command -v bc &> /dev/null; then
        log ERROR "bc is required but not installed"
        exit 1
    fi

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                usage
                exit 0
                ;;
            --topk=*)
                TOPK="${1#*=}"
                shift
                ;;
            --alpha=*)
                ALPHA="${1#*=}"
                shift
                ;;
            --rrf)
                USE_RRF=true
                shift
                ;;
            --no-rrf)
                USE_RRF=false
                shift
                ;;
            --rrf-k=*)
                RRF_K="${1#*=}"
                shift
                ;;
            --hyde)
                USE_HYDE=true
                shift
                ;;
            --rag-fusion)
                USE_RAG_FUSION=true
                shift
                ;;
            --reranker=*)
                RERANKER="${1#*=}"
                shift
                ;;
            --max-passages=*)
                MAX_PASSAGES="${1#*=}"
                shift
                ;;
            --debug)
                INCLUDE_DEBUG=true
                shift
                ;;
            --format=*)
                OUTPUT_FORMAT="${1#*=}"
                shift
                ;;
            --api-url=*)
                API_BASE_URL="${1#*=}"
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --*)
                log ERROR "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [[ -z "$QUERY" ]]; then
                    QUERY="$1"
                else
                    log ERROR "Multiple queries not supported"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$QUERY" ]]; then
        log ERROR "Query is required"
        usage
        exit 1
    fi

    # Validate parameters
    validate_parameters

    if $VERBOSE; then
        log INFO "Starting retrieval for query: \"$QUERY\""
        log INFO "Parameters: topk=$TOPK, alpha=$ALPHA, rrf=$USE_RRF, reranker=$RERANKER"
    fi

    # Create and submit request
    local request_body
    request_body=$(create_retrieval_request)

    local response
    response=$(submit_retrieval_request "$request_body" "retrieve")

    # Format and display output
    format_output "$response" "$OUTPUT_FORMAT"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
