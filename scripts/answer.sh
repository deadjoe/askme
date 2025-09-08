#!/usr/bin/env bash

# askme Question Answering Script
# Usage: ./answer.sh "<question>" [options]

set -euo pipefail

# Default values
QUESTION=""
TOPK=50
ALPHA=0.5
USE_RRF=true
RRF_K=60
USE_HYDE=false
USE_RAG_FUSION=false
RERANKER="bge_local"
MAX_PASSAGES=8
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
OUTPUT_FORMAT="text"
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
askme Question Answering Script

USAGE:
    $0 "<question>" [OPTIONS]

ARGUMENTS:
    <question>       Question to answer (required)

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
    --format=FORMAT  Output format: text, json, markdown (default: text)
    --api-url=URL    API base URL (default: http://localhost:8080)
    --verbose        Enable verbose output
    --help          Show this help message

EXAMPLES:
    # Basic question answering
    $0 "What is machine learning?"
    
    # Enhanced answering with HyDE and RAG-Fusion
    $0 "Explain neural networks" --hyde --rag-fusion --format=markdown
    
    # Sparse-focused search with Cohere reranker
    $0 "best practices for RAG systems" --alpha=0.3 --reranker=cohere
    
    # Debug mode with detailed retrieval info
    $0 "how does attention work?" --debug --verbose

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
    if [[ "$OUTPUT_FORMAT" != "json" && "$OUTPUT_FORMAT" != "text" && "$OUTPUT_FORMAT" != "markdown" ]]; then
        log ERROR "Invalid output format: $OUTPUT_FORMAT. Must be 'json', 'text', or 'markdown'"
        exit 1
    fi
}

create_query_request() {
    cat << EOF
{
    "q": "$QUESTION",
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

submit_query_request() {
    local request_body="$1"
    
    local curl_args=(
        -s
        -X POST
        -H "Content-Type: application/json"
        -d "$request_body"
        "${API_BASE_URL}/query/"
    )
    
    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi
    
    if $VERBOSE; then
        log INFO "Submitting query request..."
        echo "Request body:" >&2
        echo "$request_body" | jq . >&2
    fi
    
    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to submit query request"
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
        "markdown")
            format_markdown_output "$response"
            ;;
    esac
}

format_text_output() {
    local response="$1"
    
    # Extract basic info
    local query_id answer timestamp
    query_id=$(echo "$response" | jq -r '.query_id // "N/A"')
    answer=$(echo "$response" | jq -r '.answer // "No answer generated"')
    timestamp=$(echo "$response" | jq -r '.timestamp // "N/A"')
    
    echo "Question: \"$QUESTION\""
    echo "Query ID: $query_id"
    echo "Timestamp: $timestamp"
    echo
    echo "Answer:"
    echo "======="
    echo "$answer"
    echo
    
    # Display citations
    local citations
    citations=$(echo "$response" | jq -c '.citations[]? // empty')
    
    if [[ -n "$citations" ]]; then
        echo "Sources:"
        echo "========"
        
        local count=1
        while IFS= read -r citation; do
            if [[ -n "$citation" ]]; then
                local doc_id title score metadata
                doc_id=$(echo "$citation" | jq -r '.doc_id // "N/A"')
                title=$(echo "$citation" | jq -r '.title // "Untitled"')
                score=$(echo "$citation" | jq -r '.score // 0')
                
                echo "[$count] $title (ID: $doc_id, Score: $score)"
                
                # Show metadata if available
                metadata=$(echo "$citation" | jq -r '.metadata // empty')
                if [[ "$metadata" != "null" && -n "$metadata" ]]; then
                    local author date
                    author=$(echo "$metadata" | jq -r '.author // empty' 2>/dev/null || echo "")
                    date=$(echo "$metadata" | jq -r '.date // empty' 2>/dev/null || echo "")
                    
                    if [[ -n "$author" || -n "$date" ]]; then
                        echo "    Author: ${author:-"Unknown"}, Date: ${date:-"Unknown"}"
                    fi
                fi
                
                ((count++))
            fi
        done <<< "$citations"
        echo
    fi
    
    # Display debug info if available and requested
    if $INCLUDE_DEBUG; then
        local debug_info
        debug_info=$(echo "$response" | jq '.retrieval_debug // empty')
        if [[ "$debug_info" != "null" && -n "$debug_info" ]]; then
            echo "Debug Information:"
            echo "=================="
            
            local bm25_hits dense_hits fusion_method latency_ms
            bm25_hits=$(echo "$debug_info" | jq -r '.bm25_hits // 0')
            dense_hits=$(echo "$debug_info" | jq -r '.dense_hits // 0')
            fusion_method=$(echo "$debug_info" | jq -r '.fusion_method // "unknown"')
            latency_ms=$(echo "$debug_info" | jq -r '.latency_ms // 0')
            
            echo "BM25 hits: $bm25_hits"
            echo "Dense hits: $dense_hits"
            echo "Fusion method: $fusion_method"
            echo "Total latency: ${latency_ms}ms"
            
            local embedding_latency rerank_latency search_latency
            embedding_latency=$(echo "$debug_info" | jq -r '.embedding_latency_ms // 0')
            search_latency=$(echo "$debug_info" | jq -r '.search_latency_ms // 0')
            rerank_latency=$(echo "$debug_info" | jq -r '.rerank_latency_ms // 0')
            
            echo "  - Embedding: ${embedding_latency}ms"
            echo "  - Search: ${search_latency}ms"
            echo "  - Reranking: ${rerank_latency}ms"
            echo
        fi
    fi
}

format_markdown_output() {
    local response="$1"
    
    # Extract basic info
    local query_id answer timestamp
    query_id=$(echo "$response" | jq -r '.query_id // "N/A"')
    answer=$(echo "$response" | jq -r '.answer // "No answer generated"')
    timestamp=$(echo "$response" | jq -r '.timestamp // "N/A"')
    
    echo "# Question: $QUESTION"
    echo
    echo "**Query ID:** $query_id  "
    echo "**Timestamp:** $timestamp"
    echo
    echo "## Answer"
    echo
    echo "$answer"
    echo
    
    # Display citations
    local citations
    citations=$(echo "$response" | jq -c '.citations[]? // empty')
    
    if [[ -n "$citations" ]]; then
        echo "## Sources"
        echo
        
        local count=1
        while IFS= read -r citation; do
            if [[ -n "$citation" ]]; then
                local doc_id title content score metadata
                doc_id=$(echo "$citation" | jq -r '.doc_id // "N/A"')
                title=$(echo "$citation" | jq -r '.title // "Untitled"')
                content=$(echo "$citation" | jq -r '.content // ""')
                score=$(echo "$citation" | jq -r '.score // 0')
                
                echo "### $count. $title"
                echo
                echo "**Document ID:** $doc_id  "
                echo "**Relevance Score:** $score"
                echo
                
                if [[ -n "$content" ]]; then
                    echo "**Excerpt:**"
                    echo "> $(echo "$content" | sed 's/^/> /')"
                    echo
                fi
                
                # Show metadata if available
                metadata=$(echo "$citation" | jq -r '.metadata // empty')
                if [[ "$metadata" != "null" && -n "$metadata" ]]; then
                    local author date
                    author=$(echo "$metadata" | jq -r '.author // empty' 2>/dev/null || echo "")
                    date=$(echo "$metadata" | jq -r '.date // empty' 2>/dev/null || echo "")
                    
                    if [[ -n "$author" || -n "$date" ]]; then
                        echo "**Metadata:**"
                        [[ -n "$author" ]] && echo "- Author: $author"
                        [[ -n "$date" ]] && echo "- Date: $date"
                        echo
                    fi
                fi
                
                ((count++))
            fi
        done <<< "$citations"
    fi
    
    # Display debug info if available and requested
    if $INCLUDE_DEBUG; then
        local debug_info
        debug_info=$(echo "$response" | jq '.retrieval_debug // empty')
        if [[ "$debug_info" != "null" && -n "$debug_info" ]]; then
            echo "## Debug Information"
            echo
            
            local bm25_hits dense_hits fusion_method latency_ms
            bm25_hits=$(echo "$debug_info" | jq -r '.bm25_hits // 0')
            dense_hits=$(echo "$debug_info" | jq -r '.dense_hits // 0')
            fusion_method=$(echo "$debug_info" | jq -r '.fusion_method // "unknown"')
            latency_ms=$(echo "$debug_info" | jq -r '.latency_ms // 0')
            
            echo "| Metric | Value |"
            echo "|--------|-------|"
            echo "| BM25 hits | $bm25_hits |"
            echo "| Dense hits | $dense_hits |"
            echo "| Fusion method | $fusion_method |"
            echo "| Total latency | ${latency_ms}ms |"
            
            local embedding_latency rerank_latency search_latency
            embedding_latency=$(echo "$debug_info" | jq -r '.embedding_latency_ms // 0')
            search_latency=$(echo "$debug_info" | jq -r '.search_latency_ms // 0')
            rerank_latency=$(echo "$debug_info" | jq -r '.rerank_latency_ms // 0')
            
            echo "| Embedding latency | ${embedding_latency}ms |"
            echo "| Search latency | ${search_latency}ms |"
            echo "| Reranking latency | ${rerank_latency}ms |"
            echo
        fi
    fi
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
                if [[ -z "$QUESTION" ]]; then
                    QUESTION="$1"
                else
                    log ERROR "Multiple questions not supported"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$QUESTION" ]]; then
        log ERROR "Question is required"
        usage
        exit 1
    fi
    
    # Validate parameters
    validate_parameters
    
    if $VERBOSE; then
        log INFO "Processing question: \"$QUESTION\""
        log INFO "Parameters: topk=$TOPK, alpha=$ALPHA, rrf=$USE_RRF, reranker=$RERANKER"
        log INFO "Enhancements: hyde=$USE_HYDE, rag_fusion=$USE_RAG_FUSION"
    fi
    
    # Create and submit request
    local request_body
    request_body=$(create_query_request)
    
    local response
    response=$(submit_query_request "$request_body")
    
    if $VERBOSE; then
        log SUCCESS "Query completed successfully"
    fi
    
    # Format and display output
    format_output "$response" "$OUTPUT_FORMAT"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi