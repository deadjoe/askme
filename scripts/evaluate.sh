#!/usr/bin/env bash

# askme Evaluation Script
# Usage: ./evaluate.sh [options]

set -euo pipefail

# Default values
SUITE="baseline"
METRICS=""
DATASET_PATH=""
SAMPLE_SIZE=""
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
OUTPUT_FORMAT="text"
WAIT_FOR_COMPLETION=true
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
askme Evaluation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --suite=SUITE        Evaluation suite: baseline, custom, regression, quick (default: baseline)
    --metrics=METRICS    Comma-separated metrics (default: all available)
    --dataset=PATH       Custom dataset path (overrides suite default)
    --sample-size=N      Number of samples to evaluate (default: all)
    --format=FORMAT      Output format: text, json, table (default: text)
    --no-wait           Don't wait for completion, just start evaluation
    --api-url=URL       API base URL (default: http://localhost:8080)
    --verbose           Enable verbose output
    --help             Show this help message

AVAILABLE SUITES:
    baseline            Standard evaluation dataset
    custom              Custom dataset (requires --dataset)
    regression          Regression testing dataset
    quick               Small subset for quick testing

AVAILABLE METRICS:
    TruLens metrics:
        context_relevance    How relevant retrieved context is to query
        groundedness        How well answer is supported by context  
        answer_relevance    How relevant answer is to query
    
    Ragas metrics:
        faithfulness        Factual consistency of answer with context
        answer_relevancy    Relevance of answer to query
        context_precision   Precision of retrieved context
        context_recall      Recall of retrieved context

EXAMPLES:
    # Run full baseline evaluation
    $0 --suite=baseline
    
    # Quick evaluation with specific metrics
    $0 --suite=quick --metrics="faithfulness,context_relevance"
    
    # Custom dataset evaluation
    $0 --dataset="/path/to/qa_dataset.jsonl" --sample-size=100
    
    # Regression testing
    $0 --suite=regression --format=json --no-wait

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
    # Validate suite
    case $SUITE in
        baseline|custom|regression|quick)
            ;;
        *)
            log ERROR "Invalid suite: $SUITE. Must be one of: baseline, custom, regression, quick"
            exit 1
            ;;
    esac
    
    # Check custom dataset requirement
    if [[ "$SUITE" == "custom" && -z "$DATASET_PATH" ]]; then
        log ERROR "Custom suite requires --dataset parameter"
        exit 1
    fi
    
    # Validate dataset path if provided
    if [[ -n "$DATASET_PATH" && ! -f "$DATASET_PATH" ]]; then
        log ERROR "Dataset file not found: $DATASET_PATH"
        exit 1
    fi
    
    # Validate sample size
    if [[ -n "$SAMPLE_SIZE" && ! "$SAMPLE_SIZE" =~ ^[0-9]+$ ]]; then
        log ERROR "Sample size must be a positive integer: $SAMPLE_SIZE"
        exit 1
    fi
    
    # Validate metrics
    if [[ -n "$METRICS" ]]; then
        local available_metrics="context_relevance,groundedness,answer_relevance,faithfulness,answer_relevancy,context_precision,context_recall"
        IFS=',' read -ra metric_array <<< "$METRICS"
        for metric in "${metric_array[@]}"; do
            if [[ ",$available_metrics," != *",$metric,"* ]]; then
                log ERROR "Invalid metric: $metric"
                log ERROR "Available metrics: $available_metrics"
                exit 1
            fi
        done
    fi
    
    # Validate output format
    if [[ "$OUTPUT_FORMAT" != "json" && "$OUTPUT_FORMAT" != "text" && "$OUTPUT_FORMAT" != "table" ]]; then
        log ERROR "Invalid output format: $OUTPUT_FORMAT. Must be 'json', 'text', or 'table'"
        exit 1
    fi
}

create_evaluation_request() {
    local metrics_json="null"
    
    if [[ -n "$METRICS" ]]; then
        # Convert comma-separated metrics to JSON array
        metrics_json=$(echo "$METRICS" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')
    fi
    
    local sample_size_json="null"
    if [[ -n "$SAMPLE_SIZE" ]]; then
        sample_size_json="$SAMPLE_SIZE"
    fi
    
    local dataset_path_json="null"
    if [[ -n "$DATASET_PATH" ]]; then
        dataset_path_json="\"$DATASET_PATH\""
    fi
    
    cat << EOF
{
    "suite": "$SUITE",
    "metrics": $metrics_json,
    "dataset_path": $dataset_path_json,
    "sample_size": $sample_size_json
}
EOF
}

submit_evaluation_request() {
    local request_body="$1"
    
    local curl_args=(
        -s
        -X POST
        -H "Content-Type: application/json"
        -d "$request_body"
        "${API_BASE_URL}/eval/run"
    )
    
    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi
    
    if $VERBOSE; then
        log INFO "Submitting evaluation request..."
        echo "Request body:" >&2
        echo "$request_body" | jq . >&2
    fi
    
    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to submit evaluation request"
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

get_evaluation_results() {
    local run_id="$1"
    
    local curl_args=(
        -s
        -X GET
        "${API_BASE_URL}/eval/runs/${run_id}"
    )
    
    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi
    
    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to get evaluation results"
        exit 1
    fi
    
    echo "$response"
}

poll_evaluation_status() {
    local run_id="$1"
    local timeout=${2:-1800}  # 30 minutes default timeout
    local interval=10
    local elapsed=0
    
    log INFO "Polling evaluation status for run: $run_id"
    
    while [[ $elapsed -lt $timeout ]]; do
        local response
        response=$(get_evaluation_results "$run_id")
        
        local status
        status=$(echo "$response" | jq -r '.status // "unknown"')
        
        case $status in
            "completed")
                log SUCCESS "Evaluation completed successfully!"
                echo "$response"
                return 0
                ;;
            "failed")
                local error_msg
                error_msg=$(echo "$response" | jq -r '.error_message // "Unknown error"')
                log ERROR "Evaluation failed: $error_msg"
                return 1
                ;;
            "running"|"queued")
                local processed total
                processed=$(echo "$response" | jq -r '.processed_samples // 0')
                total=$(echo "$response" | jq -r '.total_samples // 0')
                local progress=""
                if [[ $total -gt 0 ]]; then
                    progress=" ($processed/$total)"
                fi
                log INFO "Status: $status$progress"
                ;;
            *)
                log WARN "Unknown status: $status"
                ;;
        esac
        
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log ERROR "Timeout waiting for evaluation to complete"
    return 1
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
    local run_id suite status started_at completed_at total_samples processed_samples
    run_id=$(echo "$response" | jq -r '.run_id // "N/A"')
    suite=$(echo "$response" | jq -r '.suite // "N/A"')
    status=$(echo "$response" | jq -r '.status // "N/A"')
    started_at=$(echo "$response" | jq -r '.started_at // "N/A"')
    completed_at=$(echo "$response" | jq -r '.completed_at // "N/A"')
    total_samples=$(echo "$response" | jq -r '.total_samples // 0')
    processed_samples=$(echo "$response" | jq -r '.processed_samples // 0')
    
    echo "Evaluation Results"
    echo "=================="
    echo "Run ID: $run_id"
    echo "Suite: $suite"
    echo "Status: $status"
    echo "Started: $started_at"
    echo "Completed: $completed_at"
    echo "Samples: $processed_samples/$total_samples"
    echo
    
    # Display overall metrics
    local metrics
    metrics=$(echo "$response" | jq -c '.overall_metrics[]? // empty')
    
    if [[ -n "$metrics" ]]; then
        echo "Overall Metrics:"
        echo "================"
        
        while IFS= read -r metric; do
            if [[ -n "$metric" ]]; then
                local name value threshold passed
                name=$(echo "$metric" | jq -r '.name // "N/A"')
                value=$(echo "$metric" | jq -r '.value // 0')
                threshold=$(echo "$metric" | jq -r '.threshold // null')
                passed=$(echo "$metric" | jq -r '.passed // false')
                
                local status_icon="❌"
                if [[ "$passed" == "true" ]]; then
                    status_icon="✅"
                fi
                
                printf "%-20s: %.4f" "$name" "$value"
                if [[ "$threshold" != "null" ]]; then
                    printf " (threshold: %.2f) %s" "$threshold" "$status_icon"
                fi
                echo
            fi
        done <<< "$metrics"
        echo
    fi
    
    # Display summary statistics
    local summary
    summary=$(echo "$response" | jq '.summary // empty')
    if [[ "$summary" != "null" && -n "$summary" ]]; then
        echo "Summary Statistics:"
        echo "==================="
        echo "$summary" | jq -r 'to_entries[] | "\(.key): \(.value)"'
        echo
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
    
    # Extract basic info for header
    local run_id suite status
    run_id=$(echo "$response" | jq -r '.run_id // "N/A"')
    suite=$(echo "$response" | jq -r '.suite // "N/A"')
    status=$(echo "$response" | jq -r '.status // "N/A"')
    
    echo "Evaluation Results - Run ID: $run_id, Suite: $suite, Status: $status"
    echo
    
    # Create metrics table
    echo "Overall Metrics:"
    printf "%-20s %-10s %-12s %-8s\n" "Metric" "Value" "Threshold" "Status"
    printf "%-20s %-10s %-12s %-8s\n" "--------------------" "----------" "------------" "--------"
    
    local metrics
    metrics=$(echo "$response" | jq -c '.overall_metrics[]? // empty')
    
    while IFS= read -r metric; do
        if [[ -n "$metric" ]]; then
            local name value threshold passed
            name=$(echo "$metric" | jq -r '.name // "N/A"')
            value=$(echo "$metric" | jq -r '.value // 0')
            threshold=$(echo "$metric" | jq -r '.threshold // null')
            passed=$(echo "$metric" | jq -r '.passed // false')
            
            local status_text="FAIL"
            if [[ "$passed" == "true" ]]; then
                status_text="PASS"
            fi
            
            local threshold_text="N/A"
            if [[ "$threshold" != "null" ]]; then
                threshold_text=$(printf "%.3f" "$threshold")
            fi
            
            printf "%-20s %-10.4f %-12s %-8s\n" "$name" "$value" "$threshold_text" "$status_text"
        fi
    done <<< "$metrics"
    echo
}

list_available_metrics() {
    local curl_args=(
        -s
        -X GET
        "${API_BASE_URL}/eval/metrics"
    )
    
    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi
    
    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to get available metrics"
        return 1
    fi
    
    echo "Available Evaluation Metrics:"
    echo "============================="
    echo
    echo "TruLens Metrics:"
    echo "$response" | jq -r '.trulens_metrics | to_entries[] | "  \(.key): \(.value.description)"'
    echo
    echo "Ragas Metrics:"
    echo "$response" | jq -r '.ragas_metrics | to_entries[] | "  \(.key): \(.value.description)"'
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
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                usage
                exit 0
                ;;
            --suite=*)
                SUITE="${1#*=}"
                shift
                ;;
            --metrics=*)
                METRICS="${1#*=}"
                shift
                ;;
            --dataset=*)
                DATASET_PATH="${1#*=}"
                shift
                ;;
            --sample-size=*)
                SAMPLE_SIZE="${1#*=}"
                shift
                ;;
            --format=*)
                OUTPUT_FORMAT="${1#*=}"
                shift
                ;;
            --no-wait)
                WAIT_FOR_COMPLETION=false
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
            --list-metrics)
                list_available_metrics
                exit 0
                ;;
            --*)
                log ERROR "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                log ERROR "Unexpected argument: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate parameters
    validate_parameters
    
    if $VERBOSE; then
        log INFO "Starting evaluation with suite: $SUITE"
        [[ -n "$METRICS" ]] && log INFO "Metrics: $METRICS"
        [[ -n "$SAMPLE_SIZE" ]] && log INFO "Sample size: $SAMPLE_SIZE"
        [[ -n "$DATASET_PATH" ]] && log INFO "Dataset: $DATASET_PATH"
    fi
    
    # Create and submit request
    local request_body
    request_body=$(create_evaluation_request)
    
    local response
    response=$(submit_evaluation_request "$request_body")
    
    # Extract run ID
    local run_id
    run_id=$(echo "$response" | jq -r '.run_id // empty')
    
    if [[ -z "$run_id" ]]; then
        log ERROR "No run ID received from server"
        echo "Response:" >&2
        echo "$response" | jq . >&2
        exit 1
    fi
    
    log INFO "Evaluation started with run ID: $run_id"
    
    if $WAIT_FOR_COMPLETION; then
        # Poll for completion and display results
        local final_response
        if final_response=$(poll_evaluation_status "$run_id"); then
            format_output "$final_response" "$OUTPUT_FORMAT"
        else
            log ERROR "Evaluation did not complete successfully"
            exit 1
        fi
    else
        log INFO "Evaluation started. Use 'curl ${API_BASE_URL}/eval/runs/${run_id}' to check status"
        echo "Run ID: $run_id"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi