#!/usr/bin/env bash

# askme Document Ingestion Script
# Usage: ./ingest.sh <path> [options]

set -euo pipefail

# Default values
SOURCE_TYPE=""
SOURCE_PATH=""
TAGS=""
OVERWRITE="false"
API_BASE_URL="${ASKME_API_URL:-http://localhost:8080}"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
askme Document Ingestion Script

USAGE:
    $0 <path_or_url> [OPTIONS]

ARGUMENTS:
    <path_or_url>    Path to file/directory or URL to ingest

OPTIONS:
    --tags=TAGS      Comma-separated tags (e.g., "project,team")
    --overwrite      Overwrite existing documents (default: false)
    --api-url=URL    API base URL (default: http://localhost:8080)
    --verbose        Enable verbose output
    --help          Show this help message

EXAMPLES:
    # Ingest a single PDF file
    $0 /path/to/document.pdf --tags="manual,v1.0"

    # Ingest entire directory
    $0 /path/to/documents --tags="project" --overwrite

    # Ingest from URL
    $0 https://example.com/api/docs --tags="external"

    # Upload files directly
    $0 --upload file1.pdf file2.txt --tags="upload"

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
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" >&2
            ;;
    esac
}

detect_source_type() {
    local path="$1"

    if [[ "$path" =~ ^https?:// ]]; then
        echo "url"
    elif [[ -f "$path" ]]; then
        echo "file"
    elif [[ -d "$path" ]]; then
        echo "dir"
    else
        log ERROR "Invalid path: $path"
        exit 1
    fi
}

validate_path() {
    local path="$1"
    local source_type="$2"

    case $source_type in
        "file")
            if [[ ! -f "$path" ]]; then
                log ERROR "File not found: $path"
                exit 1
            fi
            ;;
        "dir")
            if [[ ! -d "$path" ]]; then
                log ERROR "Directory not found: $path"
                exit 1
            fi
            ;;
        "url")
            # Basic URL validation
            if ! curl -s --head "$path" > /dev/null; then
                log WARN "URL may not be accessible: $path"
            fi
            ;;
    esac
}

create_ingest_request() {
    local source_type="$1"
    local source_path="$2"
    local tags="$3"
    local overwrite="$4"

    local tags_json="null"
    if [[ -n "$tags" ]]; then
        # Convert comma-separated tags to JSON array
        tags_json=$(echo "$tags" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')
    fi

    cat << EOF
{
    "source": "$source_type",
    "path": "$source_path",
    "tags": $tags_json,
    "overwrite": $overwrite
}
EOF
}

submit_ingest_job() {
    local request_body="$1"

    local curl_args=(
        -s
        -X POST
        -H "Content-Type: application/json"
        -d "$request_body"
        "${API_BASE_URL}/ingest/"
    )

    # Add API key if available
    if [[ -n "${ASKME_API_KEY:-}" ]]; then
        curl_args+=(-H "X-API-Key: $ASKME_API_KEY")
    fi

    if $VERBOSE; then
        log INFO "Submitting ingestion request..."
        echo "Request body:" >&2
        echo "$request_body" | jq . >&2
    fi

    local response
    if ! response=$(curl "${curl_args[@]}"); then
        log ERROR "Failed to submit ingestion request"
        exit 1
    fi

    echo "$response"
}

poll_task_status() {
    local task_id="$1"
    local timeout=${2:-999999}  # Effectively no timeout (~11 days)
    local interval=5
    local elapsed=0

    log INFO "Polling task status for: $task_id"

    while [[ $elapsed -lt $timeout ]]; do
        local response
        if ! response=$(curl -s "${API_BASE_URL}/ingest/status/${task_id}"); then
            log WARN "Failed to check task status, retrying..."
            sleep $interval
            elapsed=$((elapsed + interval))
            continue
        fi

        local status
        status=$(echo "$response" | jq -r '.status // "unknown"')

        case $status in
            "completed")
                local doc_count
                doc_count=$(echo "$response" | jq -r '.documents_processed // 0')
                log SUCCESS "Ingestion completed! Processed $doc_count documents"

                if $VERBOSE; then
                    echo "Final status:" >&2
                    echo "$response" | jq . >&2
                fi
                return 0
                ;;
            "failed")
                local error_msg
                error_msg=$(echo "$response" | jq -r '.error_message // "Unknown error"')
                log ERROR "Ingestion failed: $error_msg"
                return 1
                ;;
            "processing"|"queued")
                local progress
                progress=$(echo "$response" | jq -r '.progress // 0')
                local processed
                processed=$(echo "$response" | jq -r '.documents_processed // 0')
                log INFO "Status: $status (${progress}% complete, $processed docs processed)"
                ;;
            *)
                log WARN "Unknown status: $status"
                ;;
        esac

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log ERROR "Timeout waiting for ingestion to complete"
    return 1
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
            --tags=*)
                TAGS="${1#*=}"
                shift
                ;;
            --overwrite)
                OVERWRITE="true"
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
                if [[ -z "$SOURCE_PATH" ]]; then
                    SOURCE_PATH="$1"
                else
                    log ERROR "Multiple paths not supported"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$SOURCE_PATH" ]]; then
        log ERROR "Source path is required"
        usage
        exit 1
    fi

    # Detect and validate source type
    SOURCE_TYPE=$(detect_source_type "$SOURCE_PATH")
    validate_path "$SOURCE_PATH" "$SOURCE_TYPE"

    log INFO "Starting ingestion: $SOURCE_TYPE -> $SOURCE_PATH"

    # Create and submit request
    local request_body
    request_body=$(create_ingest_request "$SOURCE_TYPE" "$SOURCE_PATH" "$TAGS" "$OVERWRITE")

    local response
    response=$(submit_ingest_job "$request_body")

    # Extract task ID
    local task_id
    task_id=$(echo "$response" | jq -r '.task_id // empty')

    if [[ -z "$task_id" ]]; then
        log ERROR "No task ID received from server"
        echo "Response:" >&2
        echo "$response" | jq . >&2
        exit 1
    fi

    log INFO "Ingestion task submitted with ID: $task_id"

    # Poll for completion
    poll_task_status "$task_id"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
