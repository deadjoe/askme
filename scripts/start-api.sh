#!/bin/bash

# AskMe RAG API Startup Script
# Automatically configure environment variables and start the API server

set -euo pipefail

# Script information
SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Logging functions
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" >&2
}

# Help information
show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [options]

AskMe RAG API startup script

Options:
  -h, --help              Show this help information
  -p, --port PORT         API server port (default: 8080)
  -H, --host HOST         API server host (default: 0.0.0.0)
  --reload                Enable hot reload mode
  --workers N             Number of worker processes (default: 1)
  --milvus-host HOST      Milvus host address (default: 127.0.0.1)
  --milvus-port PORT      Milvus port (default: 19530)
  --ollama-model MODEL    Ollama model name (default: qwen3:30b-a3b)
  --ollama-endpoint URL   Ollama endpoint address (default: http://localhost:11434)
  --openai-url URL        OpenAI compatible endpoint (default: http://localhost:11434/v1)
  --ragas-model MODEL     Ragas evaluation model (default: gpt-oss:20b)
  --enable-cohere         Enable Cohere reranking (requires COHERE_API_KEY)
  --vector-backend TYPE   Vector database backend (milvus|weaviate|qdrant, default: milvus)
  --skip-heavy-init       Skip heavy service initialization (fast startup)
  --log-level LEVEL       Log level (DEBUG|INFO|WARNING|ERROR, default: INFO)
  --dry-run               Show configuration only, do not start service

Examples:
  $SCRIPT_NAME                                    # Start with default configuration
  $SCRIPT_NAME --port 8081 --reload              # Specify port and enable hot reload
  $SCRIPT_NAME --ollama-model llama3.1:latest    # Use different Ollama model
  $SCRIPT_NAME --enable-cohere                   # Enable Cohere reranking
  $SCRIPT_NAME --dry-run                         # Show configuration only

EOF
}

# Default configuration
DEFAULT_PORT=8080
DEFAULT_HOST="0.0.0.0"
DEFAULT_RELOAD=false
DEFAULT_WORKERS=1
DEFAULT_MILVUS_HOST="127.0.0.1"
DEFAULT_MILVUS_PORT=19530
# DEFAULT_OLLAMA_MODEL="qwen3:30b-a3b"
DEFAULT_OLLAMA_MODEL="gpt-oss:120b-clou"
DEFAULT_OLLAMA_ENDPOINT="http://localhost:11434"
DEFAULT_OPENAI_URL="http://localhost:11434/v1"
DEFAULT_RAGAS_MODEL="gpt-oss:20b"
DEFAULT_ENABLE_COHERE=false
DEFAULT_VECTOR_BACKEND="milvus"
DEFAULT_SKIP_HEAVY_INIT=false
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_DRY_RUN=false

# Parse command line arguments
PORT="$DEFAULT_PORT"
HOST="$DEFAULT_HOST"
RELOAD="$DEFAULT_RELOAD"
WORKERS="$DEFAULT_WORKERS"
MILVUS_HOST="$DEFAULT_MILVUS_HOST"
MILVUS_PORT="$DEFAULT_MILVUS_PORT"
OLLAMA_MODEL="$DEFAULT_OLLAMA_MODEL"
OLLAMA_ENDPOINT="$DEFAULT_OLLAMA_ENDPOINT"
OPENAI_URL="$DEFAULT_OPENAI_URL"
RAGAS_MODEL="$DEFAULT_RAGAS_MODEL"
ENABLE_COHERE="$DEFAULT_ENABLE_COHERE"
VECTOR_BACKEND="$DEFAULT_VECTOR_BACKEND"
SKIP_HEAVY_INIT="$DEFAULT_SKIP_HEAVY_INIT"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"
DRY_RUN="$DEFAULT_DRY_RUN"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --milvus-host)
            MILVUS_HOST="$2"
            shift 2
            ;;
        --milvus-port)
            MILVUS_PORT="$2"
            shift 2
            ;;
        --ollama-model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --ollama-endpoint)
            OLLAMA_ENDPOINT="$2"
            shift 2
            ;;
        --openai-url)
            OPENAI_URL="$2"
            shift 2
            ;;
        --ragas-model)
            RAGAS_MODEL="$2"
            shift 2
            ;;
        --enable-cohere)
            ENABLE_COHERE=true
            shift
            ;;
        --vector-backend)
            VECTOR_BACKEND="$2"
            shift 2
            ;;
        --skip-heavy-init)
            SKIP_HEAVY_INIT=true
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log ERROR "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate parameters
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [[ "$PORT" -lt 1 ]] || [[ "$PORT" -gt 65535 ]]; then
    log ERROR "Invalid port number: $PORT"
    exit 1
fi

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
    log ERROR "Invalid worker count: $WORKERS"
    exit 1
fi

if [[ "$VECTOR_BACKEND" != "milvus" && "$VECTOR_BACKEND" != "weaviate" && "$VECTOR_BACKEND" != "qdrant" ]]; then
    log ERROR "Invalid vector backend: $VECTOR_BACKEND"
    exit 1
fi

if [[ "$LOG_LEVEL" != "DEBUG" && "$LOG_LEVEL" != "INFO" && "$LOG_LEVEL" != "WARNING" && "$LOG_LEVEL" != "ERROR" ]]; then
    log ERROR "Invalid log level: $LOG_LEVEL"
    exit 1
fi

# Set environment variables
export ASKME_VECTOR_BACKEND="$VECTOR_BACKEND"
export ASKME_DATABASE__HOST="$MILVUS_HOST"
export ASKME_DATABASE__PORT="$MILVUS_PORT"
export ASKME_DATABASE__MILVUS__HOST="$MILVUS_HOST"
export ASKME_DATABASE__MILVUS__PORT="$MILVUS_PORT"
export ASKME_ENABLE_OLLAMA=1
export ASKME_GENERATION__PROVIDER=ollama
export ASKME_GENERATION__OLLAMA_MODEL="$OLLAMA_MODEL"
export ASKME_GENERATION__OLLAMA_ENDPOINT="$OLLAMA_ENDPOINT"
export OPENAI_BASE_URL="$OPENAI_URL"
export OPENAI_API_KEY="ollama-local"
export ASKME_RAGAS_LLM_MODEL="$RAGAS_MODEL"
export ASKME_LOG_LEVEL="$LOG_LEVEL"

if [[ "$ENABLE_COHERE" == "true" ]]; then
    export ASKME_ENABLE_COHERE=1
    if [[ -z "${COHERE_API_KEY:-}" ]]; then
        log WARNING "Cohere enabled but COHERE_API_KEY environment variable not set"
    fi
fi

if [[ "$SKIP_HEAVY_INIT" == "true" ]]; then
    export ASKME_SKIP_HEAVY_INIT=1
fi

# All available parameters from codebase analysis
# Use bash 3.2 compatible approach - separate arrays for keys and descriptions
ALL_PARAM_KEYS=(
    "ASKME_VECTOR_BACKEND"
    "ASKME_ENABLE_OLLAMA"
    "ASKME_ENABLE_COHERE"
    "ASKME_SKIP_HEAVY_INIT"
    "ASKME_DATABASE__HOST"
    "ASKME_DATABASE__PORT"
    "ASKME_DATABASE__MILVUS__HOST"
    "ASKME_DATABASE__MILVUS__PORT"
    "ASKME_DATABASE__MILVUS__USERNAME"
    "ASKME_DATABASE__MILVUS__PASSWORD"
    "ASKME_DATABASE__MILVUS__SECURE"
    "ASKME_DATABASE__MILVUS__COLLECTION_NAME"
    "ASKME_DATABASE__WEAVIATE__URL"
    "ASKME_DATABASE__WEAVIATE__API_KEY"
    "ASKME_DATABASE__WEAVIATE__CLASS_NAME"
    "ASKME_DATABASE__QDRANT__URL"
    "ASKME_DATABASE__QDRANT__API_KEY"
    "ASKME_DATABASE__QDRANT__COLLECTION_NAME"
    "ASKME_GENERATION__PROVIDER"
    "ASKME_GENERATION__OLLAMA_MODEL"
    "ASKME_GENERATION__OLLAMA_ENDPOINT"
    "ASKME_GENERATION__MODEL_NAME"
    "ASKME_GENERATION__MAX_TOKENS"
    "ASKME_GENERATION__TEMPERATURE"
    "ASKME_GENERATION__TOP_P"
    "ASKME_GENERATION__OPENAI_MODEL"
    "ASKME_GENERATION__OPENAI_BASE_URL"
    "ASKME_GENERATION__OPENAI_API_KEY_ENV"
    "OPENAI_BASE_URL"
    "OPENAI_API_KEY"
    "COHERE_API_KEY"
    "ASKME_RAGAS_LLM_MODEL"
    "ASKME_RAGAS_EMBED_MODEL"
    "ASKME_EMBEDDING__MODEL"
    "ASKME_EMBEDDING__DIMENSION"
    "ASKME_EMBEDDING__BATCH_SIZE"
    "ASKME_EMBEDDING__MAX_LENGTH"
    "ASKME_HYBRID__MODE"
    "ASKME_HYBRID__ALPHA"
    "ASKME_HYBRID__RRF_K"
    "ASKME_HYBRID__TOPK"
    "ASKME_RERANK__LOCAL_MODEL"
    "ASKME_RERANK__TOP_N"
    "ASKME_RERANK__COHERE_MODEL"
    "ASKME_API__HOST"
    "ASKME_API__PORT"
    "ASKME_API__WORKERS"
    "ASKME_API__RELOAD"
    "ASKME_ENHANCER__HYDE__ENABLED"
    "ASKME_ENHANCER__RAG_FUSION__ENABLED"
    "ASKME_PERFORMANCE__BATCH__EMBEDDING_BATCH_SIZE"
    "ASKME_PERFORMANCE__BATCH__RERANK_BATCH_SIZE"
    "ASKME_PERFORMANCE__TIMEOUTS__RETRIEVAL_TIMEOUT"
    "ASKME_LOG_LEVEL"
    "ASKME_LOGGING__LEVEL"
    "ASKME_OLLAMA_READ_TIMEOUT"
    "ASKME_OLLAMA_THINKING"
    "TOKENIZERS_PARALLELISM"
    "OLLAMA_BASE_URL"
    "OLLAMA_API_KEY"
    "WEAVIATE_URL"
)

# Get currently active environment variables
get_active_params() {
    local active_params=()
    for param in "${ALL_PARAM_KEYS[@]}"; do
        # Use eval to safely check if environment variable is set
        if eval "[[ -n \"\${${param}:-}\" ]]" 2>/dev/null; then
            active_params+=("$param")
        fi
    done
    printf '%s\n' "${active_params[@]}" | sort
}

# Show configuration summary
show_config_summary() {
    local total_params=${#ALL_PARAM_KEYS[@]}
    local active_params=($(get_active_params))
    local active_count=${#active_params[@]}
    local usage_percentage=$((active_count * 100 / total_params))

    echo
    echo "======================================================================"
    echo "                    AskMe RAG API Startup Configuration"
    echo "======================================================================"
    echo
    echo "📊 Parameter Statistics:"
    echo "   • Total available parameters: $total_params"
    echo "   • Currently enabled parameters: $active_count"
    echo "   • Parameter utilization: $usage_percentage%"
    echo
    echo "🚀 Core Configuration:"
    echo "   • Vector database: $VECTOR_BACKEND"
    echo "   • API service: $HOST:$PORT"
    echo "   • Worker processes: $WORKERS"
    echo "   • Hot reload: $RELOAD"
    echo "   • Log level: $LOG_LEVEL"
    echo
    echo "🤖 AI Model Configuration:"
    echo "   • Ollama model: $OLLAMA_MODEL"
    echo "   • Ollama endpoint: $OLLAMA_ENDPOINT"
    echo "   • Ragas model: $RAGAS_MODEL"
    echo "   • Cohere reranking: $ENABLE_COHERE"
    echo
    echo "💾 Database Configuration:"
    echo "   • Milvus host: $MILVUS_HOST:$MILVUS_PORT"
    echo "   • Skip heavy init: $SKIP_HEAVY_INIT"
    echo
    echo "🔧 Active Environment Variables ($active_count/$total_params):"
    echo "   ┌─────────────────────────────────────────────────────────────────────┐"
    for param in "${active_params[@]}"; do
        local value
        # Use eval to safely get environment variable value
        value=$(eval "echo \"\${${param}:-}\"" 2>/dev/null || echo "")
        # Truncate overly long values
        if [[ ${#value} -gt 40 ]]; then
            value="${value:0:37}..."
        fi
        printf "   │ %-35s = %-30s │\n" "$param" "$value"
    done
    echo "   └─────────────────────────────────────────────────────────────────────┘"
    echo
    echo "💡 Tip: Use --help to see all available configuration options"
    echo "======================================================================"
    echo
}

# Check dependencies
check_dependencies() {
    local missing_deps=()

    # Check uv
    if ! command -v uv &> /dev/null; then
        missing_deps+=("uv")
    fi

    # Check project directory
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log ERROR "Cannot find pyproject.toml in project root: $PROJECT_ROOT"
        log ERROR "Please ensure you are running this script from the repository root directory"
        return 1
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log ERROR "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi

    return 0
}

# Main function
main() {
    cd "$PROJECT_ROOT"

    # Show configuration summary
    show_config_summary

    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "Dry-run mode: showing configuration only, not starting service"
        exit 0
    fi

    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi

    log INFO "Starting AskMe RAG API server..."
    log INFO "Project directory: $PROJECT_ROOT"
    log INFO "Using environment variables: $(get_active_params | wc -l)/${#ALL_PARAM_KEYS[@]}"

    # Build uvicorn command
    local uvicorn_args=(
        "askme.api.main:app"
        "--host" "$HOST"
        "--port" "$PORT"
    )

    if [[ "$RELOAD" == "true" ]]; then
        uvicorn_args+=("--reload")
    fi

    if [[ "$WORKERS" -gt 1 && "$RELOAD" != "true" ]]; then
        uvicorn_args+=("--workers" "$WORKERS")
    elif [[ "$WORKERS" -gt 1 && "$RELOAD" == "true" ]]; then
        log WARNING "Hot reload mode does not support multiple workers, using single process mode"
    fi

    # Start service
    log INFO "Executing command: uv run uvicorn ${uvicorn_args[*]}"
    exec uv run uvicorn "${uvicorn_args[@]}"
}

# Signal handling
trap 'log INFO "Received interrupt signal, exiting..."; exit 130' INT TERM

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
