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
DEFAULT_OLLAMA_MODEL="qwen3:30b-a3b"
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
declare -A ALL_AVAILABLE_PARAMS=(
    # Core configuration
    ["ASKME_VECTOR_BACKEND"]="Vector database backend"
    ["ASKME_ENABLE_OLLAMA"]="Enable Ollama"
    ["ASKME_ENABLE_COHERE"]="Enable Cohere"
    ["ASKME_SKIP_HEAVY_INIT"]="Skip heavy initialization"
    # Database configuration
    ["ASKME_DATABASE__HOST"]="Database host"
    ["ASKME_DATABASE__PORT"]="Database port"
    ["ASKME_DATABASE__MILVUS__HOST"]="Milvus host"
    ["ASKME_DATABASE__MILVUS__PORT"]="Milvus port"
    ["ASKME_DATABASE__MILVUS__USERNAME"]="Milvus username"
    ["ASKME_DATABASE__MILVUS__PASSWORD"]="Milvus password"
    ["ASKME_DATABASE__MILVUS__SECURE"]="Milvus secure connection"
    ["ASKME_DATABASE__MILVUS__COLLECTION_NAME"]="Milvus collection name"
    ["ASKME_DATABASE__WEAVIATE__URL"]="Weaviate URL"
    ["ASKME_DATABASE__WEAVIATE__API_KEY"]="Weaviate API key"
    ["ASKME_DATABASE__WEAVIATE__CLASS_NAME"]="Weaviate class name"
    ["ASKME_DATABASE__QDRANT__URL"]="Qdrant URL"
    ["ASKME_DATABASE__QDRANT__API_KEY"]="Qdrant API key"
    ["ASKME_DATABASE__QDRANT__COLLECTION_NAME"]="Qdrant collection name"
    # Generation configuration
    ["ASKME_GENERATION__PROVIDER"]="LLM provider"
    ["ASKME_GENERATION__OLLAMA_MODEL"]="Ollama model"
    ["ASKME_GENERATION__OLLAMA_ENDPOINT"]="Ollama endpoint"
    ["ASKME_GENERATION__MODEL_NAME"]="Model name"
    ["ASKME_GENERATION__MAX_TOKENS"]="Max token count"
    ["ASKME_GENERATION__TEMPERATURE"]="Generation temperature"
    ["ASKME_GENERATION__TOP_P"]="Top-p sampling"
    ["ASKME_GENERATION__OPENAI_MODEL"]="OpenAI model"
    ["ASKME_GENERATION__OPENAI_BASE_URL"]="OpenAI endpoint"
    ["ASKME_GENERATION__OPENAI_API_KEY_ENV"]="OpenAI API key env var name"
    # OpenAI/evaluation configuration
    ["OPENAI_BASE_URL"]="OpenAI compatible endpoint"
    ["OPENAI_API_KEY"]="OpenAI API key"
    ["COHERE_API_KEY"]="Cohere API key"
    ["ASKME_RAGAS_LLM_MODEL"]="Ragas LLM model"
    ["ASKME_RAGAS_EMBED_MODEL"]="Ragas embedding model"
    # Embedding configuration
    ["ASKME_EMBEDDING__MODEL"]="Embedding model"
    ["ASKME_EMBEDDING__DIMENSION"]="Embedding dimension"
    ["ASKME_EMBEDDING__BATCH_SIZE"]="Embedding batch size"
    ["ASKME_EMBEDDING__MAX_LENGTH"]="Max input length"
    # Hybrid search configuration
    ["ASKME_HYBRID__MODE"]="Hybrid search mode"
    ["ASKME_HYBRID__ALPHA"]="Alpha fusion parameter"
    ["ASKME_HYBRID__RRF_K"]="RRF fusion parameter"
    ["ASKME_HYBRID__TOPK"]="Initial retrieval count"
    # Reranking configuration
    ["ASKME_RERANK__LOCAL_MODEL"]="Local reranking model"
    ["ASKME_RERANK__TOP_N"]="Reranking output count"
    ["ASKME_RERANK__COHERE_MODEL"]="Cohere reranking model"
    # API configuration
    ["ASKME_API__HOST"]="API host"
    ["ASKME_API__PORT"]="API port"
    ["ASKME_API__WORKERS"]="API worker count"
    ["ASKME_API__RELOAD"]="API hot reload"
    # Query enhancement
    ["ASKME_ENHANCER__HYDE__ENABLED"]="Enable HyDE"
    ["ASKME_ENHANCER__RAG_FUSION__ENABLED"]="Enable RAG-Fusion"
    # Performance configuration
    ["ASKME_PERFORMANCE__BATCH__EMBEDDING_BATCH_SIZE"]="Embedding batch size"
    ["ASKME_PERFORMANCE__BATCH__RERANK_BATCH_SIZE"]="Reranking batch size"
    ["ASKME_PERFORMANCE__TIMEOUTS__RETRIEVAL_TIMEOUT"]="Retrieval timeout"
    # Logging configuration
    ["ASKME_LOG_LEVEL"]="Log level"
    ["ASKME_LOGGING__LEVEL"]="Detailed log level"
    # Special configuration
    ["ASKME_OLLAMA_READ_TIMEOUT"]="Ollama read timeout"
    ["ASKME_OLLAMA_THINKING"]="Ollama thinking mode"
    ["TOKENIZERS_PARALLELISM"]="Tokenizers parallelism"
    ["OLLAMA_BASE_URL"]="Ollama base URL"
    ["OLLAMA_API_KEY"]="Ollama API key"
    ["WEAVIATE_URL"]="Weaviate test URL"
)

# Get currently active environment variables
get_active_params() {
    local active_params=()
    for param in "${!ALL_AVAILABLE_PARAMS[@]}"; do
        if [[ -n "${!param:-}" ]]; then
            active_params+=("$param")
        fi
    done
    printf '%s\n' "${active_params[@]}" | sort
}

# Show configuration summary
show_config_summary() {
    local total_params=${#ALL_AVAILABLE_PARAMS[@]}
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
        local desc="${ALL_AVAILABLE_PARAMS[$param]}"
        local value="${!param}"
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
        log ERROR "Incorrect project root directory: $PROJECT_ROOT"
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
    log INFO "Using environment variables: $(get_active_params | wc -l)/$(echo ${#ALL_AVAILABLE_PARAMS[@]})"

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
