#!/bin/bash

# AskMe RAG API Stop Script
# Safely and thoroughly stop all related services and processes

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

AskMe RAG API stop script - safely stop all related processes

Options:
  -h, --help              Show this help information
  -f, --force             Force stop (use SIGKILL)
  -p, --port PORT         Specify API port to stop (default: 8080)
  -a, --all               Stop all askme related processes
  -t, --timeout SECONDS   Timeout for graceful shutdown (default: 30 seconds)
  -v, --verbose           Verbose output
  --dry-run               Show processes to be stopped, do not execute

Examples:
  $SCRIPT_NAME                    # Stop default port (8080) API service
  $SCRIPT_NAME -p 8081            # Stop service on specified port
  $SCRIPT_NAME --all              # Stop all askme related processes
  $SCRIPT_NAME --force            # Force stop
  $SCRIPT_NAME --dry-run          # Show processes to be stopped

EOF
}

# Default configuration
DEFAULT_PORT=8080
DEFAULT_FORCE=false
DEFAULT_ALL=false
DEFAULT_TIMEOUT=30
DEFAULT_VERBOSE=false
DEFAULT_DRY_RUN=false

# Parse command line arguments
PORT="$DEFAULT_PORT"
FORCE="$DEFAULT_FORCE"
ALL="$DEFAULT_ALL"
TIMEOUT="$DEFAULT_TIMEOUT"
VERBOSE="$DEFAULT_VERBOSE"
DRY_RUN="$DEFAULT_DRY_RUN"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -a|--all)
            ALL=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
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

if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT" -lt 1 ]]; then
    log ERROR "Invalid timeout: $TIMEOUT"
    exit 1
fi

# Verbose logging function
verbose_log() {
    if [[ "$VERBOSE" == "true" ]]; then
        log "VERBOSE" "$@"
    fi
}

# Find processes function
find_askme_processes() {
    local processes=()

    # Find uvicorn askme processes
    while IFS= read -r line; do
        [[ -n "$line" ]] && processes+=("$line")
    done < <(pgrep -f "uvicorn.*askme\.api\.main" 2>/dev/null || true)

    # Find processes on specified port
    if [[ "$ALL" != "true" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && processes+=("$line")
        done < <(lsof -ti tcp:$PORT 2>/dev/null || true)
    fi

    # Find Python processes containing askme
    while IFS= read -r line; do
        [[ -n "$line" ]] && processes+=("$line")
    done < <(pgrep -f "python.*askme" 2>/dev/null || true)

    # Find uv run related processes
    while IFS= read -r line; do
        [[ -n "$line" ]] && processes+=("$line")
    done < <(pgrep -f "uv run.*askme" 2>/dev/null || true)

    # Deduplicate and return
    printf '%s\n' "${processes[@]}" | sort -n | uniq 2>/dev/null || true
}

# Get process detailed information
get_process_info() {
    local pid=$1
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        local cmd=$(ps -p "$pid" -o command= 2>/dev/null || echo "Unknown")
        local user=$(ps -p "$pid" -o user= 2>/dev/null || echo "Unknown")
        local cpu=$(ps -p "$pid" -o pcpu= 2>/dev/null || echo "0.0")
        local mem=$(ps -p "$pid" -o pmem= 2>/dev/null || echo "0.0")
        echo "PID: $pid, User: $user, CPU: ${cpu}%, MEM: ${mem}%, CMD: $cmd"
    fi
}

# Check if process exists
process_exists() {
    local pid=$1
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# Graceful stop process
graceful_stop() {
    local pid=$1
    local process_name=$2

    if ! process_exists "$pid"; then
        verbose_log "Process $pid ($process_name) does not exist"
        return 0
    fi

    verbose_log "Sending SIGTERM to process $pid ($process_name)"
    kill -TERM "$pid" 2>/dev/null || {
        verbose_log "Cannot send SIGTERM to process $pid, may have already exited"
        return 0
    }

    # Wait for process to exit
    local count=0
    while [[ $count -lt $TIMEOUT ]] && process_exists "$pid"; do
        sleep 1
        ((count++))
        if [[ $((count % 5)) -eq 0 ]]; then
            verbose_log "Waiting for process $pid to exit... ($count/${TIMEOUT}s)"
        fi
    done

    if process_exists "$pid"; then
        log WARNING "Process $pid ($process_name) did not exit gracefully within ${TIMEOUT}s"
        return 1
    else
        verbose_log "Process $pid ($process_name) exited successfully"
        return 0
    fi
}

# Force stop process
force_stop() {
    local pid=$1
    local process_name=$2

    if ! process_exists "$pid"; then
        verbose_log "Process $pid ($process_name) does not exist"
        return 0
    fi

    log WARNING "Force stopping process $pid ($process_name)"
    kill -KILL "$pid" 2>/dev/null || {
        verbose_log "Cannot force stop process $pid, may have already exited"
        return 0
    }

    sleep 1
    if process_exists "$pid"; then
        log ERROR "Cannot force stop process $pid ($process_name)"
        return 1
    else
        verbose_log "Process $pid ($process_name) force stopped"
        return 0
    fi
}

# Clean up Python cache and temporary files
cleanup_python_cache() {
    verbose_log "Cleaning Python cache files..."

    # Clean __pycache__ directories
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    # Clean .pyc files
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true

    # Clean .pyo files
    find "$PROJECT_ROOT" -name "*.pyo" -delete 2>/dev/null || true

    verbose_log "Python cache cleanup completed"
}

# Clean up log files (optional)
cleanup_logs() {
    verbose_log "Checking log files..."

    local log_dir="$PROJECT_ROOT/logs"
    if [[ -d "$log_dir" ]]; then
        local log_size=$(du -sh "$log_dir" 2>/dev/null | cut -f1 || echo "Unknown")
        verbose_log "Log directory size: $log_size"
        # Do not automatically clean logs, just inform
        verbose_log "To clean logs manually, delete: $log_dir"
    fi
}

# Check port usage
check_port_usage() {
    local port=$1
    local processes

    processes=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$processes" ]]; then
        log WARNING "Port $port is still occupied by the following processes:"
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && log WARNING "  $(get_process_info "$pid")"
        done <<< "$processes"
        return 1
    else
        verbose_log "Port $port is released"
        return 0
    fi
}

# Show target processes
show_target_processes() {
    local processes=($(find_askme_processes))

    echo
    echo "======================================================================"
    echo "                    AskMe RAG API Stop Operation Summary"
    echo "======================================================================"
    echo
    echo "🔍 Scan Configuration:"
    echo "   • Target port: $PORT"
    echo "   • Stop all related processes: $ALL"
    echo "   • Force stop: $FORCE"
    echo "   • Timeout: ${TIMEOUT}s"
    echo "   • Verbose output: $VERBOSE"
    echo

    if [[ ${#processes[@]} -eq 0 ]]; then
        echo "✅ No AskMe related processes found to stop"
        echo
        if [[ "$ALL" != "true" ]]; then
            check_port_usage "$PORT" && echo "✅ Port $PORT is not occupied" || echo "⚠️  Port $PORT is occupied by other processes"
        fi
        echo "======================================================================"
        return 0
    fi

    echo "🎯 Found ${#processes[@]} related processes:"
    echo "   ┌─────────────────────────────────────────────────────────────────────┐"

    for pid in "${processes[@]}"; do
        if [[ -n "$pid" ]] && process_exists "$pid"; then
            local info=$(get_process_info "$pid")
            printf "   │ %-67s │\n" "$info"
        fi
    done

    echo "   └─────────────────────────────────────────────────────────────────────┘"
    echo
    echo "🔧 Stop Strategy:"
    if [[ "$FORCE" == "true" ]]; then
        echo "   • Use SIGKILL to force stop all processes directly"
    else
        echo "   • Send SIGTERM signal for graceful shutdown first"
        echo "   • Wait up to ${TIMEOUT}s, then force stop if processes do not exit"
    fi
    echo
    echo "🧹 Cleanup Operations:"
    echo "   • Clean Python cache files (__pycache__, *.pyc)"
    echo "   • Check port occupation status"
    echo "   • Check log file status"
    echo "======================================================================"
    echo
}

# Main stop function
stop_processes() {
    local processes=($(find_askme_processes))
    local stopped_count=0
    local failed_count=0

    if [[ ${#processes[@]} -eq 0 ]]; then
        log INFO "No processes found to stop"
        return 0
    fi

    log INFO "Starting to stop ${#processes[@]} processes..."

    for pid in "${processes[@]}"; do
        if [[ -z "$pid" ]] || ! process_exists "$pid"; then
            continue
        fi

        local cmd=$(ps -p "$pid" -o command= 2>/dev/null | cut -c1-50 || echo "Unknown")
        local process_name="$cmd"

        if [[ "$FORCE" == "true" ]]; then
            if force_stop "$pid" "$process_name"; then
                ((stopped_count++))
                log INFO "✅ Process $pid force stopped"
            else
                ((failed_count++))
                log ERROR "❌ Cannot stop process $pid"
            fi
        else
            if graceful_stop "$pid" "$process_name"; then
                ((stopped_count++))
                log INFO "✅ Process $pid stopped gracefully"
            else
                log WARNING "⚠️  Process $pid graceful stop failed, trying force stop..."
                if force_stop "$pid" "$process_name"; then
                    ((stopped_count++))
                    log INFO "✅ Process $pid force stopped"
                else
                    ((failed_count++))
                    log ERROR "❌ Cannot stop process $pid"
                fi
            fi
        fi
    done

    log INFO "Stop operation completed: successfully stopped $stopped_count processes"
    if [[ $failed_count -gt 0 ]]; then
        log WARNING "$failed_count processes failed to stop"
        return 1
    fi

    return 0
}

# Verify stop result
verify_stop_result() {
    local remaining_processes=($(find_askme_processes))

    if [[ ${#remaining_processes[@]} -gt 0 ]]; then
        log WARNING "Still have ${#remaining_processes[@]} related processes running:"
        for pid in "${remaining_processes[@]}"; do
            [[ -n "$pid" ]] && process_exists "$pid" && log WARNING "  $(get_process_info "$pid")"
        done
        return 1
    fi

    # Check port occupation
    if [[ "$ALL" != "true" ]]; then
        if ! check_port_usage "$PORT"; then
            return 1
        fi
    fi

    log INFO "✅ All target processes have been successfully stopped"
    return 0
}

# Main function
main() {
    cd "$PROJECT_ROOT"

    # Show operation summary
    show_target_processes

    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "Dry-run mode: showing processes to be stopped, not executing operation"
        exit 0
    fi

    # Confirm operation (unless force mode or no processes to stop)
    local processes=($(find_askme_processes))
    if [[ ${#processes[@]} -gt 0 && "$FORCE" != "true" && -t 0 ]]; then
        echo -n "Confirm stopping the above processes? [y/N] "
        read -r confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            log INFO "Operation cancelled"
            exit 0
        fi
        echo
    fi

    log INFO "Starting stop operation..."

    # Stop processes
    if stop_processes; then
        # Verify result
        sleep 2  # Wait a bit for processes to fully exit
        if verify_stop_result; then
            log INFO "🎉 All processes stopped successfully"
        else
            log WARNING "⚠️  Some processes may still be running, please check above warnings"
        fi
    else
        log ERROR "❌ Stop operation encountered problems"
        exit 1
    fi

    # Cleanup operations
    if [[ "$DRY_RUN" != "true" ]]; then
        cleanup_python_cache
        cleanup_logs
    fi

    log INFO "✅ Stop operation completed"
}

# Signal handling
trap 'log INFO "Received interrupt signal, exiting..."; exit 130' INT TERM

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
