#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BINARY="${ROOT_DIR}/build/bin/llama-spec-bench-eagle2-qnn"
BENCH_FILE="${ROOT_DIR}/data/spec_bench.jsonl"
RESULTS_DIR=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)      BINARY="$2";      shift 2 ;;
        --bench-file)  BENCH_FILE="$2";  shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        --)            shift; EXTRA_ARGS+=("$@"); break ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f "$BINARY" ]]; then
    echo "Error: binary not found: $BINARY"
    echo "Build first, or pass --binary /path/to/llama-spec-bench-eagle2-qnn"
    exit 1
fi

if [[ ! -f "$BENCH_FILE" ]]; then
    echo "Error: bench file not found: $BENCH_FILE"
    exit 1
fi

CMD=("$BINARY" "--bench-file" "$BENCH_FILE")
if [[ -n "$RESULTS_DIR" ]]; then
    CMD+=("--results-dir" "$RESULTS_DIR")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "=========================================="
echo "  Spec-Bench EAGLE2 QNN"
echo "=========================================="
echo "  Binary:     $BINARY"
echo "  Bench file: $BENCH_FILE"
if [[ -n "$RESULTS_DIR" ]]; then
    echo "  Results:    $RESULTS_DIR"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra args: ${EXTRA_ARGS[*]}"
fi
if $DRY_RUN; then
    echo "  *** DRY RUN - nothing will be executed ***"
fi
echo "=========================================="

if $DRY_RUN; then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
