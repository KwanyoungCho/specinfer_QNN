#!/bin/bash
# ==========================================================================
#  Spec-Bench Benchmark Runner
#  Runs a speculative-decoding binary on all Spec-Bench prompts and saves
#  per-prompt output for later aggregation.
#
#  Usage:
#    ./run_spec_bench.sh                          # defaults (eagle-2-qnn, all prompts)
#    ./run_spec_bench.sh --binary ./llama-speculative-eagle
#    ./run_spec_bench.sh --prompt-dir spec_bench_prompts --output-dir my_results
#    ./run_spec_bench.sh --start 50 --end 100     # run subset (prompts 50-100)
#    ./run_spec_bench.sh --dry-run                # show what would run, don't execute
#
#  Before running, generate prompts with:
#    python3 generate_spec_bench_prompts.py
# ==========================================================================
set -e

# --------------- defaults (override via flags) ---------------
BINARY="./llama-speculative-eagle-2-qnn"
PROMPT_DIR="spec_bench_prompts"
OUTPUT_DIR="spec_bench_results"
START=1
END=0          # 0 = all
DRY_RUN=false
N_PREDICT=512  # max tokens to generate per prompt

# Extra args forwarded verbatim to the binary.
# Modify this array for your model/hardware setup.
EXTRA_ARGS=()

# --------------- parse CLI flags ---------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)      BINARY="$2";      shift 2 ;;
        --prompt-dir)  PROMPT_DIR="$2";  shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --start)       START="$2";       shift 2 ;;
        --end)         END="$2";         shift 2 ;;
        --n-predict)   N_PREDICT="$2";   shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        --)            shift; EXTRA_ARGS+=("$@"); break ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --------------- validation ---------------
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    echo "Build your binary first or pass --binary /path/to/binary"
    exit 1
fi

if [ ! -d "$PROMPT_DIR" ]; then
    echo "Error: Prompt directory not found: $PROMPT_DIR"
    echo "Run first: python3 generate_spec_bench_prompts.py"
    exit 1
fi

# Count available prompts
NUM_PROMPTS=$(ls "$PROMPT_DIR"/prompt_*.txt 2>/dev/null | wc -l)
if [ "$NUM_PROMPTS" -eq 0 ]; then
    echo "Error: No prompt_*.txt files found in $PROMPT_DIR"
    exit 1
fi

if [ "$END" -eq 0 ] || [ "$END" -gt "$NUM_PROMPTS" ]; then
    END=$NUM_PROMPTS
fi

mkdir -p "$OUTPUT_DIR"

# --------------- header ---------------
echo "=========================================="
echo "  Spec-Bench Benchmark"
echo "=========================================="
echo "  Binary:     $BINARY"
echo "  Prompts:    $PROMPT_DIR ($NUM_PROMPTS total)"
echo "  Range:      $START .. $END"
echo "  Output:     $OUTPUT_DIR"
echo "  n_predict:  $N_PREDICT"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra args: ${EXTRA_ARGS[*]}"
fi
if $DRY_RUN; then
    echo "  *** DRY RUN — nothing will be executed ***"
fi
echo "=========================================="
echo ""

# --------------- benchmark loop ---------------
COMPLETED=0
FAILED=0
TOTAL=$((END - START + 1))

for i in $(seq "$START" "$END"); do
    NUM=$(printf "%03d" "$i")
    PROMPT_FILE="$PROMPT_DIR/prompt_${NUM}.txt"
    OUTPUT_FILE="$OUTPUT_DIR/output_${NUM}.txt"

    if [ ! -f "$PROMPT_FILE" ]; then
        echo "[$i/$END] SKIP — $PROMPT_FILE not found"
        continue
    fi

    # Show category from meta file if available
    META_FILE="$PROMPT_DIR/meta_${NUM}.txt"
    CATEGORY=""
    if [ -f "$META_FILE" ]; then
        CATEGORY=$(grep "^Category:" "$META_FILE" | head -1 | cut -d: -f2 | xargs)
    fi

    echo -n "[$i/$END] ${CATEGORY:+[$CATEGORY] }prompt_${NUM}... "

    if $DRY_RUN; then
        echo "(dry-run)"
        continue
    fi

    # Run the binary; capture both stdout and stderr
    if "$BINARY" -n "$N_PREDICT" "${EXTRA_ARGS[@]}" -f "$PROMPT_FILE" > "$OUTPUT_FILE" 2>&1; then
        COMPLETED=$((COMPLETED + 1))
        # Extract decode t/s from output for quick feedback
        TPS=$(grep -oP 'Decode\s+:\s+\d+ tokens \|\s+[\d.]+ ms \|\s+\K[\d.]+' "$OUTPUT_FILE" 2>/dev/null || echo "?")
        echo "done (${TPS} t/s)"
    else
        FAILED=$((FAILED + 1))
        echo "FAILED (exit=$?)"
    fi
done

# --------------- summary ---------------
echo ""
echo "=========================================="
echo "  Benchmark Complete"
echo "=========================================="
echo "  Completed: $COMPLETED / $TOTAL"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failed:    $FAILED"
fi
echo "  Results:   $OUTPUT_DIR/"
echo ""
echo "To aggregate results:"
echo "  python3 aggregate_spec_bench.py $OUTPUT_DIR --meta-dir $PROMPT_DIR"
