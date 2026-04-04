#!/bin/bash
# Benchmark Script for EAGLE-QNN (NPU)
# Runs llama-speculative-eagle-qnn multiple times with different prompts

set -e

# Accept dataset parameter (mtbench or gsm8k)
DATASET="${1:-mtbench}"

if [ "$DATASET" != "mtbench" ] && [ "$DATASET" != "gsm8k" ]; then
    echo "Error: Invalid dataset. Use 'mtbench' or 'gsm8k'"
    echo "Usage: $0 [mtbench|gsm8k]"
    exit 1
fi

# Configuration
BINARY="./llama-speculative-eagle-2-qnn"
PROMPT_DIR="${DATASET}_prompts"
OUTPUT_DIR="${DATASET}_results"
NUM_SAMPLES=20

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    exit 1
fi

# Check if prompt directory exists
if [ ! -d "$PROMPT_DIR" ]; then
    echo "Error: Prompt directory not found: $PROMPT_DIR"
    if [ "$DATASET" = "mtbench" ]; then
        echo "Run: python3 generate_mtbench_prompts.py first"
    else
        echo "Run: python3 generate_gsm8k_prompts.py first"
    fi
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Common arguments (modify as needed)
COMMON_ARGS=(
    --qnn
    -n 512
    --multi-context
    --ctx-dir ctx_out_0316
    --backend-so libQnnHtp.so
    --system-so libQnnSystem.so
    --tokenizer tokenizer.gguf
    --params params.json
    --log-level 1
    -md EAGLE-llama3-q4_0.gguf
    -ngld 20
    --temp 0.0
    --min-p 0.0
    --top-p 1.0
    --draft-max 500
    --draft-min 1
    -np 500
    -kvu
    -s 1234
    --no-mmap
    -fa off
    --deferred_kv
    --top-k 10
    --n-depth 5
    --rerank-k 59
)

echo "=========================================="
echo "  $(echo $DATASET | tr '[:lower:]' '[:upper:]') Benchmark (EAGLE NPU)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Binary: $BINARY"
echo "Samples: $NUM_SAMPLES"
echo "Prompt Dir: $PROMPT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run benchmark for each sample
for i in $(seq 1 $NUM_SAMPLES); do
    SAMPLE_NUM=$(printf "%02d" $i)
    PROMPT_FILE="$PROMPT_DIR/prompt_${SAMPLE_NUM}.txt"
    OUTPUT_FILE="$OUTPUT_DIR/output_${SAMPLE_NUM}.txt"
    
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Warning: Prompt file not found: $PROMPT_FILE"
        continue
    fi
    
    echo "[$i/$NUM_SAMPLES] Running sample $SAMPLE_NUM..."
    echo "  Prompt: $PROMPT_FILE"
    echo "  Output: $OUTPUT_FILE"
    
    # Run the binary and capture output
    $BINARY "${COMMON_ARGS[@]}" -f "$PROMPT_FILE" > "$OUTPUT_FILE" 2>&1 || {
        echo "  ERROR: Failed to run sample $SAMPLE_NUM"
        continue
    }
    
    echo "  ✓ Completed"
    echo ""
done

echo ""
echo "=========================================="
echo "  Benchmark Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To aggregate results, run:"
echo "  python3 aggregate_results.py $OUTPUT_DIR"
