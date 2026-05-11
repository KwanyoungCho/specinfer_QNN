#!/usr/bin/env bash
# Run baseline and approx (k=1,2,3,5) over GSM8K prompts on device.
# Outputs:
#   - per-run full logs:  $HOST_OUT/{config}_p{XX}.log
#   - per-run TBT CSVs:   pulled from device tbt_out/gsm8k/{config}_p{XX}.csv to $HOST_OUT/
set -e

DEVICE_DIR=/data/local/tmp/chokwans99/executorch/QNN_test
HOST_OUT=/home/chokwans99/dev/llm/specinfer.cpp/tmp/approx_results/gsm8k
mkdir -p "$HOST_OUT"

COMMON_ARGS="-m ../../gguf/vicuna_q4_0_output4.gguf -md ../../gguf/EAGLE_q4_0.gguf -c 0 -fa off --top-p 1.0 --min-p 0.0 --temp 0.0 --draft-max 500 --draft-min 1 --n-predict 200 -ngl 40 -ngld 20 -np 200 -s 1234 -kvu --top-k 4 --n-depth 5 --rerank-k 6"

for P in $(seq -w 1 20); do
    PROMPT_FILE="../../executorch/QNN_test/gsm8k_prompts/prompt_${P}.txt"
    echo "=========== prompt $P ==========="

    # baseline (exact greedy)
    CFG=baseline
    echo "-- $CFG"
    adb shell "export LD_LIBRARY_PATH=$DEVICE_DIR && cd \$LD_LIBRARY_PATH && \
        ./llama-speculative-eagle-2 $COMMON_ARGS \
        -f gsm8k_prompts/prompt_${P}.txt \
        --tbt-csv tbt_out/gsm8k/${CFG}_p${P}.csv 2>&1" \
        > "$HOST_OUT/${CFG}_p${P}.log" 2>&1

    # approx k=1,2,3,5
    for K in 1 2 3 5; do
        CFG=k${K}
        echo "-- approx $CFG"
        adb shell "export LD_LIBRARY_PATH=$DEVICE_DIR && cd \$LD_LIBRARY_PATH && \
            ./llama-speculative-eagle-2-approx $COMMON_ARGS \
            -f gsm8k_prompts/prompt_${P}.txt \
            --accept-top-k $K \
            --tbt-csv tbt_out/gsm8k/${CFG}_p${P}.csv 2>&1" \
            > "$HOST_OUT/${CFG}_p${P}.log" 2>&1
    done
done

echo
echo "pulling TBT CSVs from device..."
adb pull $DEVICE_DIR/tbt_out/gsm8k "$HOST_OUT/csv_raw" 2>&1 | tail -5
echo "done."
