#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Sweep kv-offload-slots from 2 to 31
for slots in {2..31}; do
    echo "Running experiments with --kv-offload-slots ${slots}"
    
    # Run 3 trials for each slot value
    for trial in {1..3}; do
        echo "  Trial ${trial}/3 for slots=${slots}"
        
        # Log file name
        LOG_FILE="results/slots_${slots}_trial_${trial}.log"
        
        # Run the experiment
        adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
        cd \$LD_LIBRARY_PATH && \
        rm -f ./kv_dir/* && ./llama-eagle-kv-offloading \
        -m ../../gguf/llama-38i_q4_0_output4.gguf \
        -f ../../gguf/prompt.txt \
        -md ../../gguf/EAGLE-llama38i_q4_0_output4.gguf -ngld 40 -ngl 40 \
        -c 0 --color --top-k 4 --top-p 1.0 --min-p 0.0 --temp 0.0 \
        --draft-max 25 --draft-min 1 --n-predict 100 -ngl 40 -ngld 40 \
        -np 20 -s 1234 -kvu --no-mmap \
        --kv-offload-dir ./kv_dir --tree-budget 25 --kv-offload-slots ${slots}" > "${LOG_FILE}" 2>&1
        
        echo "  Saved to ${LOG_FILE}"
    done
    
    echo "Completed all trials for slots=${slots}"
    echo "---"
done

echo "All experiments completed!"
echo "Results saved in results/ directory"