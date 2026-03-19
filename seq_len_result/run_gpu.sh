#!/bin/bash

# Length 16
mkdir -p GPU_result/run_16
echo "Running experiments for length 16..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_16/$j.log
done

# Length 24
mkdir -p GPU_result/run_24
echo "Running experiments for length 24..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_24/$j.log
done

# Length 32
mkdir -p GPU_result/run_32
echo "Running experiments for length 32..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_32/$j.log
done

# Length 40
mkdir -p GPU_result/run_40
echo "Running experiments for length 40..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_40/$j.log
done

# Length 48
mkdir -p GPU_result/run_48
echo "Running experiments for length 48..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_48/$j.log
done

# Length 56
mkdir -p GPU_result/run_56
echo "Running experiments for length 56..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_56/$j.log
done

# Length 64
mkdir -p GPU_result/run_64
echo "Running experiments for length 64..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_64/$j.log
done

# Length 128
mkdir -p GPU_result/run_128
echo "Running experiments for length 128..."
for j in {1..10}; do
    echo "  Trial $j/10"
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
    cd \$LD_LIBRARY_PATH && \
    ./llama-cli \
    -n 1 -ngl 40 \
    -m ../../gguf/llama-38i_q4_0_output4.gguf \
    -p '[dummy] [dummy] [dummy] [dummy] [dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy] [prompt dummy]' \
    -fa off -no-cnv" 2>&1 | tee GPU_result/run_128/$j.log
done

echo "All experiments completed!"