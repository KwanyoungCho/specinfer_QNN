# for i in 16 24 40 48 56 64 128; do
#     mkdir -p run_$i
#     for j in {1..10}; do
#         adb shell "
#         export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
#         cd \$LD_LIBRARY_PATH && \
#         ./llama-simple-qnn \
#         --qnn \
#         -n 1 \
#         --multi-context \
#         --ctx-dir artifacts_prefill_$i \
#         --backend-so libQnnHtp.so \
#         --system-so libQnnSystem.so \
#         --tokenizer ../../gguf/llama3-8b-q4-0.gguf \
#         --params ctx_out/params.json \
#         --log-level 1 \
#         -f ../../gguf/prompt_1.txt" 2>&1 | tee run_$i/$j.log
#     done
# done

mkdir -p run_32
for j in {6..10}; do
    adb shell "
        export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/executorch/QNN_test && \
        cd \$LD_LIBRARY_PATH && \
        ./llama-simple-qnn \
        --qnn \
        -n 1 \
        --multi-context \
        --ctx-dir ctx_out \
        --backend-so libQnnHtp.so \
        --system-so libQnnSystem.so \
        --tokenizer ../../gguf/llama3-8b-q4-0.gguf \
        --params ctx_out/params.json \
        --log-level 1 \
        -f ../../gguf/prompt_1.txt" 2>&1 | tee run_32/$j.log
done