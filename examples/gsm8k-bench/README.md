# Benchmark Suite (MTBench & GSM8K)

Benchmarking tools for evaluating LLM performance on MTBench and GSM8K datasets.

## Overview

Instead of modifying the binary, this approach:
- Uses existing binaries (EAGLE speculative or autoregressive)
- Runs it multiple times with different prompts (MTBench or GSM8K)
- Each run is completely independent (no state leakage)
- Aggregates results into a CSV file
- Uses Llama chat template format for prompts

## Datasets

### MTBench
- 20 diverse questions covering: cooking, writing, coding, math, reasoning, Fermi estimation
- Multi-domain evaluation for general capabilities

### GSM8K
- 20 grade school math word problems
- Focused evaluation on mathematical reasoning

## Benchmark Versions

### Speculative Decoding (EAGLE)
- **NPU**: `llama-speculative-eagle-qnn` with `run_mtbench_benchmark.sh`
- **GPU**: `llama-speculative-eagle` with `run_mtbench_benchmark_gpu.sh`

### Autoregressive (Baseline)
- **NPU**: `llama-simple-qnn` with `run_mtbench_benchmark_ar_npu.sh`
- **GPU**: `llama-cli` with `run_mtbench_benchmark_ar_gpu.sh`

### 1. Generate Prompt Files

**For MTBench:**
```bash
cd examples/gsm8k-bench
python3 generate_mtbench_prompts.py
```
Creates: `mtbench_prompts/` with 20 prompt files

**For GSM8K:**
```bash
cd examples/gsm8k-bench
python3 generate_gsm8k_prompts.py
```
Creates: `gsm8k_prompts/` with 20 prompt files

All prompts use Llama chat template format:
```
<|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe...
<|eot_id|><|start_header_id|>user<|end_header_id|>
[Question]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### 2. Run Benchmark

**Transfer prompts and scripts to device:**
```bash
# Transfer prompt directories
adb push mtbench_prompts/ /data/local/tmp/chokwans99/executorch/QNN_test/
adb push gsm8k_prompts/ /data/local/tmp/chokwans99/executorch/QNN_test/

# Transfer benchmark scripts
adb push run_mtbench_benchmark.sh /data/local/tmp/chokwans99/executorch/QNN_test/
adb push run_mtbench_benchmark_gpu.sh /data/local/tmp/chokwans99/executorch/QNN_test/
adb push run_mtbench_benchmark_ar_npu.sh /data/local/tmp/chokwans99/executorch/QNN_test/
adb push run_mtbench_benchmark_ar_gpu.sh /data/local/tmp/chokwans99/executorch/QNN_test/
```

#### Option A: NPU Version (EAGLE-QNN)

**On device via adb shell:**
```bash
cd /data/local/tmp/chokwans99/executorch/QNN_test
chmod +x run_mtbench_benchmark.sh

# Run MTBench
./run_mtbench_benchmark.sh mtbench

# Or run GSM8K
./run_mtbench_benchmark.sh gsm8k
```

#### Option B: GPU Version (Speculative-EAGLE)

**On device via adb shell:**
```bash
cd /data/local/tmp/chokwans99/executorch/QNN_test
chmod +x run_mtbench_benchmark_gpu.sh

# Run MTBench
./run_mtbench_benchmark_gpu.sh mtbench

# Or run GSM8K
./run_mtbench_benchmark_gpu.sh gsm8k
```

#### Option C: Autoregressive NPU Version (llama-simple-qnn)

**On device via adb shell:**
```bash
cd /data/local/tmp/chokwans99/executorch/QNN_test
chmod +x run_mtbench_benchmark_ar_npu.sh

# Run MTBench
./run_mtbench_benchmark_ar_npu.sh mtbench

# Or run GSM8K
./run_mtbench_benchmark_ar_npu.sh gsm8k
```

#### Option D: Autoregressive GPU Version (llama-cli)

**On device via adb shell:**
```bash
cd /data/local/tmp/chokwans99/executorch/QNN_test
chmod +x run_mtbench_benchmark_ar_gpu.sh

# Run MTBench
./run_mtbench_benchmark_ar_gpu.sh mtbench

# Or run GSM8K
./run_mtbench_benchmark_ar_gpu.sh gsm8k
```

### 3. Aggregate Results

**Pull results from device and aggregate:**

#### Speculative EAGLE (NPU):
```bash
# MTBench
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/mtbench_results/ .
python3 aggregate_results.py mtbench_results

# GSM8K
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/gsm8k_results/ .
python3 aggregate_results.py gsm8k_results
```

#### Speculative EAGLE (GPU):
```bash
# MTBench
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/mtbench_results_gpu/ .
python3 aggregate_results.py mtbench_results_gpu

# GSM8K
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/gsm8k_results_gpu/ .
python3 aggregate_results.py gsm8k_results_gpu
```

#### Autoregressive (NPU):
```bash
# MTBench
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/mtbench_results_ar_npu/ .
python3 aggregate_results_ar.py mtbench_results_ar_npu

# GSM8K
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/gsm8k_results_ar_npu/ .
python3 aggregate_results_ar.py gsm8k_results_ar_npu
```

#### Autoregressive (GPU):
```bash
# MTBench
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/mtbench_results_ar_gpu/ .
python3 aggregate_results_ar.py mtbench_results_ar_gpu

# GSM8K
adb pull /data/local/tmp/chokwans99/executorch/QNN_test/gsm8k_results_ar_gpu/ .
python3 aggregate_results_ar.py gsm8k_results_ar_gpu
```

**CSV Contents:**
- Individual sample metrics
- Average across all samples

## Output Metrics

### Speculative EAGLE Metrics
- **Prefill**: tokens, latency (ms), throughput (t/s)
- **Decode**: tokens, latency (ms), throughput (t/s), per-token latency
- **Draft length**: average number of tokens drafted per step
- **Accept length**: average number of accepted tokens per step
- **Accept ratio**: percentage of drafted tokens accepted
- **Avg draft phase**: average draft phase latency (ms)
- **Avg verification**: average verification latency (ms)
- **Avg T_d**: average single-token draft time (ms)

### Autoregressive Metrics
- **Prefill**: tokens, latency (ms), throughput (t/s)
- **Decode**: tokens, latency (ms), throughput (t/s)

## Files

### Prompt Generation Scripts
- `generate_mtbench_prompts.py` - Generate 20 MTBench prompt files with Llama chat template
- `generate_gsm8k_prompts.py` - Generate 20 GSM8K prompt files with Llama chat template

#### Speculative EAGLE
- `run_mtbench_benchmark.sh [mtbench|gsm8k]` - NPU benchmark (llama-speculative-eagle-qnn)
- `run_mtbench_benchmark_gpu.sh [mtbench|gsm8k]` - GPU benchmark (llama-speculative-eagle)
- `aggregate_results.py <output_dir>` - Parse EAGLE outputs and create CSV

#### Autoregressive Baseline
- `run_mtbench_benchmark_ar_npu.sh [mtbench|gsm8k]` - NPU autoregressive (llama-simple-qnn)
- `run_mtbench_benchmark_ar_gpu.sh [mtbench|gsm8k]` - GPU autoregressive (llama-cli)
- `aggregate_results_ar.py <output_dir>` - Parse autoregressive outputs and create CSV

### Directories
- `mtbench_prompts/` - MTBench prompt and metadata files
- `gsm8k_prompts/` - GSM8K prompt and metadata files
- `{dataset}_results/` - EAGLE NPU output files
- `{dataset}_results_gpu/` - EAGLE GPU output files
- `{dataset}_results_ar_npu/` - Autoregressive NPU output files
- `{dataset}_results_ar_gpu/` - Autoregressive GPU output files
- `*_results.csv` - Final aggregated results

## Dataset Details

### MTBench Categories

Questions from various categories:
- **Cooking**: Practical cooking instructions
- **Writing**: Creative and persuasive writing tasks
- **Coding**: Programming challenges (Fibonacci, regex, etc.)
- **Math**: Mathematical problem solving
- **General**: Life skills and advice
- **Reasoning**: Complex reasoning and analysis
- **Fermi**: Estimation problems

### GSM8K Dataset

- 20 grade school math word problems
- Tests arithmetic reasoning and multi-step problem solving
- Problems involve real-world scenarios (shopping, time, distance, etc.)

## Advantages

✅ **No binary modification** - use existing binaries  
✅ **Complete independence** - each run starts fresh, no KV cache issues  
✅ **Easy debugging** - can inspect individual run outputs  
✅ **Flexible** - easy to modify arguments or add more samples  
✅ **Accurate** - no state leakage between samples  
✅ **Diverse evaluation** - covers multiple task types (coding, math, writing, reasoning)
