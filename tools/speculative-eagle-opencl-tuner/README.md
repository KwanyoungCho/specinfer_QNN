# speculative-eagle-opencl-tuner

Microbenchmark and work-group search tool for the dynamic EAGLE selector path.

It sweeps OpenCL local work sizes for:

- `kernel_bitonic_sort_step_f32_i32` used by selector `top-k`
- `kernel_bitonic_sort_step_i32` used by ascending gather-id reorder
- the reduced LM-head `Q4_0` gather path, using the legacy kernel as a baseline plus a tuner-only structural gather kernel
- a `Q4_0` matvec comparison between packed dense rows through `mul_mat_Ab_Bi_8x4`, direct subgroup-indexed rows, `Ab_Bi_8x4` / `Ab_Bi_4x4` indexed tiles, and no-split / local-B experimental variants

Search policy:

- 1D tuning uses `auto` as a baseline plus power-of-two candidates `1, 2, 4, ..., 1024`
- Gather tuning keeps the legacy `64x2` kernel as a baseline and searches tuner-only structural configs over:
  - power-of-two `WG_X`/`WG_Y` pairs up to `1024`, filtered by runtime device and kernel limits
  - `rows/thread = {1, 2, 4, 8}`
  - `k4/thread = {1, 2, 4}`
  - `ids staging = {global, local}`
- `auto` is preserved only as a comparison baseline; the exhaustive tuned search is over the power-of-two candidates
- Indexed `Ab_Bi` local sizes default to power-of-two `WI_M/WI_K` combinations for safer deployment; use `--allow-non-power-local` for exploratory wider sweeps.
- Indexed matvec search also includes no-split `WI_K=1` variants, experimental `8x2` / `8x8` M-tiles, and a local-B staging variant for `--lmhead-batch >= 8`.

The defaults are chosen to mimic the current dynamic path:

- `--n-scores 128256`
- `--src-rows 128256`
- `--hidden-dim 4096`
- `--top-k 64`
- `--gather-rows 512`

Build:

```bash
cmake -S /data/jongjip/specinfer_QNN -B /data/jongjip/specinfer_QNN/build -DGGML_OPENCL=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build /data/jongjip/specinfer_QNN/build --target speculative-eagle-opencl-tuner -j
```

Run:

```bash
/data/jongjip/specinfer_QNN/build/bin/speculative-eagle-opencl-tuner \
  --n-scores 128256 \
  --src-rows 128256 \
  --hidden-dim 4096 \
  --top-k 64 \
  --gather-rows 512 \
  --iters 5 \
  --warmup 1
```

Useful variants:

```bash
# Only search the gather kernel
/data/jongjip/specinfer_QNN/build/bin/speculative-eagle-opencl-tuner --search gather

# Compare packed dense vs direct indexed Q4_0 matvec
/data/jongjip/specinfer_QNN/build/bin/speculative-eagle-opencl-tuner \
  --search indexed \
  --n-scores 128256 \
  --src-rows 128256 \
  --top-k 25102 \
  --gather-rows 35136 \
  --hidden-dim 4096 \
  --lmhead-batch 4

# Exploratory indexed sweep including non-power-of-two Ab_Bi local sizes
/data/jongjip/specinfer_QNN/build/bin/speculative-eagle-opencl-tuner \
  --search indexed \
  --n-scores 128256 \
  --src-rows 128256 \
  --top-k 32768 \
  --gather-rows 32768 \
  --hidden-dim 4096 \
  --lmhead-batch 4 \
  --allow-non-power-local

# Heavier run for stabler numbers
/data/jongjip/specinfer_QNN/build/bin/speculative-eagle-opencl-tuner --iters 20 --warmup 3
```

The tool prints a recommendation block for:

- `top-k` and `id-sort` local work sizes
- `gather` work-group plus per-thread tile configuration
- `indexed/dense q4 mv` when `--search indexed` is enabled

For `gather`, the recommendation is no longer limited to a single `local_work_size`;
it now describes the tuned kernel structure as well.

By default the binary can fall back to embedded copies of `argsort.cl` and
`set_rows.cl`, so it can run on a device even when the original source tree is
not present. If you want to override that, pass:

```bash
--kernel-dir /path/to/ggml/src/ggml-opencl/kernels
```
