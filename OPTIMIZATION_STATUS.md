# QNN Dynamic EAGLE Optimization Status

Last updated: 2026-04-26 23:30 KST

## Loop Result

- Scope: one code optimization loop for `examples/speculative-eagle-dynamic-qnn/speculative-eagle-dynamic-qnn.cpp`.
- Fixed dynamic policy: `--selector-top-p 3e-6`.
- Attempted optimization: reuse padded selector device ids in reduced LM head projector setup.
- Result: failed validation; dynamic latency regressed.
- Action taken: source edit reverted. No tracked source changes are retained.
- Device state: restored to a build 41 (`59a2fbb`) dynamic binary candidate because the exact original build 40 (`9cfee3a`) device binary was overwritten and not recoverable from the available worktrees.

## Key Logs

| Run | Log |
| --- | --- |
| Static baseline | `adb_static_baseline_20260426_224117_KST.log` |
| Dynamic baseline | `adb_dynamic_baseline_20260426_224117_KST.log` |
| Post-change dynamic | `dynamic_postchange_20260426_230419.log` |
| Reverted source, current build dynamic | `dynamic_revert_verify_top_p_3e-6_20260426_231247.log` |
| Static rerun | `static_rerun_20260426_230911.log` |
| Restored 59a2fbb dynamic | `dynamic_restore_59a2fbb_20260426_232433.log` |

## Metrics Summary

| Metric | Original dynamic b40 | Post-change b44 | Restored b41 |
| --- | ---: | ---: | ---: |
| Decode | 9.88 t/s | 7.12 t/s | 9.56 t/s |
| Decode latency | 101.19 ms/tok | 140.43 ms/tok | 104.65 ms/tok |
| Avg draft phase | 93.185 ms | 144.553 ms | 103.746 ms |
| Avg verification | 170.721 ms | 221.709 ms | 169.194 ms |
| Main Selector | 38.327 ms/round | 73.291 ms/round | 47.586 ms/round |
| Selector Post-QNN | 23.584 ms/round | 56.616 ms/round | 30.101 ms/round |
| Reduced Projector Init | 13.943 ms/round | 33.808 ms/round | 18.656 ms/round |

## Next Candidate

Next loop should focus on one bottleneck only: selector post-QNN / reduced projector initialization in `build_round_selection_from_selector_result()` and `ReducedLmHeadProjector::initialize()`.
