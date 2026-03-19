#!/usr/bin/env python3
"""
SnapKV Prefill Overhead Analysis
Compares standard attention operations vs SnapKV-specific overhead
"""

import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_csv(filepath: str) -> List[Tuple[str, str, float]]:
    """Parse CSV and return (op_name, kernel_name, duration_ms)"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                op_name = row[0].strip()
                kernel_name = row[1].strip()
                try:
                    duration = float(row[2].strip())
                    data.append((op_name, kernel_name, duration))
                except ValueError:
                    continue
    return data

def categorize_operations(data: List[Tuple[str, str, float]]) -> Dict[str, List[float]]:
    """Categorize operations into standard vs SnapKV-specific"""
    categories = defaultdict(list)
    
    for op_name, kernel_name, duration in data:
        # Extract layer number
        layer_match = re.search(r'-(\d+)$', op_name)
        
        # Standard operations (exist in naive too)
        if op_name.startswith('attn_norm-'):
            categories['standard_attn_norm'].append(duration)
        elif op_name.startswith('Qcur-'):
            categories['standard_qcur'].append(duration)
        elif op_name.startswith('Vcur-'):
            categories['standard_vcur'].append(duration)
        elif op_name.startswith('Kcur-'):
            categories['standard_kcur'].append(duration)
        elif 'cache_k_l' in op_name and 'view)' in op_name and 'permuted' not in op_name:
            # Initial cache write (standard)
            categories['standard_cache_k_write'].append(duration)
        elif 'cache_v_l' in op_name and 'reshaped) (view)' in op_name and 'view) (view)' not in op_name:
            # Initial cache write (standard)
            categories['standard_cache_v_write'].append(duration)
        elif op_name.startswith('kq_soft_max-') and '(view)' not in op_name:
            categories['standard_kq_softmax'].append(duration)
        elif op_name.startswith('kqv_out-'):
            categories['standard_kqv_out'].append(duration)
        elif op_name.startswith('attn_out-'):
            categories['standard_attn_out'].append(duration)
        elif op_name.startswith('ffn_inp-'):
            categories['standard_ffn_inp'].append(duration)
        elif op_name.startswith('ffn_norm-'):
            categories['standard_ffn_norm'].append(duration)
        elif op_name.startswith('ffn_gate-'):
            categories['standard_ffn_gate'].append(duration)
        elif op_name.startswith('ffn_up-'):
            categories['standard_ffn_up'].append(duration)
        elif op_name.startswith('ffn_swiglu-'):
            categories['standard_ffn_swiglu'].append(duration)
        elif op_name.startswith('ffn_out-'):
            categories['standard_ffn_out'].append(duration)
        elif op_name.startswith('l_out-'):
            categories['standard_l_out'].append(duration)
        elif op_name == 'result_norm':
            categories['standard_result_norm'].append(duration)
        elif op_name == 'result_output':
            categories['standard_result_output'].append(duration)
        
        # SnapKV-specific operations
        elif op_name.startswith('snapkv_kqv-'):
            # Full causal attention for importance scoring
            categories['snapkv_full_kqv'].append(duration)
        elif 'kq_soft_max-' in op_name and '(view) (view) (permuted) (cont)' in op_name:
            # Copy softmax weights for importance
            categories['snapkv_kq_copy'].append(duration)
        elif op_name.startswith('snapkv_importance-') and 'reshaped' not in op_name:
            # Sum over observation window
            categories['snapkv_importance_sum'].append(duration)
        elif 'snapkv_importance-' in op_name and 'reshaped' in op_name:
            # Reshape for GQA reduction
            categories['snapkv_importance_reshape'].append(duration)
        elif op_name.startswith('snapkv_imp_kv-'):
            # GQA reduction
            categories['snapkv_imp_gqa_reduce'].append(duration)
        elif 'node_' in op_name and 'argsort' in kernel_name:
            # Sort importance scores
            categories['snapkv_argsort'].append(duration)
        elif 'node_' in op_name and 'get_rows_f16' in kernel_name:
            # Select K past
            categories['snapkv_k_get_rows'].append(duration)
        elif 'cache_k_l' in op_name and '(view) (permuted) (view) (copy)' in op_name:
            # K window copy
            categories['snapkv_k_win_copy'].append(duration)
        elif op_name.startswith('snapkv_K_packed-'):
            # Concat K_sel_past + K_win
            categories['snapkv_k_concat'].append(duration)
        elif 'cache_k_l' in op_name and '(view) (permuted) (view)' in op_name and 'copy' not in op_name:
            # Pack K into cache
            categories['snapkv_k_pack_write'].append(duration)
        elif 'node_' in op_name and 'get_rows_f32' in kernel_name:
            # Select V past
            categories['snapkv_v_get_rows'].append(duration)
        elif op_name.startswith('snapkv_V_packed-') and 'permuted' not in op_name:
            # Concat V_sel_past + V_win
            categories['snapkv_v_concat'].append(duration)
        elif 'snapkv_V_packed-' in op_name and '(permuted) (cont)' in op_name:
            # Permute V for transposed layout
            categories['snapkv_v_permute'].append(duration)
        elif 'cache_v_l' in op_name and '(view) (view) (view)' in op_name:
            # Pack V into cache (transposed)
            categories['snapkv_v_pack_write'].append(duration)
        elif 'node_' in op_name and 'mul_mm' in kernel_name and 'kq' in kernel_name:
            # Standard kq (if any remains separate)
            categories['standard_kq'].append(duration)
    
    return categories

def compute_stats(categories: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean, sum, and count for each category"""
    stats = {}
    for cat, durations in categories.items():
        if durations:
            stats[cat] = {
                'mean': sum(durations) / len(durations),
                'sum': sum(durations),
                'count': len(durations)
            }
        else:
            stats[cat] = {'mean': 0.0, 'sum': 0.0, 'count': 0}
    return stats

def print_analysis(stats: Dict[str, Dict[str, float]]):
    """Print formatted analysis"""
    
    # Group by category
    standard_ops = {k: v for k, v in stats.items() if k.startswith('standard_')}
    snapkv_ops = {k: v for k, v in stats.items() if k.startswith('snapkv_')}
    
    print("=" * 80)
    print("SnapKV Prefill Overhead Analysis (125 tokens, 32 layers)")
    print("=" * 80)
    print()
    
    # Standard operations
    print("━━━ Standard Operations (exist in naive) ━━━")
    print(f"{'Operation':<35} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
    print("-" * 80)
    
    standard_total = 0.0
    for op, stat in sorted(standard_ops.items()):
        name = op.replace('standard_', '').replace('_', ' ').title()
        print(f"{name:<35} {stat['mean']:>10.3f}   {stat['sum']:>10.3f}   {stat['count']:>6.0f}")
        standard_total += stat['sum']
    
    print("-" * 80)
    print(f"{'Standard Total':<35} {'':<12} {standard_total:>10.3f}")
    print()
    
    # SnapKV-specific operations
    print("━━━ SnapKV-Specific Operations (compression overhead) ━━━")
    print(f"{'Operation':<35} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
    print("-" * 80)
    
    # Group by stage
    importance_ops = ['snapkv_full_kqv', 'snapkv_kq_copy', 'snapkv_importance_sum', 
                      'snapkv_importance_reshape', 'snapkv_imp_gqa_reduce', 'snapkv_argsort']
    k_pack_ops = ['snapkv_k_get_rows', 'snapkv_k_win_copy', 'snapkv_k_concat', 'snapkv_k_pack_write']
    v_pack_ops = ['snapkv_v_get_rows', 'snapkv_v_concat', 'snapkv_v_permute', 'snapkv_v_pack_write']
    
    print("  [Importance Scoring Stage]")
    importance_total = 0.0
    for op in importance_ops:
        if op in snapkv_ops:
            stat = snapkv_ops[op]
            name = '    ' + op.replace('snapkv_', '').replace('_', ' ').title()
            print(f"{name:<35} {stat['mean']:>10.3f}   {stat['sum']:>10.3f}   {stat['count']:>6.0f}")
            importance_total += stat['sum']
    print(f"    {'Subtotal':<31} {'':<12} {importance_total:>10.3f}")
    print()
    
    print("  [K Packing Stage]")
    k_pack_total = 0.0
    for op in k_pack_ops:
        if op in snapkv_ops:
            stat = snapkv_ops[op]
            name = '    ' + op.replace('snapkv_', '').replace('_', ' ').title()
            print(f"{name:<35} {stat['mean']:>10.3f}   {stat['sum']:>10.3f}   {stat['count']:>6.0f}")
            k_pack_total += stat['sum']
    print(f"    {'Subtotal':<31} {'':<12} {k_pack_total:>10.3f}")
    print()
    
    print("  [V Packing Stage]")
    v_pack_total = 0.0
    for op in v_pack_ops:
        if op in snapkv_ops:
            stat = snapkv_ops[op]
            name = '    ' + op.replace('snapkv_', '').replace('_', ' ').title()
            print(f"{name:<35} {stat['mean']:>10.3f}   {stat['sum']:>10.3f}   {stat['count']:>6.0f}")
            v_pack_total += stat['sum']
    print(f"    {'Subtotal':<31} {'':<12} {v_pack_total:>10.3f}")
    print()
    
    snapkv_total = importance_total + k_pack_total + v_pack_total
    print("-" * 80)
    print(f"{'SnapKV Overhead Total':<35} {'':<12} {snapkv_total:>10.3f}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    total_time = standard_total + snapkv_total
    print(f"Standard operations:       {standard_total:>10.3f} ms ({standard_total/total_time*100:>5.1f}%)")
    print(f"SnapKV overhead:           {snapkv_total:>10.3f} ms ({snapkv_total/total_time*100:>5.1f}%)")
    print(f"  - Importance scoring:    {importance_total:>10.3f} ms ({importance_total/snapkv_total*100:>5.1f}% of overhead)")
    print(f"  - K packing:             {k_pack_total:>10.3f} ms ({k_pack_total/snapkv_total*100:>5.1f}% of overhead)")
    print(f"  - V packing:             {v_pack_total:>10.3f} ms ({v_pack_total/snapkv_total*100:>5.1f}% of overhead)")
    print(f"{'─'*35}")
    print(f"Total prefill time:        {total_time:>10.3f} ms")
    print()
    
    # Per-layer breakdown
    if snapkv_ops:
        print("Per-layer SnapKV overhead breakdown (average):")
        print(f"  Importance scoring:  {importance_total/32:>8.3f} ms/layer")
        print(f"  K packing:           {k_pack_total/32:>8.3f} ms/layer")
        print(f"  V packing:           {v_pack_total/32:>8.3f} ms/layer")
        print(f"  Total overhead:      {snapkv_total/32:>8.3f} ms/layer")

def main():
    csv_file = '/home/chokwans99/dev/llm/specinfer.cpp/tmp_result/snap_kv_03140809.csv'
    
    print("Loading and parsing CSV...")
    data = parse_csv(csv_file)
    print(f"Loaded {len(data)} operations")
    print()
    
    print("Categorizing operations...")
    categories = categorize_operations(data)
    print(f"Found {len(categories)} operation categories")
    print()
    
    print("Computing statistics...")
    stats = compute_stats(categories)
    print()
    
    print_analysis(stats)

if __name__ == '__main__':
    main()
