#!/usr/bin/env python3
"""Aggregate autoregressive benchmark results from multiple runs"""

import sys
import re
import csv
from pathlib import Path

def parse_qnn_output(filepath):
    """Parse QNN (NPU) autoregressive output format"""
    metrics = {
        'sample': filepath.stem.replace('output_', ''),
        'prefill_tokens': 0,
        'prefill_latency_ms': 0.0,
        'prefill_tps': 0.0,
        'decode_tokens': 0,
        'decode_latency_ms': 0.0,
        'decode_tps': 0.0,
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Parse QNN Performance Summary table format
        # ║  Prefill    │  %6d  │  %12.2f  │  %16.2f  ║
        prefill_match = re.search(r'║\s+Prefill\s+│\s+(\d+)\s+│\s+([\d.]+)\s+│\s+([\d.]+)\s+║', content)
        if prefill_match:
            metrics['prefill_tokens'] = int(prefill_match.group(1))
            metrics['prefill_latency_ms'] = float(prefill_match.group(2))
            metrics['prefill_tps'] = float(prefill_match.group(3))
        
        # ║  Decode     │  %6d  │  %12.2f  │  %16.2f  ║
        decode_match = re.search(r'║\s+Decode\s+│\s+(\d+)\s+│\s+([\d.]+)\s+│\s+([\d.]+)\s+║', content)
        if decode_match:
            metrics['decode_tokens'] = int(decode_match.group(1))
            metrics['decode_latency_ms'] = float(decode_match.group(2))
            metrics['decode_tps'] = float(decode_match.group(3))
            
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
    
    return metrics

def parse_llama_output(filepath):
    """Parse llama-cli (GPU) output format with llama_perf_context_print"""
    metrics = {
        'sample': filepath.stem.replace('output_', ''),
        'prefill_tokens': 0,
        'prefill_latency_ms': 0.0,
        'prefill_tps': 0.0,
        'decode_tokens': 0,
        'decode_latency_ms': 0.0,
        'decode_tps': 0.0,
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # llama_perf_context_print outputs lines like:
        # eval time     =    1234.56 ms /    50 tokens (   24.69 ms per token,    40.50 tokens per second)
        # prompt eval time =     987.65 ms /    20 tokens (   49.38 ms per token,    20.25 tokens per second)
        
        # Parse prompt eval (prefill)
        prompt_match = re.search(r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second', content)
        if prompt_match:
            metrics['prefill_latency_ms'] = float(prompt_match.group(1))
            metrics['prefill_tokens'] = int(prompt_match.group(2))
            metrics['prefill_tps'] = float(prompt_match.group(3))
        
        # Parse eval (decode) - match from line start to avoid matching "prompt eval time"
        eval_match = re.search(r'^common_perf_print:\s+eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens).*?([\d.]+)\s*tokens per second', content, re.MULTILINE)
        if eval_match:
            metrics['decode_latency_ms'] = float(eval_match.group(1))
            metrics['decode_tokens'] = int(eval_match.group(2))
            metrics['decode_tps'] = float(eval_match.group(3))
            
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
    
    return metrics

def main():
    # Accept output directory as command line argument
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("mtbench_results_ar_npu")
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Run the benchmark script first")
        return
    
    # Detect format based on directory name
    is_npu = 'npu' in output_dir.name.lower()
    parse_func = parse_qnn_output if is_npu else parse_llama_output
    
    print(f"Parsing results from: {output_dir}")
    print(f"Format: {'QNN (NPU)' if is_npu else 'llama-cli (GPU)'}")
    print()
    
    # Find all output files
    output_files = sorted(output_dir.glob("output_*.txt"))
    
    if not output_files:
        print(f"Error: No output files found in {output_dir}")
        return
    
    print(f"Found {len(output_files)} output files")
    
    # Parse each output file
    results = []
    for output_file in output_files:
        metrics = parse_func(output_file)
        results.append(metrics)
    
    # Calculate averages
    avg_metrics = None
    if results:
        avg_metrics = {
            'sample': 'AVERAGE',
            'prefill_tokens': sum(r['prefill_tokens'] for r in results) / len(results),
            'prefill_latency_ms': sum(r['prefill_latency_ms'] for r in results) / len(results),
            'prefill_tps': sum(r['prefill_tps'] for r in results) / len(results),
            'decode_tokens': sum(r['decode_tokens'] for r in results) / len(results),
            'decode_latency_ms': sum(r['decode_latency_ms'] for r in results) / len(results),
            'decode_tps': sum(r['decode_tps'] for r in results) / len(results),
        }
        results.append(avg_metrics)
    
    # Write CSV
    csv_file = f"{output_dir.name}_results.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'sample',
            'prefill_tokens', 'prefill_latency_ms', 'prefill_tps',
            'decode_tokens', 'decode_latency_ms', 'decode_tps'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n✓ Results written to: {csv_file}")
    
    # Print summary
    if avg_metrics:
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY (AVERAGE)")
        print("="*60)
        print(f"Prefill:  {avg_metrics['prefill_tokens']:.1f} tokens, "
              f"{avg_metrics['prefill_latency_ms']:.2f} ms, "
              f"{avg_metrics['prefill_tps']:.2f} t/s")
        print(f"Decode:   {avg_metrics['decode_tokens']:.1f} tokens, "
              f"{avg_metrics['decode_latency_ms']:.2f} ms, "
              f"{avg_metrics['decode_tps']:.2f} t/s")
        print("="*60)

if __name__ == "__main__":
    main()
