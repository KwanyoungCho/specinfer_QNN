#!/usr/bin/env python3
"""Aggregate GSM8K benchmark results from multiple runs"""

import sys
import re
import csv
from pathlib import Path

def parse_output_file(filepath):
    """Parse a single output file and extract metrics"""
    metrics = {
        'sample_id': None,
        'prefill_tokens': None,
        'prefill_ms': None,
        'prefill_tps': None,
        'decode_tokens': None,
        'decode_ms': None,
        'decode_tps': None,
        'decode_lat_ms': None,
        'draft_len': None,
        'accept_len': None,
        'accept_ratio': None,
        'avg_draft_lat_ms': None,
        'avg_verify_lat_ms': None,
        'avg_td_ms': None,
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract sample number from filename
        match = re.search(r'output_(\d+)\.txt', filepath)
        if match:
            metrics['sample_id'] = int(match.group(1))
        
        # Parse performance summary section
        # Example:
        #   Prefill           :   123 tokens |   1234.56 ms |    98.76 t/s
        #   Decode            :   100 tokens |   5678.90 ms |    17.62 t/s
        #   Decode latency    :              |     56.79 ms/tok
        
        prefill_match = re.search(r'Prefill\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s', content)
        if prefill_match:
            metrics['prefill_tokens'] = int(prefill_match.group(1))
            metrics['prefill_ms'] = float(prefill_match.group(2))
            metrics['prefill_tps'] = float(prefill_match.group(3))
        
        decode_match = re.search(r'Decode\s+:\s+(\d+)\s+tokens\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+t/s', content)
        if decode_match:
            metrics['decode_tokens'] = int(decode_match.group(1))
            metrics['decode_ms'] = float(decode_match.group(2))
            metrics['decode_tps'] = float(decode_match.group(3))
        
        decode_lat_match = re.search(r'Decode latency\s+:\s+\|\s+([\d.]+)\s+ms/tok', content)
        if decode_lat_match:
            metrics['decode_lat_ms'] = float(decode_lat_match.group(1))
        
        draft_len_match = re.search(r'Draft length\s+:\s+([\d.]+)', content)
        if draft_len_match:
            metrics['draft_len'] = float(draft_len_match.group(1))
        
        accept_len_match = re.search(r'Avg accept length\s+:\s+([\d.]+)', content)
        if accept_len_match:
            metrics['accept_len'] = float(accept_len_match.group(1))
        
        accept_ratio_match = re.search(r'Accept ratio\s+:\s+([\d.]+)%', content)
        if accept_ratio_match:
            metrics['accept_ratio'] = float(accept_ratio_match.group(1))
        
        avg_draft_match = re.search(r'Avg draft phase\s+:\s+([\d.]+)\s+ms', content)
        if avg_draft_match:
            metrics['avg_draft_lat_ms'] = float(avg_draft_match.group(1))
        
        avg_verify_match = re.search(r'Avg verification\s+:\s+([\d.]+)\s+ms', content)
        if avg_verify_match:
            metrics['avg_verify_lat_ms'] = float(avg_verify_match.group(1))
        
        avg_td_match = re.search(r'Avg T_d \(1-tok dft\)\s+:\s+([\d.]+)\s+ms', content)
        if avg_td_match:
            metrics['avg_td_ms'] = float(avg_td_match.group(1))
        
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
    
    return metrics

def main():
    # Accept output directory as command line argument
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("gsm8k_results")
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Run the benchmark script first: ./run_gsm8k_benchmark.sh")
        return
    
    # Collect all output files
    output_files = sorted(output_dir.glob("output_*.txt"))
    
    if not output_files:
        print(f"Error: No output files found in {output_dir}")
        return
    
    print(f"Found {len(output_files)} output files")
    print("Parsing results...")
    
    # Parse all files
    all_metrics = []
    for filepath in output_files:
        metrics = parse_output_file(str(filepath))
        if metrics['sample_id'] is not None:
            all_metrics.append(metrics)
    
    # Sort by sample_id
    all_metrics.sort(key=lambda x: x['sample_id'])
    
    # Calculate averages
    avg_metrics = {}
    numeric_fields = [k for k in all_metrics[0].keys() if k != 'sample_id']
    
    for field in numeric_fields:
        values = [m[field] for m in all_metrics if m[field] is not None]
        if values:
            avg_metrics[field] = sum(values) / len(values)
        else:
            avg_metrics[field] = None
    
    # Write CSV
    csv_file = f"{output_dir.name}_results.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['sample_id'] + numeric_fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow(metrics)
        
        # Add average row
        avg_row = {'sample_id': 'AVERAGE'}
        avg_row.update(avg_metrics)
        writer.writerow(avg_row)
    
    print(f"\n✓ Results saved to: {csv_file}")
    print(f"\nProcessed {len(all_metrics)} samples")
    
    # Print summary
    print("\n" + "="*60)
    print("  Summary Statistics")
    print("="*60)
    if avg_metrics['prefill_ms']:
        print(f"  Avg Prefill:        {avg_metrics['prefill_ms']:.2f} ms  |  {avg_metrics['prefill_tps']:.2f} t/s")
    if avg_metrics['decode_ms']:
        print(f"  Avg Decode:         {avg_metrics['decode_ms']:.2f} ms  |  {avg_metrics['decode_tps']:.2f} t/s")
    if avg_metrics['decode_lat_ms']:
        print(f"  Avg Decode Latency: {avg_metrics['decode_lat_ms']:.2f} ms/tok")
    print("-"*60)
    if avg_metrics['draft_len']:
        print(f"  Avg Draft Length:   {avg_metrics['draft_len']:.3f}")
    if avg_metrics['accept_len']:
        print(f"  Avg Accept Length:  {avg_metrics['accept_len']:.3f}")
    if avg_metrics['accept_ratio']:
        print(f"  Avg Accept Ratio:   {avg_metrics['accept_ratio']:.2f}%")
    print("-"*60)
    if avg_metrics['avg_draft_lat_ms']:
        print(f"  Avg Draft Phase:    {avg_metrics['avg_draft_lat_ms']:.3f} ms")
    if avg_metrics['avg_verify_lat_ms']:
        print(f"  Avg Verification:   {avg_metrics['avg_verify_lat_ms']:.3f} ms")
    if avg_metrics['avg_td_ms']:
        print(f"  Avg T_d:            {avg_metrics['avg_td_ms']:.3f} ms")
    print("="*60)

if __name__ == "__main__":
    main()
