#!/usr/bin/env python3
import os
import re
import glob
from collections import defaultdict
import statistics

def parse_log_file(filepath):
    """Parse a single log file and extract metrics."""
    metrics = {}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract common metrics (adjust patterns based on your actual log format)
    patterns = {
        'tokens_per_second': r'(\d+\.?\d*)\s+tokens per second',
        'total_time': r'total time\s*[=:]\s*(\d+\.?\d*)\s*ms',
        'prompt_time': r'prompt eval time\s*[=:]\s*(\d+\.?\d*)\s*ms',
        'eval_time': r'eval time\s*[=:]\s*(\d+\.?\d*)\s*ms',
        'acceptance_rate': r'acceptance rate\s*[=:]\s*(\d+\.?\d*)%?',
        'kv_offload_ratio': r'KV offload ratio\s*[=:]\s*(\d+\.?\d*)%?',
        'avg_latency': r'average latency\s*[=:]\s*(\d+\.?\d*)\s*ms',
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics[metric_name] = float(match.group(1))
    
    return metrics

def main():
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found!")
        return
    
    # Parse all log files
    log_files = glob.glob(os.path.join(results_dir, 'slots_*_trial_*.log'))
    
    if not log_files:
        print(f"No log files found in {results_dir}/")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Group metrics by slot number
    slot_data = defaultdict(list)
    
    for log_file in log_files:
        # Extract slot number from filename
        match = re.search(r'slots_(\d+)_trial_(\d+)\.log', log_file)
        if not match:
            continue
        
        slot_num = int(match.group(1))
        trial_num = int(match.group(2))
        
        metrics = parse_log_file(log_file)
        
        if metrics:
            slot_data[slot_num].append(metrics)
            print(f"Parsed: slots={slot_num}, trial={trial_num}, metrics={len(metrics)}")
    
    if not slot_data:
        print("No metrics extracted from log files!")
        return
    
    # Calculate averages and write summary
    output_file = 'results/summary.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KV Offload Slots Experiment Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Get all metric names
        all_metrics = set()
        for trials in slot_data.values():
            for trial in trials:
                all_metrics.update(trial.keys())
        
        sorted_slots = sorted(slot_data.keys())
        
        for slot_num in sorted_slots:
            trials = slot_data[slot_num]
            f.write(f"\n{'='*60}\n")
            f.write(f"Slots: {slot_num} (Trials: {len(trials)})\n")
            f.write(f"{'='*60}\n")
            
            if not trials:
                f.write("No data available\n")
                continue
            
            # Calculate averages for each metric
            for metric_name in sorted(all_metrics):
                values = [trial[metric_name] for trial in trials if metric_name in trial]
                
                if values:
                    avg = statistics.mean(values)
                    std = statistics.stdev(values) if len(values) > 1 else 0.0
                    f.write(f"  {metric_name:25s}: {avg:10.3f} ± {std:8.3f}\n")
            
            # Individual trial values
            f.write(f"\n  Individual trials:\n")
            for i, trial in enumerate(trials, 1):
                f.write(f"    Trial {i}: {trial}\n")
    
    print(f"\n{'='*80}")
    print(f"Summary written to: {output_file}")
    print(f"{'='*80}\n")
    
    # Also create CSV output
    csv_file = 'results/summary.csv'
    with open(csv_file, 'w') as f:
        # Write header
        metric_names = sorted(all_metrics)
        f.write("slots,trials," + ",".join([f"{m}_avg,{m}_std" for m in metric_names]) + "\n")
        
        # Write data
        for slot_num in sorted_slots:
            trials = slot_data[slot_num]
            row = [str(slot_num), str(len(trials))]
            
            for metric_name in metric_names:
                values = [trial[metric_name] for trial in trials if metric_name in trial]
                if values:
                    avg = statistics.mean(values)
                    std = statistics.stdev(values) if len(values) > 1 else 0.0
                    row.extend([f"{avg:.3f}", f"{std:.3f}"])
                else:
                    row.extend(["", ""])
            
            f.write(",".join(row) + "\n")
    
    print(f"CSV summary written to: {csv_file}\n")

if __name__ == '__main__':
    main()
