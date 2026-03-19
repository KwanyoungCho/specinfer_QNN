#!/bin/bash

OUTPUT_FILE="prefill_latency_results.txt"

echo "len latency" > $OUTPUT_FILE

for dir in run_*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    len=$(echo $dir | sed 's/run_//;s/\///')
    
    sum=0
    count=0
    
    for log in "$dir"/{1..10}.log; do
        if [ -f "$log" ]; then
            latency=$(grep "Prefill" "$log" | grep -oP '│\s+\K[0-9.]+(?=\s+│\s+[0-9.]+\s+║)')
            if [ ! -z "$latency" ]; then
                sum=$(echo "$sum + $latency" | bc)
                count=$((count + 1))
            fi
        fi
    done
    
    if [ $count -gt 0 ]; then
        avg=$(echo "scale=2; $sum / $count" | bc)
        echo "$len $avg" >> $OUTPUT_FILE
    fi
done

sort -n -o $OUTPUT_FILE.tmp $OUTPUT_FILE
head -n 1 $OUTPUT_FILE > $OUTPUT_FILE.sorted
tail -n +2 $OUTPUT_FILE.tmp >> $OUTPUT_FILE.sorted
mv $OUTPUT_FILE.sorted $OUTPUT_FILE
rm -f $OUTPUT_FILE.tmp

echo "Results saved to $OUTPUT_FILE"
cat $OUTPUT_FILE
