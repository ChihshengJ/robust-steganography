#!/bin/bash

# List of IDs
ids=("1" "2" "3") # Modify this list as needed

# Max parallel processes
num_processes=3

# Run watermark_generator.py in parallel
parallel --keep-order --line-buffer --joblog parallel_log.log -j $num_processes python watermarked_text_generator.py ::: "${ids[@]}"

echo "All watermarking jobs completed."
