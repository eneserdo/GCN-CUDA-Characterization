#!/bin/bash



# Extract numbers from the "Avg (ns)" column using awk
avg_ns_numbers=$(sudo nsys profile -t nvtx --stats=true main.py | awk 'NR>2 {gsub(/[,.]/,"",$4); print $4}')

# Print the extracted numbers
for number in $avg_ns_numbers; do
  echo "$number"
done
