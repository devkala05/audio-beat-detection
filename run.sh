#!/bin/bash

mkdir -p data/output
> logs/run_logs.txt

echo "🛠️  Building..."
make clean && make

echo "🚀 Running beat detection on input files..."
for f in data/input/*.wav; do
    echo "Processing: $f"
    ./beat_detect "$f" >> logs/run_logs.txt
done

echo "✅ All done. Output in data/output/ and logs/run_logs.txt"
