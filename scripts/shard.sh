#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_path> <output_path>"
    exit 1
fi

# Get the arguments
MODEL_PATH=$1
OUTPUT_PATH=$2

# Run the Python script
python ./src/shard.py "$MODEL_PATH" "$OUTPUT_PATH"
