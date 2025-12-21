#!/bin/bash

# Add .gitkeep to all empty directories in the experiments folder
find /home/vinh/Documents/code/FedAvgLS/experiment/FATE-LLM/experiments -type d -empty -not -path "*/\.*" -exec touch {}/.gitkeep \;

echo "Added .gitkeep to all empty directories"
