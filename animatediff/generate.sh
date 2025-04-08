#!/bin/bash
# need to fix:

#echo "Adding $SCRIPT_DIR to PYTHONPATH"
#
#export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
#echo "Current PYTHONPATH: $PYTHONPATH"
#echo "Running module..."

python  -m scripts.animate_sdipaste \
        --config configs/prompts/5-RealisticVision.yaml \
        --num_scenes 5