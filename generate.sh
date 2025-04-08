#!/bin/bash

# Set the desired number of sequences for each object class by changing --num_scenes below:

python  -m animatediff.scripts.animate_sdipaste \
        --config animatediff/configs/prompts/5-RealisticVision.yaml \
        --num_scenes 5

