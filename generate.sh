#!/bin/bash
# need to fix:

#echo "Adding $SCRIPT_DIR to PYTHONPATH"
#
#export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
#echo "Current PYTHONPATH: $PYTHONPATH"
#echo "Running module..."

# Generate dynamic object instances using AnimateDiff

#python  -m animatediff.scripts.animate_sdipaste \
#        --config animatediff/configs/prompts/5-RealisticVision.yaml \
#        --num_scenes 5

python -m TokenCut.unsupervised_saliency_detection.get_saliency_sdipaste \
        --sigma-spatial 16 \
        --sigma-luma 16 \
        --sigma-chroma 8 \
        --vit-arch small \
        --patch-size 16 \
        --min_clip 21 \
        --min_area 0.05 \
        --max_area 0.95 \
        --tolerance 1 \
        --img_folder_path ./animatediff/samples \
        --out_dir ./TokenCut/segmented_samples \
        --out_file ./TokenCut/tokencut_YTVIS.json
