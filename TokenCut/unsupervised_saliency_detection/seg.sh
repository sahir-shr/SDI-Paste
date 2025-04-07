#!/bin/bash


python get_saliency_vxpaste.py \
        --sigma-spatial 16 \
        --sigma-luma 16 \
        --sigma-chroma 8 \
        --vit-arch small \
        --patch-size 16 \
        --min_clip 21 \
        --min_area 0.05 \
        --max_area 0.95 \
        --tolerance 1 \
        --img_folder_path /home/users/u6566739/project-004/AnimateDiff/VXpaste/youtube_vis_470/ \
        --out-dir /home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_seg_clip_470 \
        --out-file output_path="/home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_youtube_vis_instance_pools_clip_470.json"






#        --out-dir /home/users/u6566739/project-004/XPaste/output/youtube_vis_seg/tokencut
#        --img_folder_path /home/users/u6566739/project-004/XPaste/output/youtube_vis/ \
#        --img_folder_path ../examples/0000.png \
