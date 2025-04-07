#!/bin/bash


python get_saliency_bushfire.py \
        --sigma-spatial 16 \
        --sigma-luma 16 \
        --sigma-chroma 8 \
        --vit-arch small \
        --patch-size 16 \
        --img_folder_path /home/users/u6566739/project-004/AnimateDiff/Bushfire/ \
        --out-dir /home/users/u6566739/project-004/AnimateDiff/Bushfire/Segments\

#        --min_clip 21 \
#        --min_area 0.05 \
#        --max_area 0.95 \
#        --tolerance 1 \
#        --out-file output_path="/home/users/u6566739/project-004/AnimateDiff/Bushfire/tokencut_bushfire.json"





#        --out-dir /home/users/u6566739/project-004/XPaste/output/youtube_vis_seg/tokencut
#        --img_folder_path /home/users/u6566739/project-004/XPaste/output/youtube_vis/ \
#        --img_folder_path ../examples/0000.png \
