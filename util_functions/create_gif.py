import imageio
import os
import numpy as np

def create_gif():
    images = []
    # path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste/youtube_vis_470/fox/20"
    path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste/tokencut_seg_clip_470/dog/14"
    output_path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste"
    filenames = sorted(os.listdir(path))
    print(f'filenames: {filenames}')
    for filename in filenames:
        if filename[-4:] != "json":
            images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave(os.path.join(output_path, "dog14_tokencut.gif"), images, duration=0.15)

def create_gif_segmented():
    images = []
    # path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste/youtube_vis_470/fox/20"
    path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste/for_gifs/segmented_frames/fox/20"
    output_path = "/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/project-004/AnimateDiff/SDI_Paste/gifs/segmented_gifs"
    filenames = sorted(os.listdir(path))
    print(f'filenames: {filenames} here')
    for filename in filenames:
        if filename[-4:] != "json":
            # place image onto a blank canvas so that gif appears uniform for each frame
            # canvas = np.ones([512,512,3], dtype=np.float32)

            images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave(os.path.join(output_path, "fox20_tokencut.gif"), images, duration=0.15)

create_gif_segmented()