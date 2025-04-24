import sys
sys.path.append('./model')
import dino # model

import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os
import clip

from shutil import copyfile
# import PIL.Image as Image
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch
import json
import gc
# Image transformation applied to all images
ToTensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_tokencut_binary_map(img_pth, backbone,patch_size, tau) :
    I = Image.open(img_pth).convert('RGB')
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0).cuda()
    feat = backbone(tensor)[0]

    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
    return bipartition, eigvec

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)

def get_largest_connect_component(img):
    # print(f'cv2: {len(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))}')
    # exit()
    # _ , contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'contours: {len(contours)}')
    # print(f'_: {len(_)}')
    # exit()

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) >= 1:
        max_idx = np.argmax(area)
        img2=np.zeros_like(img)
        cv2.fillPoly(img2, [contours[max_idx]], 1)
        return img2
    else:
        return img

def mask_segment_compose(org, mask, mask_color = [173, 216, 230]) :

    # mask_fg = mask > 0.5
    # print(f'mask fg: {mask_fg.shape}')
    rgb = np.copy(org)
    # print(f'rgb: {rgb}')
    # rgb = np.expand_dims(rgb, axis=3) # make rgba
    rgb_shape = rgb.shape

    # rgba = np.ones([rgb_shape[0], rgb_shape[1], rgb_shape[2]+1], dtype=np.uint8) *255
    # rgba[:,:,:3] = rgb
    # return Image.fromarray(rgb)


    # print(f'rgb: {rgb.shape} {rgba}')
    # print(f'mask only rgb: {mask_only_img.shape}')
    # rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    mask_fg=(mask>0.5).astype('uint8')
    mask_fg = np.expand_dims(mask_fg, 2)

    seg_im = rgb * mask_fg
    # print(f'mask fg: {seg_im.shape} {rgb.shape} {mask_fg.shape}')
    # exit()
    # if np.sum(mask_fg) == 0:
    #     return None, None
    # mask_fg=get_largest_connect_component(mask_fg)
    # seg_mask_ = np.where(mask_fg)
    # # print(f'seg_mask: {seg_mask_}')
    # y_min,y_max,x_min,x_max = np.min(seg_mask_[0]), np.max(seg_mask_[0]), np.min(seg_mask_[1]), np.max(seg_mask_[1])
    # if y_max<=y_min or x_max<=x_min:
    #     return None, None
    # # img_RGBA[:,:,3:]*=seg_mask
    # rgba[:,:,3:]*=mask_fg
    # rgba=rgba[y_min:y_max+1,x_min:x_max+1]
    # # print(f'rgba: {rgba.shape} {rgba[:5,:5]}')
    # # exit()

    return Image.fromarray(seg_im), mask_fg


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, help='output directory')

parser.add_argument('--out-file', type=str, help='output json file')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')


parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

parser.add_argument('--img_folder_path', type=str, default=None, help='image folder visualization')

parser.add_argument('--min_clip', type=float, default=25)
parser.add_argument('--min_area', type=float, default=0.0)
parser.add_argument('--max_area', type=float, default=1.0)
parser.add_argument('--tolerance', type=float, default=1)


args = parser.parse_args()
print (args)

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

# rank = torch.multiprocessing.current_process()._identity[0] - 1
# print("init process GPU:", rank)
clip_model, preprocess = clip.load("ViT-L/14", device=0)
n_px = 224
preprocess = transforms.Compose([
    transforms.Resize(n_px, interpolation=BICUBIC),
    transforms.CenterCrop(n_px),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# exit()

#    resume_path = './model/dino_vitbase16_pretrain.pth' if args.patch_size == 16 else './model/dino_vitbase8_pretrain.pth'

#    feat_dim = 768
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#
#else :
#    resume_path = './model/dino_deitsmall16_pretrain.pth' if args.patch_size == 16 else './model/dino_deitsmall8_pretrain.pth'
#    feat_dim = 384
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)


msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print (msg)
backbone.eval()
backbone.cuda()

if args.dataset == 'ECSSD' :
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'

elif args.dataset == 'DUTS' :
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'

elif args.dataset == 'DUT' :
    args.img_dir = '../datasets/DUT_OMRON/img'
    args.gt_dir = '../datasets/DUT_OMRON/gt'

elif args.dataset is None :
    args.gt_dir = None


print(args.dataset)

if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

# if args.img_path is not None:
#     args.nb_vis = 1
#     img_list = [args.img_path]
# else:
#     img_list = sorted(os.listdir(args.img_dir))

# if args.img_folder_path is not None:



count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
class_folders = sorted(os.listdir(args.img_folder_path))
result = {}
print(f'Segmenting Class: {class_folders}')
for cid, folder in enumerate(class_folders):
    text=f'a photo of {folder}'
    # print(f'text: {text}')

    # if cid < 3:
    #     continue
    print(f'--------ClassID:{cid} FolderID:{folder}---------------------')
    folder_path = os.path.join(args.img_folder_path, folder)
    if not os.path.isdir(folder_path):
        continue
    sequences = sorted(os.listdir(folder_path))
    out_folder = os.path.join(args.out_dir, folder)
    # args.out_dir = out_folder_path
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    sequence_files_collect = {}
    for seq_id ,sequence in enumerate(sequences):
        sequence_path = os.path.join(folder_path, sequence)
        out_folder_sequence_path = os.path.join(out_folder, sequence)
        if not os.path.exists(out_folder_sequence_path):
            os.mkdir(out_folder_sequence_path)
        img_list = sorted(os.listdir(sequence_path))
        img_list = [os.path.join(sequence_path, im) for im in img_list]
        # print(f'img list: {img_list} {folder}')
        img_file_collect = []
        # if seq_id > 2:
        #     break
        for img_name in tqdm(img_list) :
            print(f'img name: {img_name}')
            # if args.img_path is not None:
            # print(f'img_name: {img_name[-3:]}')
            # exit()
            # filter only images:
            if img_name[-3:] != 'png':
                continue
            img_pth = img_name
            img_name = img_name.split("/")[-1]
            # print(img_name)
            # exit()
            # else:
            #     img_pth = os.path.join(args.img_dir, img_name)

            bipartition, eigvec = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)
            # mask_lost.append(bipartition)

            output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(binary_solver).cuda()
            if metric.IoU(mask1, mask2) < 0.5:
                binary_solver = binary_solver * -1


            # mask_bfs.append(output_solver)

            # if args.gt_dir is not None :
            #     mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
            #     gt.append(mask_gt)

            # if count_vis != args.nb_vis :
            # print(f'args.out_dir: {out_folder_sequence_path}, img_name: {img_name}')
            # out_name = os.path.join(args.out_dir, img_name)
            # out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut.jpg'))
            # out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
            out_name = os.path.join(out_folder_sequence_path, img_name)
            # print(f'out_name: {out_name}')
            out_lost = os.path.join(out_folder_sequence_path, img_name.replace('.jpg', '_tokencut.jpg'))
            out_bfs = os.path.join(out_folder_sequence_path, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
            out_segmented = os.path.join(out_folder_sequence_path, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
            #out_eigvec = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_eigvec.jpg'))

            # copyfile(img_pth, out_name)
            org = np.array(Image.open(img_pth).convert('RGB'))

            # print(f'out name: {out_name} {img_pth}')
            # exit()

            #plt.imsave(fname=out_eigvec, arr=eigvec, cmap='cividis')
            # mask_color_compose(org, bipartition).save(out_lost)
            # mask_color_compose(org, binary_solver).save(out_bfs)

            # print(f'out seg: {out_segmented}')
            res, mask = mask_segment_compose(org, binary_solver)
            res.save(out_segmented)
            continue
            if res is not None:
                # get clip scores:
                # img = transforms.ToTensor()(Image.open(img_pth).convert('RGB'))
                img = transforms.ToTensor()(org)
                mask = mask.transpose(2, 0, 1)
                mask_area = np.sum(mask, axis=(0, 1, 2)) / mask.shape[1] / mask.shape[2]
                print(f'mask area: {mask_area}')

                # print(f'imgs: {img.shape}')
                # mask = seg_model.forward(imgs, prompt=text)
                # print(f'mask: {mask.shape}')
                # exit()
                # mask_im = torch.from_numpy(mask > 128).float().unsqueeze(1)
                wb_im=img*mask+torch.ones_like(img)*(1-mask)

                # exit()

                # text_feature = clip.tokenize(text).cuda()
                # _, logits_per_text = clip_model(preprocess(wb_im.unsqueeze(0)).cuda(), text_feature)
                # logits_per_text=logits_per_text.view(-1).cpu().tolist()[0]
                # this_bar = min(args.min_clip, np.max(logits_per_text) - args.tolerance)
                # print(f'logtis: {logits_per_text} {this_bar}')
                # if logits_per_text < this_bar or mask_area < args.min_area or mask_area > args.max_area:
                #     print(f'continue')
                #     continue
                # else:
                # res.save(out_segmented)
                wb_im = np.array(wb_im)
                print(f'wb_im: {wb_im.shape}')
                wb_im = Image.fromarray(wb_im)
                wb_im.save(out_segmented)
                # img_file_collect.append('*'+out_segmented)

            # if args.gt_dir is not None :
            #     # out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            #     out_gt = os.path.join(out_folder_sequence_path, img_name.replace('.jpg', '_gt.jpg'))
            #     mask_color_compose(org, mask_gt).save(out_gt)

        # seq_files = {sequence: img_file_collect}
        sequence_files_collect[sequence] = img_file_collect
        # print(f'seq file: {sequence_files_collect}')

    result[cid] = sequence_files_collect
    # output_file = f"/home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_youtube_vis_instance_pools_{folder}.json"

    # with open(output_file, 'w') as f:
    #     json.dump(result, f)

    # if cid == 2:
    #     print(f'result: {result.keys()}')
    #     exit()

    # print(f'result: {result}')

# output_file = f"/home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_youtube_vis_instance_pools_clip_final_150.json"
# output_file = f"/home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_youtube_vis_instance_pools_{folder}.json"

# with open(output_file, 'w') as f:
#     json.dump(result, f)

# sequence_files_collect.clear()
# result.clear()


if args.gt_dir is not None and args.img_path is None:
    print ('TokenCut evaluation:')
    print (metric.metrics(mask_lost, gt))
    print ('\n')

    print ('TokenCut + bilateral solver evaluation:')
    print (metric.metrics(mask_bfs, gt))
    print ('\n')
    print ('\n')
