import torch
use_gpu = torch.cuda.is_available() and torch.cuda.device_count()>0
device = torch.device('cuda:0' if use_gpu else 'cpu')
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import os
from tqdm.auto import tqdm
import cv2

from stereoflow.test import _load_model_and_criterion
from stereoflow.engine import tiled_pred, tiled_pred_tqdm
from stereoflow.datasets_stereo import img_to_tensor, vis_disparity
from stereoflow.datasets_flow import flowToColor
from stereoflow.warp_utils import fwarp_wrapper

tile_overlap=0.7 # recommended value, higher value can be slightly better but slower

model, _, cropsize, with_conf, task, tile_conf_mode = _load_model_and_criterion('stereoflow_models/crocoflow.pth', None, device)

# load MULTIPLE video sequence & handle saving directories
#sequence_root = "datasets_realvideo/DAVIS_SAMPLE"
#sequence_root = "datasets_realvideo/SINTEL_SAMPLE"
#sequence_root = "datasets_realvideo/XVFI/Longer_testset/Type1/TEST01_003_f0433"
sequence_roots = [
    #"datasets_realvideo/SINTEL_SAMPLE/",
    #"datasets_realvideo/DAVIS_SAMPLE/",
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST01_003_f0433",
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST02_045_f0465",
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST03_081_f4833",
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST04_140_f3889",
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST05_158_f0321",
    #"datasets_realvideo/XVFI/Longer_testset/Type2/TEST06_001_f0273",
    #"datasets_realvideo/XVFI/Longer_testset/Type2/TEST07_076_f1889",
    #"datasets_realvideo/XVFI/Longer_testset/Type2/TEST08_079_f0321",
    #"datasets_realvideo/XVFI/Longer_testset/Type2/TEST09_112_f0177",
    #"datasets_realvideo/XVFI/Longer_testset/Type2/TEST10_172_f1905",
    #"datasets_realvideo/XVFI/Longer_testset/Type3/TEST11_078_f4977",
    #"datasets_realvideo/XVFI/Longer_testset/Type3/TEST12_087_f2721",
    #"datasets_realvideo/XVFI/Longer_testset/Type3/TEST13_133_f4593",
    #"datasets_realvideo/XVFI/Longer_testset/Type3/TEST14_146_f1761",
    #"datasets_realvideo/XVFI/Longer_testset/Type3/TEST15_148_f0465",
    "datasets_realvideo/Samsung_dataset/full_res/apo_car_thinline",
    "datasets_realvideo/Samsung_dataset/full_res/Dubai-CityofGold_Highfrequency1_2k",
    "datasets_realvideo/Samsung_dataset/full_res/Dubai-CityofGold_Highfrequency2_2k",
    "datasets_realvideo/Samsung_dataset/full_res/Impossible_Saves"
]
#save_root = "stereoflow_models/crocoflow.pth_debugs"
#save_root = "stereoflow_models/crocoflow.pth_XVFI"
save_root = "stereoflow_models/crocoflow.pth_Samsung"

for sequence_root in tqdm(sequence_roots, desc='processing sequences'):
    image_list_raw = sorted(os.listdir(sequence_root))
    image_list = []
    for i in range(len(image_list_raw)-1):
        image_list += [
            [
                os.path.join(sequence_root, image_list_raw[i]),
                os.path.join(sequence_root, image_list_raw[i+1])
            ]
        ]

    save_dir = os.path.join(save_root, os.path.basename(sequence_root))
    os.makedirs(save_dir, exist_ok=True)
    save_dir_vis = os.path.join(save_root, os.path.basename(sequence_root), 'flow_vis')
    os.makedirs(save_dir_vis, exist_ok=True)
    save_dir_fwarp = os.path.join(save_root, os.path.basename(sequence_root), 'image1_fwarp')
    os.makedirs(save_dir_fwarp, exist_ok=True)
    save_dir_bwarp = os.path.join(save_root, os.path.basename(sequence_root), 'image1_bwarp')
    os.makedirs(save_dir_bwarp, exist_ok=True)
    save_dir_grid = os.path.join(save_root, os.path.basename(sequence_root), 'grid_vis')
    os.makedirs(save_dir_grid, exist_ok=True)

    print()

    # actual execution
    for i in tqdm(range(len(image_list)), desc=f'processing frames in sequence {os.path.basename(sequence_root)}'):
        image1_name = image_list[i][0]
        image2_name = image_list[i][1]

        image1 = np.asarray(Image.open(image1_name))
        image2 = np.asarray(Image.open(image2_name))
        #image1 = Image.open(image1_name).convert('RGB')
        #image2 = Image.open(image2_name).convert('RGB')
        #image1 = np.array(image1).astype(np.uint8)[..., :3]
        #image2 = np.array(image2).astype(np.uint8)[..., :3]
        im1 = img_to_tensor(image1).to(device).unsqueeze(0)
        im2 = img_to_tensor(image2).to(device).unsqueeze(0)
        with torch.inference_mode():
            #pred, _, _ = tiled_pred(model, None, im1, im2, None, conf_mode=tile_conf_mode, overlap=tile_overlap, crop=cropsize, with_conf=with_conf, return_time=False)
            pred, _, _ = tiled_pred_tqdm(model, None, im1, im2, None, conf_mode=tile_conf_mode, overlap=tile_overlap, crop=cropsize, with_conf=with_conf, return_time=False)
        flo = pred.clone().cpu()  # to be used for warping ops
        pred = pred.squeeze(0).permute(1,2,0).cpu().numpy()

        # visualize flow
        flow_vis = flowToColor(pred)
        flow_vis_path = os.path.join(save_dir_vis, os.path.basename(image1_name))
        cv2.imwrite(flow_vis_path, flow_vis[:, :, [2,1,0]])

        # visualize forward warping
        img = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0)
        image1_fwarp, _ = fwarp_wrapper(img=img, flo=flo)
        fwarp_path = os.path.join(save_dir_fwarp, os.path.basename(image1_name))
        cv2.imwrite(fwarp_path, image1_fwarp[:, :, [2,1,0]])

        # grid visualization
        row_1 = np.concatenate([image1, image2], axis=1)
        row_2 = np.concatenate([flow_vis, image1_fwarp], axis=1)
        grid_vis = np.concatenate([row_1, row_2], axis=0)
        grid_vis_path = os.path.join(save_dir_grid, os.path.basename(image1_name))
        cv2.imwrite(grid_vis_path, grid_vis[:, :, [2,1,0]])