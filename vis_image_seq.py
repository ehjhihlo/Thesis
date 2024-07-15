import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
# from lib.hrnet.gen_kpts import gen_video_kpts_realtime
from lib.hrnet.gen_pose_2d import gen_vis_realtime
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy


sys.path.append(os.getcwd())
# from model.stcformer import Model
# from model.ModelEJ import MyModel as Model
# from lib.model.DSTformer import DSTformer as Model
# from lib.utils.camera import *

# from model.MixedGLC import GLCModel as Model
from model.MotionAGFormer import MotionAGFormer as Model_baseline
from model.Mixed_GLC_motion_cross_hanet_2 import GLCModel as Model

from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from demo.lib.utils import get_config

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# for xtion pro
# from openni import openni2
import time
# from collections import deque
import torchvision.transforms as transforms
import torchvision


from lib.hrnet.lib.models.pose_hrnet import get_pose_net
import torch.backends.cudnn as cudnn
from lib.hrnet.lib.config import cfg, update_config

k = 0

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    # lcolor=(0,0,1)
    # rcolor=(1,0,0)

    # color_map = {
    #     0: 'red',    # Color for LR[i] == 0
    #     1: 'blue',  # Color for LR[i] == 1
    #     2: 'orange'    # Color for LR[i] == 2
    # }
    color_map = ['red', 'blue', 'green']
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    # LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)
    LRT = np.array([0, 1, 0, 1, 0, 1, 2, 2, 0,   1,  0,  0,  1,  1, 2, 2], dtype=int)
    
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        # ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)
        col = color_map[LRT[i]]
        ax.plot(x, y, z, lw=2, color=col)
        ax.scatter(x, y, z, color=col)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def generate_2D_pose(image, frame_count, box_model, pose_model, cfg, device):
    
    # cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #### 2D pose ####
    print('\nGenerating 2D pose...')
    # keypoints, scores = gen_video_kpts_realtime(image, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores = gen_vis_realtime(image, box_model, pose_model, cfg, device)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # print(keypoints.shape) # 1, 1, 17, 2
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
    
    print('\nGenerating 2D pose image...')
    input_2D = keypoints[0][0]
    img_pose = show2Dpose(input_2D, copy.deepcopy(image))

    return keypoints, img_pose

def load_model(args, opts):
    ## Reload 
    # model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    # model = Model(args).cuda()
    if opts.usebaseline:
        model = Model_baseline(n_layers=args.n_layers,
                                dim_in=args.dim_in,
                                dim_feat=args.dim_feat,
                                dim_rep=args.dim_rep,
                                dim_out=args.dim_out,
                                mlp_ratio=args.mlp_ratio,
                                act_layer=nn.GELU,
                                attn_drop=args.attn_drop,
                                drop=args.drop,
                                drop_path=args.drop_path,
                                use_layer_scale=args.use_layer_scale,
                                layer_scale_init_value=args.layer_scale_init_value,
                                use_adaptive_fusion=args.use_adaptive_fusion,
                                num_heads=args.num_heads,
                                qkv_bias=args.qkv_bias,
                                qkv_scale=args.qkv_scale,
                                hierarchical=args.hierarchical,
                                num_joints=args.num_joints,
                                use_temporal_similarity=args.use_temporal_similarity,
                                temporal_connection_len=args.temporal_connection_len,
                                use_tcn=args.use_tcn,
                                graph_only=args.graph_only,
                                neighbour_num=args.neighbour_num,
                                n_frames=args.n_frames).cuda()

    else:    
        model = Model(n_layers=args.n_layers,
                                    dim_in=args.dim_in,
                                    dim_feat=args.dim_feat,
                                    dim_rep=args.dim_rep,
                                    dim_out=args.dim_out,
                                    mlp_ratio=args.mlp_ratio,
                                    act_layer=nn.GELU,
                                    attn_drop=args.attn_drop,
                                    drop=args.drop,
                                    drop_path=args.drop_path,
                                    use_layer_scale=args.use_layer_scale,
                                    layer_scale_init_value=args.layer_scale_init_value,
                                    use_adaptive_fusion=args.use_adaptive_fusion,
                                    num_heads=args.num_heads,
                                    qkv_bias=args.qkv_bias,
                                    qkv_scale=args.qkv_scale,
                                    hierarchical=args.hierarchical,
                                    num_joints=args.num_joints,
                                    use_temporal_similarity=args.use_temporal_similarity,
                                    temporal_connection_len=args.temporal_connection_len,
                                    use_tcn=args.use_tcn,
                                    graph_only=args.graph_only,
                                    neighbour_num=args.neighbour_num,
                                    n_frames=args.n_frames).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    if opts.usebaseline:
        model_path = os.path.join('motionagformer-b-h36m.pth.tr')
        # model_path = os.path.join('motionagformer-b-mpi.pth.tr')
    else:
        model_path = os.path.join('checkpoint_final', 'latest_epoch28_mo_38.0949671528231_31.980921960780424.pth.tr')
        # model_path = os.path.join('mpi-checkpoint', 'best_epoch_15.68.pth.tr')
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('Loading checkpoint', model_path)

    temp_dict = checkpoint['model']
    for name, key in list(checkpoint['model'].items()):
        new_name = name[7:]
        temp_dict[new_name] = key
        del temp_dict[name]
    # print(temp_dict.keys())
    # print(model_dict.keys())
    for name, key in model_dict.items():
        # print(pre_dict['model_pos'][name])
        # model_dict[name] = pre_dict['model_pos'][name]
        model_dict[name] = temp_dict[name]



    model.load_state_dict(checkpoint['model'], strict=True)


    model.eval()

    return model

def generate_3D_pose(args, opts, keypoints_cache, image_2d, output_dir, image_file):

    # args = vars(args)
    # print(args)
    model = load_model(args, opts)

    clips, downsample = turn_into_clips(keypoints_cache)
    clip = clips[0]

    global k
    print('\nGenerating 3D pose...')
    img_size = image_2d.shape
    ## input frames
    # start = max(0, i - args.pad)
    # end =  min(i + args.pad, len(keypoints_cache[0])-1)

    # input_2D_no = keypoints_cache[0][start:end+1]
    input_2D_no = np.array(clips)
    
    left_pad, right_pad = 0, 0
    # if input_2D_no.shape[0] != args.clip_len:
    #     if i < args.pad:
    #         left_pad = args.pad - i
    #     if i > len(keypoints_cache[0]) - args.pad - 1:
    #         right_pad = i + args.pad - (len(keypoints_cache[0]) - 1)

    #     input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
    
    input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
    
    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

    if opts.usebaseline:
        output_3D = model(input_2D)
    else:
        _, output_3D = model(input_2D)

    # if idx == len(clips) - 1:
    #     output_3D = output_3D[:, downsample]


    # N = input_2D.size(0)

    ## estimation
    # output_3D_non_flip = model(input_2D[:, 0])
    # output_3D_flip     = model(input_2D[:, 1])

    # print(output_3D_non_flip.shape)
    # exit()

    # output_3D_flip[:, :, :, 0] *= -1
    # output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

    # output_3D = (output_3D_non_flip + output_3D_flip) / 2

    # output_3D = output_3D[0:, args.pad].unsqueeze(1) 
    output_3D[:, :, 0, :] = 0
    post_out_all = output_3D[0].cpu().detach().numpy()
    post_out = post_out_all[-1]

    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])
    max_value = np.max(post_out)
    post_out /= max_value

    # input_2D_no = input_2D_no[args.pad]

    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(post_out, ax)
    if opts.usebaseline:
        # plt.savefig(os.path.join(output_dir, f'mpi-pose3d_{opts.image_path}_baseline.png'))
        plt.savefig(os.path.join(output_dir, 'pose3d_baseline', f'pose3d_{image_file}_baseline.png'))
    else:
        # plt.savefig(os.path.join(output_dir, f'mpi-pose3d_{opts.image_path}.png'))
        plt.savefig(os.path.join(output_dir, 'pose3d', f'pose3d_{image_file}.png'))
    plt.close(fig)


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    # even = np.linspace(0, n_frames, num=27, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1] # 1
    # if n_frames <= 243:
    new_indices = resample(n_frames)
    clips.append(keypoints[:, new_indices, ...])
    downsample = np.unique(new_indices, return_index=True)[1]
    # else:
    #     for start_idx in range(0, n_frames, 243):
    #         keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
    #         clip_length = keypoints_clip.shape[1]
    #         if clip_length != 243:
    #             new_indices = resample(clip_length)
    #             clips.append(keypoints_clip[:, new_indices, ...])
    #             downsample = np.unique(new_indices, return_index=True)[1]
    #         else:
    #             clips.append(keypoints_clip)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", help="Path to the config file.")
    # parser.add_argument("--config", type=str, default="configs/mpi/MotionAGFormer-base.yaml", help="Path to the config file.")
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--window', type=str, default='243', help='sliding window length')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    # parser.add_argument('--pose2d', type=str, default='hrnet', help='type of 2d pose estimator')
    parser.add_argument('--camera', type=str, default='offline', help='camera used(webcam or xtion_pro)')
    parser.add_argument('--cfg', type=str, default='/home/enjhih/Enjhih/MotionAGFormer_new/demo/lib/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml')
    parser.add_argument('--root', type=str, default='/home/enjhih/Enjhih/MotionAGFormer_new/demo')
    parser.add_argument('--image-dir', type=str, default='/home/enjhih/Enjhih/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/TS6/imageSequence')
    parser.add_argument('--outputdir', type=str, default='./demo/output_seq')
    parser.add_argument('--usebaseline', action='store_true')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    opts = parser.parse_args()

    # args expected by supporting codebase  
    opts.modelDir = ''
    opts.logDir = ''
    opts.dataDir = ''
    opts.prevModelDir = ''
    return opts

def reset_config(opts):
    update_config(cfg, opts)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


if __name__ == "__main__":
    opts = parse_args()
    reset_config(opts)

    args = get_config(opts.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    output_dir = opts.outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'pose2d')):
        os.makedirs(os.path.join(output_dir, 'pose2d'))
    if opts.usebaseline:
        if not os.path.exists(os.path.join(output_dir, 'pose3d_baseline')):
            os.makedirs(os.path.join(output_dir, 'pose3d_baseline'))
    else:
        if not os.path.exists(os.path.join(output_dir, 'pose3d')):
            os.makedirs(os.path.join(output_dir, 'pose3d'))

    print(f'Using camera: {opts.camera}')
    #### initialize camera ####
    if opts.camera == 'xtion_pro':
        openni2.initialize()     # can also accept the path of the OpenNI redistribution
        dev = openni2.Device.open_any()
        print(dev.get_device_info())

        depth_stream = dev.create_depth_stream()
        color_stream = dev.create_color_stream()

        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        color_stream.start()
        depth_stream.start()

    elif opts.camera == 'webcam':
        cap = cv2.VideoCapture(0)


    previous_time = time.time()
    delta = 0
    sec = 0
    frame_count = 0
    
    update_config(cfg, opts)
    pose_model = get_pose_net(cfg, is_train=False)
   
    print('=> loading 2d hrnet model ...')
    pose_model.load_state_dict(torch.load('/home/enjhih/Enjhih/MotionAGFormer_new/demo/lib/checkpoint/pose_hrnet_w48_384x288.pth'), strict=False)
    pose_model.to(device)
    pose_model.eval()


    # image_cache = []
    keypoints_cache = [] # store 2d keypoints
    pose_cache = [] # store 2d HPE images

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(device)
    box_model.eval()


    if opts.camera == 'xtion_pro':
        #### read RGB ####
        cframe = color_stream.read_frame()
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([240, 320, 3])
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        frame = np.transpose(np.array([B, G, R]), [1, 2, 0])
    elif opts.camera == 'webcam':
        ret, frame = cap.read()
    elif opts.camera == 'offline':
        k=0
        file_names = sorted(os.listdir(opts.image_dir))
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        image_files = [f for f in file_names if f.lower().endswith(image_extensions)]

        # 3D HPE
        if opts.usebaseline:
            print("Using MotionAGFormer baseline model!")

        for image_file in image_files:
            k+=1
            print(k)
            frame = cv2.imread(os.path.join(opts.image_dir, image_file))
            # cv2.imwrite(os.path.join(output_dir, f'org_{image_file}.png'), frame)

            # keypoints, image_2d = generate_2D_pose(frame.copy(), frame_count) # 2D keypoint and 2d pose image of current frame
            keypoints, image_2d = generate_2D_pose(frame.copy(), frame_count, box_model, pose_model, cfg, device) # 2D keypoint and 2d pose image of current frame
            cv2.imwrite(os.path.join(output_dir, 'pose2d', f'pose2d_{image_file}.png'), image_2d)
    
            if len(keypoints_cache) <= 0: # At initialization, populate clip with initial frame
                for i in range(int(opts.window)):
                    pose_cache.append(image_2d)
                    keypoints_cache.append(keypoints)

            # Add the predicted pose and keypoints to last and pop out the oldest one
            pose_cache.append(image_2d)
            pose_cache.pop(0)
            keypoints_cache.append(keypoints)
            keypoints_cache.pop(0)
            # update_cache(frame.copy())

            # if frame_count % int(args.window) == 0: # full 2d sequence -> inference 3d pose sequence
            print("start 3D pose estimation")
            keypoints_cache_r = np.array(keypoints_cache)
            keypoints_cache_r = np.reshape(keypoints_cache_r, (1, int(opts.window), 17, 3))
            # pose_cache = np.array(pose_cache)

            generate_3D_pose(args, opts, keypoints_cache_r, image_2d, output_dir, image_file)
    
    print('Generating demo successful!')

    # color_stream.stop()
    # dev.close()
    cv2.destroyAllWindows()