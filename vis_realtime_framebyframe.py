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
from model.Mix_gl_cross_0514 import GLCModel as Model

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

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        # ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2)
        ax.scatter(x, y, z)

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
    #     pose_cache_2d.append(image)
    cv2.imwrite(os.path.join('./temp/', f'pose_2D_{frame_count:03}.png'), img_pose)

    return keypoints, img_pose

def load_model(args):
    ## Reload 
    # model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    # model = Model(dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
    #              depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, 
    #              num_joints=args.num_joints, maxlen=args.maxlen, 
    #              qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate, norm_layer=args.norm_layer, att_fuse=args.att_fuse).cuda()
    model = Model(args).cuda()
    model_dict = model.state_dict()
    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    # model_path = os.path.join('checkpoint', 'pose3d', 'MB_train_h36m', 'best_epoch22.bin')
    # model_path = os.path.join('checkpoint', '0423', 'mask_noise', 'best_epoch.bin')
    # model_path = os.path.join('checkpoint', '0427', 'best_epoch_39.7_33.05.bin')
    
    # model_path = os.path.join('checkpoint', '0428', 'best_epoch.bin')
    model_path = os.path.join('checkpoint', '0514-cross-umlp', 'best_epoch.bin')
    pre_dict = torch.load(model_path)
    # print(model_dict.keys())
    # print('44444')
    # print(pre_dict['model_pos'].keys())
    temp_dict = pre_dict['model_pos']
    for name, key in list(pre_dict['model_pos'].items()):
        new_name = name[7:]
        temp_dict[new_name] = key
        del temp_dict[name]
    # print(temp_dict.keys())
    # print(model_dict.keys())
    for name, key in model_dict.items():
        # print(pre_dict['model_pos'][name])
        # model_dict[name] = pre_dict['model_pos'][name]
        model_dict[name] = temp_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    return model

def generate_3D_pose(args, keypoints_cache, image_2d, frame_count):
    #### 3D pose ####
    # args, _ = argparse.ArgumentParser().parse_known_args()
    # args.layers, args.channel, args.d_hid, args.frames = 16, 256, 256, 243

    # args.pad = (args.frames - 1) // 2
    # args.previous_dir = './checkpoint/'
    # args.n_joints, args.out_joints = 17, 17

    # args.dim_in = 3
    # args.dim_out = 3
    # args.n_frames = 243
    # args.num_joints = 17

    # args.in_channels, args.dim_feat, args.dim_rep, args.out_channels = 3, 128, 512, 3
    # args.mlp_ratio, args.act_layer = 4, nn.GELU
    # args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    # args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    # args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    # args.hierarchical = False
    # args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 3, 1
    # args.use_tcn, args.graph_only = False, False


    # args.maxlen=243
    # args.dim_feat=512
    # args.mlp_ration=2
    # args.depth=5
    # args.dim_rep=512
    # args.num_heads=8
    # args.att_fuse=True
    # args.clip_len=243
    # args.data_stride=81
    # args.rootrel=True
    # args.sample_stride=1
    # args.num_joints=17
    # args.no_conf=False
    # args.gt_2d=False
    # args.dim_in=3
    # args.dim_out= 3
    # args.mlp_ratio=2
    # args.qkv_bias=True
    # args.qk_scale=None
    # args.drop_rate=0.
    # args.attn_drop_rate=0.
    # args.drop_path_rate=0.
    # args.norm_layer=nn.LayerNorm
    # args.att_fuse=True
    # args.pad = (args.clip_len - 1) // 2


    # args = vars(args)
    print(args)
    model = load_model(args)

    ## input
    # keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)
    
    clips, downsample = turn_into_clips(keypoints_cache)
    clip = clips[0]
    # clips = []
    # n_frames = keypoints_cache.shape[1]
    # new_indices = resample(n_frames)
    # clips.append(keypoints_cache[:, new_indices, ...])
    # downsample = np.unique(new_indices, return_index=True)[1]
    # video_length = len(keypoints_cache)
    # img_size = image_2d.shape

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
    # input_2D_aug = flip_data(input_2D)
    
    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
    # input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

    # output_3D_non_flip = model(input_2D) 
    # output_3D_flip = flip_data(model(input_2D_aug))
    # output_3D = (output_3D_non_flip + output_3D_flip) / 2
    output_3D = model(input_2D)

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

    # output_dir_3D = output_dir + 'pose3D/'
    # os.makedirs(output_dir_3D, exist_ok=True)

    fig.canvas.draw()
    image_3d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    sep='')
    image_3d = image_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.savefig('pose_3D.png', dpi=200, format='png', bbox_inches='tight')
    # plt.close(fig)
    plt.savefig(f'./temp/pose_3D_{frame_count:03}.png', dpi=200, format='png', bbox_inches = 'tight')
    # image_3d = plt.imread('pose_3D.png')

    # image_3d_cache.append(image_3d)

    k+=1
    print('Generating 3D pose successful!')
    return image_3d
        # plt.savefig('pose_3D.png', dpi=200, format='png', bbox_inches='tight')
        # plt.close(fig)
        
        # image_3d = plt.imread('pose_3D.png')


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()        # plt.savefig('pose_3D.png', dpi=200, format='png', bbox_inches='tight')
        # plt.close(fig)
        
        # image_3d = plt.imread('pose_3D.png')


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
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--window', type=str, default='243', help='sliding window length')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    # parser.add_argument('--pose2d', type=str, default='hrnet', help='type of 2d pose estimator')
    parser.add_argument('--camera', type=str, default='webcam', help='camera used(webcam or xtion_pro)')
    parser.add_argument('--cfg', type=str, default='/home/enjhih/Enjhih/STCFormer/demo/lib/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml')
    # parser.add_argument('--showFps',action='store_true')

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

    video_path = './demo/video/' + opts.video
    video_name = video_path.split('/')[-1].split('.')[0]

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

    # use_cam = False
    # if use_cam == True:
    #     cap = cv2.VideoCapture(0)
    # else:
    #     video_path = './demo/video/' + args.video
    #     cap = cv2.VideoCapture(video_path)
    elif opts.camera == 'webcam':
        cap = cv2.VideoCapture(0)

    # videoWriter = cv2.VideoWriter(os.path.join(outputpath, 'video_2d.mp4'), fourcc, 8.0, (640, 480))
    # videoWriter = cv2.VideoWriter(os.path.join('video.mp4'), fourcc, 8.0, (1920, 480))
    save_video = True
    if save_video == True:
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        # videoWriter = cv2.VideoWriter('video.mp4', fourcc, 8.0, (1920, 480))
        videoWriter = cv2.VideoWriter('video.mp4', fourcc, 7.0, (1280, 480))

    output_dir = './temp/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir_2d = './temp_2d/'
    os.makedirs(output_dir_2d, exist_ok=True)


    previous_time = time.time()
    delta = 0
    sec = 0
    frame_count = 0
    
    ### image cache ###
    # image_cache = deque(maxlen = args.window)

    # def update_cache(new_image):
    #     if len(image_cache) == args.window:
    #         image_cache.popleft()  # Remove oldest image
    #     image_cache.append(new_image)  # Add newest image

    update_config(cfg, opts)
    pose_model = get_pose_net(cfg, is_train=False)
   
    print('=> loading 2d hrnet model ...')
    pose_model.load_state_dict(torch.load('/home/enjhih/Enjhih/STCFormer/demo/lib/checkpoint/pose_hrnet_w48_384x288.pth'), strict=False)
    pose_model.to(device)
    pose_model.eval()


    # image_cache = []
    keypoints_cache = [] # store 2d keypoints
    pose_cache = [] # store 2d HPE images

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(device)
    box_model.eval()

    # while(cap.isOpened()):
    while(True):

        frame_count += 1
        current_time = time.time()
        delta += current_time - previous_time
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        print('fps = ', fps)
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


        cv2.imwrite(os.path.join(output_dir_2d, f'out_{frame_count:03}.png'), frame)        
        
        # keypoints, image_2d = generate_2D_pose(frame.copy(), frame_count) # 2D keypoint and 2d pose image of current frame
        keypoints, image_2d = generate_2D_pose(frame.copy(), frame_count, box_model, pose_model, cfg, device) # 2D keypoint and 2d pose image of current frame

        
        # cv2.imshow('2d pose', np.array(image_2d))
        
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
        
        # model
        # model = load_model()
        
        # 3D HPE
        image_3d_original = generate_3D_pose(args, keypoints_cache_r, image_2d, frame_count)
        image_3d = image_3d_original
        # print(len(image_3d_cache))
        # demo
        print('\nGenerating demo...')
        # merged_cache = list(zip(pose_cache, image_3d_cache))

        # for i, (image_2d, image_3d) in enumerate(merged_cache):
        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("2D Human Pose", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("3D Human Pose", fontsize = font_size)
    
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        fig.canvas.draw()
        result = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        sep='')
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imwrite(os.path.join(output_dir, f'out_{frame_count:03}.png'), result)
        plt.close(fig)
    # img is rgb, convert to opencv's default bgr
    # result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

        # if estimation_count > 0:
        result = cv2.imread(os.path.join(output_dir, f'out_{frame_count:03}.png'))
        cv2.imshow('result', result)


        if save_video == True:
            videoWriter.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # img2video(video_path, output_dir)
        print('Generating demo successful!')

    # cap.release()
    # close the device
    color_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
    if save_video == True:
        videoWriter.release()