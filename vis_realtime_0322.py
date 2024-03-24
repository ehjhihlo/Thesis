import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts_realtime
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.strided_transformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# for xtion pro
from openni import openni2
import time
from lib.openpose.gen_pose import get_openpose_2d
# from collections import deque

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


def generate_2D_pose(image, frame_count):
    
    # cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #### 2D pose ####
    print('\nGenerating 2D pose...')
    keypoints, scores = gen_video_kpts_realtime(image, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # print(keypoints.shape) # 1, 1, 17, 2
    # Add conf score to the last dim

    # keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
    
    print('\nGenerating 2D pose image...')
    input_2D = keypoints[0][0]
    img_pose = show2Dpose(input_2D, copy.deepcopy(image))
    #     pose_cache_2d.append(image)
    cv2.imwrite(os.path.join('./temp/', f'pose_2D_{frame_count:03}.png'), img_pose)
    return keypoints, img_pose

def load_model(args):
    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))

    for path in model_paths:
        if os.path.split(path)[-1][0] == 'n':
            model_path = path

    pre_dict = torch.load(model_path)

    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]    
    model.load_state_dict(model_dict)

    model.eval()

    return model

def generate_3D_pose(keypoints_cache, image_2d, frame_count):
    args, _ = argparse.ArgumentParser().parse_known_args()
    # args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, 351
    # args.stride_num = [3, 9, 13]
    args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, 27
    args.stride_num = [3, 3, 3]    
    args.pad = (args.frames - 1) // 2
    # args.previous_dir = 'checkpoint/pretrained'
    args.previous_dir = 'checkpoint/0221_1446_55_27'
    args.n_joints, args.out_joints = 17, 17

    model = load_model(args)


    global k
    print('\nGenerating 3D pose...')
    # output_3d_all = []
    ## input frames
    # start = max(0, i - args.pad)
    # end =  min(i + args.pad, len(keypoints[0])-1)

    # input_2D_no = keypoints[0][start:end+1]
    input_2D_no = keypoints_cache[0]
    
    left_pad, right_pad = 0, 0
    if input_2D_no.shape[0] != args.frames:
        if i < args.pad:
            left_pad = args.pad - i
        if i > len(keypoints[0]) - args.pad - 1:
            right_pad = i + args.pad - (len(keypoints[0]) - 1)

        input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=image_2d.shape[1], h=image_2d.shape[0])  

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[ :, :, 0] *= -1
    input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
    
    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

    N = input_2D.size(0)

    ## estimation
    output_3D_non_flip, _ = model(input_2D[:, 0])
    output_3D_flip, _     = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0].cpu().detach().numpy()

    # output_3d_all.append(post_out)

    # print('ddd')
    # print(post_out)
    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])

    # input_2D_no = input_2D_no[args.pad]   

    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(post_out, ax)

    fig.canvas.draw()
    image_3d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    sep='')
    image_3d = image_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    
    plt.savefig(f'./temp/pose_3D_{frame_count}.png', dpi=200, format='png', bbox_inches = 'tight')


    k+=1
    print(f'frame_count and k: {frame_count}, {k}')
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
    # even = np.linspace(0, n_frames, num=243, endpoint=False)
    even = np.linspace(0, n_frames, num=27, endpoint=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--window', type=str, default='27', help='sliding window length')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--pose2d', type=str, default='hrnet', help='type of 2d pose estimator')
    args = parser.parse_args()
    # print(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]


    #### initialize camera ####
    openni2.initialize()     # can also accept the path of the OpenNI redistribution
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    color_stream.start()
    depth_stream.start()

    use_cam = False
    if use_cam == True:
        cap = cv2.VideoCapture(0)
    else:
        video_path = './demo/video/' + args.video
        cap = cv2.VideoCapture(video_path)

    # videoWriter = cv2.VideoWriter(os.path.join(outputpath, 'video_2d.mp4'), fourcc, 8.0, (640, 480))
    # videoWriter = cv2.VideoWriter(os.path.join('video.mp4'), fourcc, 8.0, (1920, 480))
    save_video = True
    if save_video == True:
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        # videoWriter = cv2.VideoWriter('video.mp4', fourcc, 8.0, (1920, 480))
        videoWriter = cv2.VideoWriter('video.mp4', fourcc, 3, (1500, 540))

    output_dir = './temp/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir_2d = './temp_2d/'
    os.makedirs(output_dir_2d, exist_ok=True)

    previous_time = time.time()
    delta = 0
    sec = 0
    frame_count = 0
    estimation_count = 0
    
    ### image cache ###
    # image_cache = deque(maxlen = args.window)

    # def update_cache(new_image):
    #     if len(image_cache) == args.window:
    #         image_cache.popleft()  # Remove oldest image
    #     image_cache.append(new_image)  # Add newest image

    # image_cache = []
    keypoints_cache = [] # store 2d keypoints
    pose_cache = [] # store 2d HPE images

    # while(cap.isOpened()):
    while(True):
        # ret, frame = cap.read()

        frame_count += 1
        print(frame_count)
        current_time = time.time()
        delta += current_time - previous_time
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        print('fps = ', fps)
        #### read RGB ####
        cframe = color_stream.read_frame()
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([240, 320, 3])
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        frame = np.transpose(np.array([B, G, R]), [1, 2, 0])
        cv2.imwrite(os.path.join(output_dir_2d, f'out_{frame_count:03}.png'), frame)
        
        '''
        Generate 2D Pose
        1. HRNet (more accurate but very slow)
        2. OpenPose (less accurate but faster)
        '''
        # HRNet
        if args.pose2d == 'hrnet':
            keypoints, image_2d = generate_2D_pose(frame.copy(), frame_count) # 2D keypoint and 2d pose image of current frame
        
        # OpenPose
        if args.pose2d == 'openpose':
            keypoints, image_2d = get_openpose_2d(frame.copy())


        # print(keypoints)
        # print(keypoints.shape)        
        # cv2.imshow('2d pose', np.array(image_2d))
        
        if len(keypoints_cache) <= 0: # At initialization, populate clip with initial frame
            for i in range(int(args.window)):
                pose_cache.append(image_2d)
                keypoints_cache.append(keypoints)

        # Add the predicted pose and keypoints to last and pop out the oldest one
        pose_cache.append(image_2d)
        pose_cache.pop(0)
        keypoints_cache.append(keypoints)
        keypoints_cache.pop(0)
        # update_cache(frame.copy())

        # refresh keypoint cache
        keypoints_cache_fresh = []

        for i in range(len(keypoints_cache)):
            if i < 6:
                keypoints_cache_fresh.append(keypoints_cache[len(keypoints_cache)//2-3])            
            elif i>=6 and i< 10:
                keypoints_cache_fresh.append(keypoints_cache[len(keypoints_cache)//2-2])             
            elif i>=10 and i<16:
                keypoints_cache_fresh.append(keypoints_cache[len(keypoints_cache)//2-1])  
            elif i>=16 and i<20:
                keypoints_cache_fresh.append(keypoints_cache[len(keypoints_cache)//2])
            else:
                keypoints_cache_fresh.append(keypoints_cache[len(keypoints_cache)//2+1])

        print("start 3D pose estimation")
        estimation_count += 1
        keypoints_cache_r = np.array(keypoints_cache)
        keypoints_cache_r = np.reshape(keypoints_cache_r, (1, int(args.window), 17, 2))

        keypoints_cache_fresh = np.array(keypoints_cache_fresh)
        keypoints_cache_fresh = np.reshape(keypoints_cache_fresh, (1, int(args.window), 17, 2))
    
        # print(keypoints_cache_r)
        # print(keypoints_cache_r.shape)

        # 3D HPE
        # if (frame_count-1) % (int(args.window)//2) < 5:
        # if (frame_count-1) % 3 == 0:
        image_3d_original = generate_3D_pose(keypoints_cache_fresh, image_2d, frame_count)
        image_3d = image_3d_original
        
        print(image_3d.shape)
        print('Done!!')
        print(frame_count)
        
               
        # demo
        print('\nGenerating demo...')

        if frame_count > 13:

            ## crop
            image_2d = pose_cache[13]
            edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

            edge = 102
            # if (frame_count-1) % 4 == 0: 
            #     image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]
            # else:
            #     pass
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

            result = cv2.imread(os.path.join(output_dir, f'out_{frame_count:03}.png'))
            cv2.imshow('result', result)
            # print(result.shape) # 540, 1500, 3

            # cv2.imshow('image_2d', image_2d)

            if save_video == True:
                videoWriter.write(result)
                # videoWriter.write(image_2d)
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