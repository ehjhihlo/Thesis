"""Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/model/loss.py"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)


def acc_error(predicted, target):
    """
    Calculates acceleration error:
         1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    """
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = predicted[:-2] - 2 * predicted[1:-1] + predicted[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    return np.mean(normed, axis=1)


def jpe(predicted, target):
    """
    per-joint position error
    """
    assert predicted.shape == target.shape
    return np.linalg.norm(predicted - target, axis=len(target.shape) - 1)


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:, :, :, :2]
    target_2d = target[:, :, :, :2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length


def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length) / gt_3d_length, 2).mean()
    return loss_length


def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens


def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var


def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt)  # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)


def loss_limb_gt_hyperbone(x, gt):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]]
    
    # directed_edges_hop1 = [(parrent, child) for child, parrent in enumerate(skeleton.parents()) if parrent >= 0]
    directed_edges_hop2 = [(0,1,2),(0,4,5),(0,7,8),(1,2,3),(4,5,6),(7,8,9),(7,8,11),(7,8,14),(8,9,10),(8,11,12),(8,14,15),(11,12,13),(14,15,16)] # (parrent, child)
    directed_edges_hop3 = [(0,1,2,3),(0,4,5,6),(0,7,8,9),(7,8,9,10),(7,8,11,12),(7,8,14,15),(8,11,12,13),(8,14,15,16)]
    directed_edges_hop4 = [(0,7,8,9,10),(0,7,8,11,12),(0,7,8,14,15),(7,8,11,12,13),(7,8,14,15,16)]
    
    connections_2 = [[0, 1], [3, 4], [6, 7], [1, 2], [4, 5], [7, 8], [7, 10], [7, 13], [8, 9], [10, 11], [13, 14], [11, 12], [14, 15]]
    connections_3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [7, 8, 9], [7, 10, 11], [7, 13, 14], [10, 11, 12], [13, 14, 15]]
    connections_4 = [[6, 7, 8, 9], [6, 7, 11, 12], [6, 7, 13, 14], [7, 10, 11, 12], [7, 13, 14, 15]]

    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    limb_lens_1st_order = torch.norm(limbs, dim=-1) # (N, T, 16)
    
    limb_lens_2nd_order = limb_lens_1st_order[:, :, connections_2]
    limb_lens_2nd_order = limb_lens_2nd_order[:, :, :, 0] + limb_lens_2nd_order[:, :, :, 1]

    limb_lens_3rd_order = limb_lens_1st_order[:, :, connections_3]
    limb_lens_3rd_order = limb_lens_3rd_order[:, :, :, 0] + limb_lens_3rd_order[:, :, :, 1] + limb_lens_3rd_order[:, :, :, 2]

    limb_lens_4th_order = limb_lens_1st_order[:, :, connections_4]
    limb_lens_4th_order = limb_lens_4th_order[:, :, :, 0] + limb_lens_4th_order[:, :, :, 1] + limb_lens_4th_order[:, :, :, 2] + limb_lens_4th_order[:, :, :, 3]

    limbs_gt = gt[:, :, limbs_id, :]
    limbs_gt = limbs_gt[:, :, :, 0, :] - limbs_gt[:, :, :, 1, :]
    limb_lens_1st_order_gt = torch.norm(limbs_gt, dim=-1) # (N, T, 16)
    
    limb_lens_2nd_order_gt = limb_lens_1st_order_gt[:, :, connections_2]

    limb_lens_2nd_order_gt = limb_lens_2nd_order_gt[:, :, :, 0] + limb_lens_2nd_order_gt[:, :, :, 1]

    limb_lens_3rd_order_gt = limb_lens_1st_order_gt[:, :, connections_3]
    limb_lens_3rd_order_gt = limb_lens_3rd_order_gt[:, :, :, 0] + limb_lens_3rd_order_gt[:, :, :, 1] + limb_lens_3rd_order_gt[:, :, :, 2]

    limb_lens_4th_order_gt = limb_lens_1st_order_gt[:, :, connections_4]
    limb_lens_4th_order_gt = limb_lens_4th_order_gt[:, :, :, 0] + limb_lens_4th_order_gt[:, :, :, 1] + limb_lens_4th_order_gt[:, :, :, 2] + limb_lens_4th_order_gt[:, :, :, 3]

    loss = 0.3*nn.L1Loss()(limb_lens_1st_order, limb_lens_1st_order_gt) + 0.3*nn.SmoothL1Loss()(limb_lens_2nd_order, limb_lens_2nd_order_gt) \
            + 0.2*nn.SmoothL1Loss()(limb_lens_3rd_order, limb_lens_3rd_order_gt) + 0.2*nn.SmoothL1Loss()(limb_lens_4th_order, limb_lens_4th_order_gt)

    # loss = loss/4

    return loss


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))


def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)


def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]
    # left shoulder - left elbow 11-12
    # left elbow - left wrist 12-13
    # right shoulder - right elbow 14-15
    # right elbow - right wrist 15-16 
    angle_id = [[0, 3],
                [0, 6],
                [3, 6],
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 10],
                [7, 13],
                [8, 13],
                [10, 13],
                [7, 8],
                [8, 9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15]]
    eps = 1e-7
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    angles = limbs[:, :, angle_id, :]
    angle_cos = F.cosine_similarity(angles[:, :, :, 0, :], angles[:, :, :, 1, :], dim=-1)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))


def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)


def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:, 1:] - x_a[:, :-1]
    gt_av = gt_a[:, 1:] - gt_a[:, :-1]
    return nn.L1Loss()(x_av, gt_av)

### Enjhih add
def bone_len_loss(x, gt):
    """
    ref: A geometric loss combination for 3d human pose estimation (WACV 2024)
    """
    
    weight = 1
    
    limb_lens_x = get_limb_lens(x) # (N, T, 16)
    limb_lens_gt = get_limb_lens(gt)

    upper_arm = limb_lens_x[:, :, 11] + limb_lens_x[:, :, 14]
    lower_arm = limb_lens_x[:, :, 12] + limb_lens_x[:, :, 15]
    upper_leg = limb_lens_x[:, :, 1] + limb_lens_x[:, :, 4]
    lower_leg = limb_lens_x[:, :, 2] + limb_lens_x[:, :, 5]

    upper_arm_gt = limb_lens_gt[:, :, 11] + limb_lens_gt[:, :, 14]
    lower_arm_gt = limb_lens_gt[:, :, 12] + limb_lens_gt[:, :, 15]
    upper_leg_gt = limb_lens_gt[:, :, 1] + limb_lens_gt[:, :, 4]
    lower_leg_gt = limb_lens_gt[:, :, 2] + limb_lens_gt[:, :, 5]

    loss_up = torch.norm((upper_arm - upper_arm_gt), dim=-1) + torch.norm((lower_arm - lower_arm_gt), dim=-1)
    loss_down = torch.norm((upper_leg - upper_leg_gt), dim=-1) + torch.norm((lower_leg - lower_leg_gt), dim=-1)

    return torch.mean(weight*0.5*(loss_up + loss_down))

def get_angle_between_two_vector(n1, n2):
    eps = 1e-7
    angle_cos = F.cosine_similarity(n1, n2, dim=-1)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))  

def body_part_orientive_loss(x, gt):
    assert x.shape == gt.shape
    
    # normal vector
    n_pred = []
    n_gt = []
    # select the vector
    allvectors = [[0, 1], [0, 2], [1, 2], [1, 3],
                [0, 4], [0, 5], [4, 5], [4, 6],
                [8, 11], [8, 12], [11, 12], [11, 13],
                [8, 14], [8, 15], [14, 15], [14, 16],
                [8, 9], [8, 10], [0, 7], [0, 8]]
    vectors = x[:, :, allvectors, :]
    vectors_gt = gt[:, :, allvectors, :]
    vectors = (vectors[:, :, :, 0, :] - vectors[:, :, :, 1, :]).squeeze() # N, len(vector), 3
    vectors_gt = (vectors_gt[:, :, :, 0, :] - vectors_gt[:, :, :, 1, :]).squeeze()
    # print(vectors.shape) # B, T, 20, 3
    # print(vectors_gt.shape)
    vectors = list(vectors.chunk(len(vectors[0]), dim=1))
    vectors_gt = list(vectors_gt.chunk(len(vectors_gt[0]), dim=1))

    for i in range(0, len(allvectors), 2):
        # vectors[i] = vectors[i].squeeze()
        # vectors[i+1] = vectors[i+1].squeeze()
        # vectors_gt[i] = vectors_gt[i].squeeze()
        # vectors_gt[i+1] = vectors_gt[i+1].squeeze()

        # print(vectors[i].shape) # B, 20, 3
        # print(vectors_gt[i].shape)

        n_pred.append(torch.cross(vectors[i], vectors[i+1], dim=-1)/torch.linalg.norm(torch.cross(vectors[i], vectors[i+1], dim=-1)))
        n_gt.append(torch.cross(vectors_gt[i], vectors_gt[i+1], dim=-1)/torch.linalg.norm(torch.cross(vectors_gt[i], vectors_gt[i+1], dim=-1)))

    n_pred = torch.stack(n_pred)
    n_gt = torch.stack(n_gt)

    loss = nn.L1Loss(reduction='mean')
    out = loss(n_pred, n_gt)

    normal_angles = get_angle_between_two_vector(n_pred, n_gt)
    normal_angle_loss = torch.mean(torch.sum(normal_angles, dim=0))
    # print(normal_angle_loss) #15.43
    # print(out) #0.06
    return out, normal_angle_loss
    # return out

def root_relative_loss(predicted, target):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    assert predicted.shape == target.shape
    # calculate the distance between any other joints to root
    error = 0
    for i in range(1, predicted.shape[2]):
        dist = predicted[:, :, i, :] - predicted[:, :, 0, :]
        dist_gt = target[:, :, i, :] - target[:, :, 0, :]
        error += torch.norm(dist - dist_gt, dim=-1)

    return torch.mean(error/(predicted.shape[2]-1)) # no torso

def focal_mpjpe(predicted, target):
    """
    Apply focal loss on joints (farther from bottom torso with higher weights)
    """
    assert predicted.shape == target.shape
    error = 0
    weight = [1]

    for i in range(1, predicted.shape[2]):
        dist = predicted[:, :, i, :] - predicted[:, :, 0, :]
        dist_gt = target[:, :, i, :] - target[:, :, 0, :]
        weight.append(torch.mean(torch.norm(dist - dist_gt, dim=-1)).item())
        # weight = torch.cat(weight, [torch.mean(torch.norm(dist - dist_gt, dim=-1))], dim=0)
        error += torch.mean(torch.norm(dist - dist_gt, dim=-1))

    error2 = error.item()
    new_weight = [k/error2*(predicted.clone().shape[2]-1) for k in weight]
    new_weight[0] = 1

    # print(torch.mean(weight * torch.norm(predicted - target, dim=len(target.shape) - 1)))
    mpjpe = torch.norm(predicted.clone() - target.clone(), dim=len(target.shape) - 1)
    mpjpe2 = mpjpe.clone()
    # apply calculated weight
    for i in range(predicted.shape[2]):
        mpjpe2[:, :, i] = new_weight[i]*mpjpe[:, :, i]

    return torch.mean(mpjpe2)