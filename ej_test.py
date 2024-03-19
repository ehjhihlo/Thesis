import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_angle_between_two_vector(n1, n2):
    eps = 1e-7
    angle_cos = F.cosine_similarity(n1, n2, dim=-1)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))    

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    print(predicted.shape)
    if predicted.shape[1] <= 1:
        print('dfas')
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    print(torch.norm(velocity_predicted - velocity_target, dim=-1))
    print(torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1)))
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def body_part_orientive_loss(x, gt):
    assert x.shape == gt.shape
    
    # normal vector
    n_pred = []
    n_gt = []
    # select the vector
    allvectors = [[0, 1], [0, 2], [1, 2], [1, 3],
                [0, 4], [0, 5], [4, 5], [4, 6],
                [8, 14], [8, 15], [14, 15], [14, 16],
                [8, 11], [8, 12], [11, 12], [11, 13],
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

        n_pred.append(torch.cross(vectors[i], vectors[i+1], dim=-1))
        n_gt.append(torch.cross(vectors_gt[i], vectors_gt[i+1], dim=-1))


    n_pred = torch.stack(n_pred)
    n_gt = torch.stack(n_gt)
    # print(n_pred) # 10, B, 1, 3
    # print(n_pred.shape)
    loss = nn.L1Loss(reduction='mean')
    out = loss(n_pred, n_gt)
    # print(out)
    normal_angles = get_angle_between_two_vector(n_pred, n_gt)
    # print('eeeee')
    # print(normal_angles)
    # print(normal_angles.shape)
    # print(torch.mean(torch.sum(normal_angles, dim=0)))
    normal_angle_loss = torch.mean(torch.sum(normal_angles, dim=0))
    return out, normal_angle_loss


def bone_len_loss(x, gt):
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

    print('ddd')
    print(upper_arm - upper_arm_gt)
    print(torch.norm((upper_arm - upper_arm_gt), dim=-1))
    loss_up = torch.norm((upper_arm - upper_arm_gt), dim=-1) + torch.norm((lower_arm - lower_arm_gt), dim=-1)
    loss_down = torch.norm((upper_leg - upper_leg_gt), dim=-1) + torch.norm((lower_leg - lower_leg_gt), dim=-1)
    print(0.5*(loss_up + loss_down))
    out = 0.5*(loss_up + loss_down)[0]
    print(out)
    print(torch.mean(0.5*(loss_up + loss_down)))
    return 0.5*(loss_up + loss_down)


def hyperbone_loss(x, gt):
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


    print('fff')
    print(limb_lens_1st_order)
    print(limb_lens_2nd_order)
    print(limb_lens_3rd_order)
    print(limb_lens_4th_order)

    print(nn.L1Loss()(limb_lens_1st_order, limb_lens_1st_order_gt))
    print(nn.L1Loss()(limb_lens_2nd_order, limb_lens_2nd_order_gt))
    print(nn.SmoothL1Loss()(limb_lens_2nd_order, limb_lens_2nd_order_gt))
    print(nn.L1Loss()(limb_lens_3rd_order, limb_lens_3rd_order_gt))
    print(nn.SmoothL1Loss()(limb_lens_3rd_order, limb_lens_3rd_order_gt))
    print(nn.L1Loss()(limb_lens_4th_order, limb_lens_4th_order_gt))
    print(nn.SmoothL1Loss()(limb_lens_4th_order, limb_lens_4th_order_gt))
    loss = nn.L1Loss()(limb_lens_1st_order, limb_lens_1st_order_gt) + nn.SmoothL1Loss()(limb_lens_2nd_order, limb_lens_2nd_order_gt) \
            + nn.SmoothL1Loss()(limb_lens_3rd_order, limb_lens_3rd_order_gt) + nn.SmoothL1Loss()(limb_lens_4th_order, limb_lens_4th_order_gt)

    print(loss)


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
    # print(';jsd;a')
    # print(angles[:, :, :, 0, :])
    # print(angles[:, :, :, 1, :])
    # print(angles[:, :, :, 0, :].shape)
    # print(angles[:, :, :, 1, :].shape)
    angle_cos = F.cosine_similarity(angles[:, :, :, 0, :], angles[:, :, :, 1, :], dim=-1)
    # print(angle_cos.shape)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))

# Input: (N, T, 17, 3)
# Output: (N, T, 16)
# b, c, t, j = 1, 3, 27, 17
b, c, t, j = 2, 3, 1, 17
x = torch.randn((b, t, j, c)).to('cuda')
gt = torch.randn((b, t, j, c)).to('cuda')
print(x)
print(x[0][0][0])
print(x[0][0][1])
# print(gt)
hyperbone_loss(x, gt)
x[0][0][0] = torch.Tensor([0, 0, 0])
gt[0][0][0] = torch.Tensor([1, 1, 1])
# print(x)
# print(gt)

loss1 = bone_len_loss(x, gt)
# print('sjlda;')
loss2 = body_part_orientive_loss(x, gt)
# print(loss1)
# print(loss2)

# print(get_angles(x))
print('fsfs')
loss_velocity(x, gt)
print('sadsa')

def root_relative_loss(predicted, target):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    assert predicted.shape == target.shape
    root_joint = predicted[:, :, 0, :]
    root_joint_gt = target[:, :, 0, :]
    # calculate the distance between any other joints to root
    print(root_joint)
    # print(root_joint.shape)
    print(root_joint_gt)
    error = 0
    for i in range(1, predicted.shape[2]):
        dist = predicted[:, :, i, :] - predicted[:, :, 0, :]
        dist_gt = target[:, :, i, :] - target[:, :, 0, :]
        error += torch.norm(dist - dist_gt, dim=-1)

    return torch.mean(error/(predicted.shape[2]-1)) # no torso

root_relative_loss(x, gt)


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))

a = get_limb_lens(x)
b = get_limb_lens(gt)
    # limbs_id = [[0, 1], [1, 2], [2, 3],
    #             [0, 4], [4, 5], [5, 6],
    #             [0, 7], [7, 8], [8, 9], [9, 10],
    #             [8, 11], [11, 12], [12, 13],
    #             [8, 14], [14, 15], [15, 16]
    #             ]
print(a)

def focal_mpjpe(predicted, target):
    """
    Apply focal loss on joints (farther from bottom torso with higher weights)
    """
    assert predicted.shape == target.shape
    error = 0
    weight = []
    weight.append(1)

    for i in range(1, predicted.shape[2]):
        dist = predicted[:, :, i, :] - predicted[:, :, 0, :]
        dist_gt = target[:, :, i, :] - target[:, :, 0, :]
        weight.append(torch.mean(torch.norm(dist - dist_gt, dim=-1)).item())
        # weight = torch.cat(weight, [torch.mean(torch.norm(dist - dist_gt, dim=-1))], dim=0)
        error += torch.mean(torch.norm(dist - dist_gt, dim=-1))

    error = error.item()
    weight = [i/error*(predicted.shape[2]-1) for i in weight]
    weight[0] = 1

    # print(torch.mean(weight * torch.norm(predicted - target, dim=len(target.shape) - 1)))
    mpjpe = torch.norm(predicted - target, dim=len(target.shape) - 1)

    # apply calculated weight
    for i in range(predicted.shape[2]):
        mpjpe[:, :, i] = weight[i]*mpjpe[:, :, i]
    print(torch.mean(mpjpe))
    return torch.mean(mpjpe)

def root_relative_loss(predicted, target):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    assert predicted.shape == target.shape

    error = 0
    for i in range(1, predicted.shape[2]):
        dist = predicted[:, :, i, :] - predicted[:, :, 0, :]
        dist_gt = target[:, :, i, :] - target[:, :, 0, :]
        error += torch.norm(dist - dist_gt, dim=-1)

    return torch.mean(error/(predicted.shape[2]-1)) # no torso


focal_mpjpe(x, gt)