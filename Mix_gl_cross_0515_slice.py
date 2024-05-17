from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath
from einops import rearrange, repeat

from model.modules.graph import GCN
from model.modules.mlp import MLP, UMlp

from model.modules.skeleton import get_skeleton
import numpy as np

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points

# Apply Token Pruning Cluster
# Dynamically select a few pose tokens of representative frames
def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GraphAttentionBlock(nn.Module):
    # def __init__(self, d_time, d_joint, d_coor, head=8):
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,drop_path=0.,
                 use_temporal_similarity=True, temporal_connection_len=1, neighbour_num=4, n_frames=27,use_layer_scale=True,layer_scale_init_value=1e-5):
        super().__init__()
        
        ###ST_Attention###
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.head = num_heads
        self.layer_norm = nn.LayerNorm(dim_in)

        self.scale = (dim_in// 2) ** -0.5
        self.proj = nn.Linear(dim_in, dim_in)
        self.head = num_heads

        # sep1
        # print(d_coor)
        self.emb = nn.Embedding(5, dim_in//num_heads//2)
        self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()

        # sep2
        self.sep2_t = nn.Conv2d(dim_in // 2, dim_in // 2, kernel_size=3, stride=1, padding=1, groups=dim_in // 2)
        self.sep2_s = nn.Conv2d(dim_in // 2, dim_in // 2, kernel_size=3, stride=1, padding=1, groups=dim_in // 2)

        self.drop = DropPath(0.5)

        ###ST_Graph###
        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.gcn_s = GCN(dim_in//2, dim_in//2,
                        num_nodes=17,
                        neighbour_num=neighbour_num,
                        mode='spatial',
                        use_temporal_similarity=use_temporal_similarity,
                        temporal_connection_len=temporal_connection_len)

        self.gcn_t = GCN(dim_in//2, dim_in//2,
                        num_nodes=n_frames,
                        neighbour_num=neighbour_num,
                        mode='temporal',
                        use_temporal_similarity=use_temporal_similarity,
                        temporal_connection_len=temporal_connection_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = True
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim_in//2), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim_in), requires_grad=True)

        self.gcn2Attn_drop = nn.Dropout(p = 0.15)
        self.Attn2gcn_drop = nn.Dropout(p = 0.12)
        self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False) 
        self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False) 

        mlp_hidden_dim = dim_in//2
        self.umlp = UMlp(in_features=dim_in, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=0.1)


        self.fusion = nn.Linear(dim_in * 2, 2)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, input):
        b, t, s, c = input.shape

        ### STAttention
        h = input
        x = self.layer_norm(input)

        qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c

        # space group and time group
        qkv_s, qkv_t = qkv.chunk(2, 4)  # [3,b,t,s,c//2],  [3,b,t,s,c//2]

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c//2
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c//2

        # reshape for mat
        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)  # b,t,s,c//2-> b*h*t,s,c//2//h
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)  # b,t,s,c//2-> b*h*t,c//2//h,s

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)  # b,t,s,c//2 -> b*h*s,t,c//2//h
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  # b,t,s,c//2->  b*h*s,c//2//h,t

        att_s = (q_s @ k_s) * self.scale  # b*h*t,s,s
        att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t

        att_s = att_s.softmax(-1)  # b*h*t,s,s
        att_t = att_t.softmax(-1)  # b*h*s,t,t

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        # sep2 
        sep2_s = self.sep2_s(v_s)  # b,c//2,t,s
        sep2_t = self.sep2_t(v_t)  # b,c//2,t,s
        sep2_s = rearrange(sep2_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        sep2_t = rearrange(sep2_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        # sep1
        # v_s = rearrange(v_s, 'b c t s -> (b t ) s c')
        # v_t = rearrange(v_t, 'b c t s -> (b s ) t c')
        sep_s = self.emb(self.part).unsqueeze(0)  # 1,s,c//2//h
        sep_t = self.emb(self.part).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1,1,1,s,c//2//h

        # MSA
        v_s = rearrange(v_s, 'b (h c) t s   -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        x_s = att_s @ v_s + sep2_s + 0.0001 * self.drop(sep_s)  # b*h*t,s,c//2//h
        x_t = att_t @ v_t + sep2_t  # b*h,t,c//h                # b*h*s,t,c//2//h

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2 
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2 

        x_t = x_t + 1e-9 * self.drop(sep_t)

        x_t = rearrange(x_t, 'b h t s c -> b  t s (h c) ')  # b,t,s,c//2
        x_s = rearrange(x_s, 'b h t s c -> b  t s (h c) ')  # b,t,s,c//2

        att_s = x_s
        att_t = x_t

        ###STGraph
        x_graph = self.norm1(input)
        x_graph_s, x_graph_t = x_graph.chunk(2, 3) # b,t,j,c//2 , b,t,j,c//2

        x_graph_s = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.gcn_s(x_graph_s))

        x_graph_t = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.gcn_t(x_graph_t))

        # x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        # x = rearrange(x, 'b h t s c -> b  t s (h c) ')  # b,t,s,c
        
        # cross interaction
        x_s = x_s + self.gcn2Attn_drop(x_graph_s*self.s_gcn2attn)
        x_t = x_t + self.gcn2Attn_drop(x_graph_t*self.s_gcn2attn)
        
        x_graph_s = x_graph_s + self.Attn2gcn_drop(att_s*self.s_attn2gcn)
        x_graph_t = x_graph_t + self.Attn2gcn_drop(att_t*self.s_attn2gcn)

        x_graph_s = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.gcn_s(x_graph_s))
        x_graph_t = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.gcn_t(x_graph_t))
        
        # projection and skip-connection
        x = torch.cat((x_s, x_t), -1)
        x = self.proj(x)
        x = x + h

        x_graph = torch.cat((x_graph_s, x_graph_t), -1)

        # umlp
        x_graph = x_graph + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.umlp(self.norm2(x_graph)))

        x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.umlp(self.norm2(x)))

        alpha = torch.cat((x_graph, x), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_graph * alpha[..., 0:1] + x * alpha[..., 1:2]

        return x

def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243, skeleton=get_skeleton()):
    """
    generates MotionAGFormer layers
    """
    layers = []
    for layer in range(n_layers):
        layers.append(GraphAttentionBlock(dim_in=dim,
                                          dim_out=3,
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          attn_drop=attn_drop,
                                          proj_drop=0.,
                                          drop_path=0.,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value
                                          ))
    layers = nn.Sequential(*layers)

    return layers

class MotionEncoder(nn.Module):
    def __init__(self, num_heads, dim, dim_feat, n_joints, out_channels, qkv_bias, mlp_ratio, drop=0, drop_path=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_feat)
        self.motion_att = MotionAttentionBlock(num_heads, dim, dim_feat, n_joints, out_channels, qkv_bias)
        # mlp_hidden_dim = int(dim_feat * mlp_ratio//2)
        mlp_hidden_dim = dim_feat//2
        self.umlp = UMlp(in_features=dim_feat, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, v, a):
        B, T, J, C = x.shape
        x = x + self.drop_path(self.motion_att(self.layer_norm(x), v, a))
        x = x + self.drop_path(self.umlp(self.layer_norm(x)))

        return x
    
class MotionAttentionBlock(nn.Module):
    def __init__(self, num_heads, dim, dim_feat, n_joints, out_channels, qkv_bias):
        super().__init__()

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim_feat, dim_feat * 3, bias=qkv_bias)
        self.scale = dim_feat ** -0.5
        
        self.proj = nn.Linear(dim_feat, dim_feat)
        self.proj_drop = nn.Dropout(0.1)
        self.attn_drop = nn.Dropout(0.)
        self.layer_norm = nn.LayerNorm(dim_feat)


    def forward(self, x, v, a):
        B, T, J, C = x.shape

        attn = (v @ a.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ x
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class GLCModel(nn.Module):
    """
    MyModel, the main class of our model.
    """
    def __init__(self, args):
        
        super().__init__()

        # self.joints_embed = nn.Linear(args.in_channels, args.dim_feat)
        self.joints_embed = nn.Linear(args.dim_in, args.dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, args.n_joints, args.dim_feat))
        
        ### Apply SPE before feeding into main structure
        self.emb = nn.Embedding(5, args.dim_feat)
        self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()
        
        self.spe2 = nn.Conv2d(args.dim_feat, args.dim_feat, kernel_size=3, stride=1, padding=1, groups=args.dim_feat)

        self.norm = nn.LayerNorm(args.dim_feat)

        self.skeleton = get_skeleton()
        self.layers = create_layers(dim=args.dim_feat,
                                    n_layers=args.layers,
                                    mlp_ratio=args.mlp_ratio,
                                    act_layer=args.act_layer,
                                    attn_drop=args.attn_drop,
                                    drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    num_heads=args.num_heads,
                                    use_layer_scale=args.use_layer_scale,
                                    qkv_bias=args.qkv_bias,
                                    qkv_scale=args.qkv_scale,
                                    layer_scale_init_value=args.layer_scale_init_value,
                                    use_adaptive_fusion=args.use_adaptive_fusion,
                                    hierarchical=args.hierarchical,
                                    use_temporal_similarity=args.use_temporal_similarity,
                                    temporal_connection_len=args.temporal_connection_len,
                                    use_tcn=args.use_tcn,
                                    graph_only=args.graph_only,
                                    neighbour_num=args.neighbour_num,
                                    n_frames=args.n_frames,
                                    skeleton=self.skeleton)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(args.dim_feat, args.dim_rep)),
            ('act', nn.Tanh())
        ]))
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.dim_feat, momentum=0.1),
            nn.Conv1d(args.dim_feat, 3*args.num_joints, kernel_size=1)
        )

        # self.occlusion = OcclusionAttentionBlock(args.num_heads, args.in_channels, args.dim_feat, args.n_joints, args.out_channels, args.qkv_bias)

        self.v_embed = nn.Linear(args.dim_in, args.dim_feat)
        self.acc_embed = nn.Linear(args.dim_in, args.dim_feat)

        self.motion_layers = 1
        self.motion_aware = MotionEncoder(args.num_heads, args.dim_in, args.dim_feat, 
                                          args.num_joints, args.dim_out, args.qkv_bias,
                                          args.mlp_ratio, args.drop, args.drop_path)

        self.motion_aware_block = []
        for l in range(self.motion_layers):
            self.motion_aware_block.append(MotionEncoder(args.num_heads, args.dim_in, args.dim_feat, 
                                          args.num_joints, args.dim_out, args.qkv_bias,
                                          args.mlp_ratio, args.drop, args.drop_path))
        self.motion_aware_block = nn.ModuleList(self.motion_aware_block)

        self.head = nn.Linear(args.dim_rep, args.dim_out)


        # self.use_adaptive_fusion = args.use_adaptive_fusion
        # if self.use_adaptive_fusion:
        self.motion_fusion = nn.Linear(args.dim_feat*2, 2)
        self.fusion = nn.Linear(args.dim_feat * 3, 3)
        self._init_fusion()

        # self.embedd_mid = nn.Conv1d(in_channels=2176, out_channels=dim_feat, kernel_size=3, padding=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=17, stride=17)

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.2)
        self.motion_fusion.weight.data.fill_(0)
        self.motion_fusion.bias.data.fill_(0.25)

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        B, T, J, C = x.shape

        # occ = self.occlusion(x)
        x_org = x
        x = self.joints_embed(x)

        # positional encoding
        x = x + self.pos_embed
        # spe = self.emb(self.part).unsqueeze(0)  # 1,j,c
        # spe2 = self.spe2(rearrange(x, 'b t j c -> b c t j'))
        # spe2 = rearrange(spe2, 'b c t j -> b t j c')
        # x = x + spe + 0.0001*spe2

        v = x_org[:, 1:] - x_org[:, :-1]
        zero_list = torch.zeros((B, 1, J, C)).cuda()
        v = torch.cat((zero_list, v), dim=1)
        a = v[:, 1:] - v[:, :-1]
        a = torch.cat((zero_list, a), dim=1)
        v = self.v_embed(v) # v-> q
        a = self.acc_embed(a) # a -> k

        # x_mo = self.motion_aware(x, v, a)

        # for i in range(self.motion_layers):
        #     x = self.motion_aware_block[i](x, v, a)

    
        lowlevel_feature = torch.zeros((B, T, J, C)).cuda()
        intermediate_feature = torch.zeros((B, T, J, C)).cuda()
        highlevel_feature = torch.zeros((B, T, J, C)).cuda()

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i==5:
                lowlevel_feature = x
            elif i==10:
                intermediate_feature = x
            elif i==15:
                highlevel_feature = x

        # intermediate_feature = torch.cat((intermediate_feature,intermediate_feature,intermediate_feature),dim=1) # 81 -> 243
        complex_feature = torch.cat((lowlevel_feature, intermediate_feature, highlevel_feature),dim=-1)  # B, T, J, 3*C          
        
        ##### Adaptive fusion
        beta = self.fusion(complex_feature)
        beta = beta.softmax(dim=-1)
        
        x = lowlevel_feature * beta[..., 0:1] + intermediate_feature * beta[..., 1:2] +\
            highlevel_feature * beta[..., 2:3]
        #####
        
        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = MyModel()(n_layers=12, dim_in=3, dim_feat=64, mlp_ratio=4, hierarchical=False,
                           use_tcn=False, graph_only=False, n_frames=t).to('cuda')
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100 
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")
    

    out = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()