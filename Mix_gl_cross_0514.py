from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath
from einops import rearrange, repeat

from model.modules.attention_graph import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP, UMlp
from model.modules.tcn import MultiScaleTCN

# from model.modules.stcformer import STC_BLOCK
# from model.modules.local_constraint_former import LOCALFORMER_BLOCK
from model.modules.GC_former import GLOBALFORMER_BLOCK
# from utils.skeleton import get_skeleton
from model.modules.skeleton import get_skeleton

from model.modules.LC_graph import LocalConstraintGraph
from model.modules.graphinfo import Graph as DG
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



class GraphBlock(nn.Module):
    """
    Implementation of GraphFormer block.
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=27):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)


        self.gcn = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.gcn2Attn_drop = nn.Dropout(p = 0.15)
        self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False) 

    def forward(self, x, attention_feat):
        """
        x: tensor with shape [B, T, J, C]
        """

        res = x
        x = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.gcn(self.norm1(x)))

        graph_feat = self.gcn2Attn_drop(x*self.s_gcn2attn)
        x = x+attention_feat
        x = x+res

        return x, graph_feat


class AttentionBlock(nn.Module):
    """
    Implementation of Attention block.
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=27):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)


        self.atten = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.Attn2gcn_drop = nn.Dropout(p = 0.12)
        self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False) 

    def forward(self, x, attention_feat):
        """
        x: tensor with shape [B, T, J, C]
        """

        res = x
        x = self.norm1(x)

        x = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.mixer(self.norm1(x)))

        graph_feat = self.gcn2Attn_drop(x*self.s_gcn2attn)
        x = x+attention_feat
        x = x+res

        return x, graph_feat


class GraphTransformerBlock(nn.Module):
    """
    Implementation of Cross graph attention block.
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=27):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.atten = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        self.graph = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.norm2 = nn.LayerNorm(dim)

        self.gcn2Attn_drop = nn.Dropout(p = 0.12)
        self.Attn2gcn_drop = nn.Dropout(p = 0.12)
        self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False) 
        self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False) 

        # mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = dim//2
        self.umlp = UMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=drop)


        self.fusion = nn.Linear(dim * 2, 2)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x_graph, x_att):
        """
        x: tensor with shape [B, T, J, C]
        """
        # x_graph = x.clone()
        # x_att = x.clone()

        res_graph = x_graph
        res_att = x_att

        x_graph = self.drop_path(
            self.layer_scale_1.unsqueeze(0).unsqueeze(0)
            * self.graph(self.norm1(x_graph)))

        x_att = self.norm1(x_att)
        x_att, attn2gcn = self.atten(x_att, self.gcn2Attn_drop(x_graph*self.s_gcn2attn))
        x_att = self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)*x_att)
        x_att = res_att + x_att

        x_graph = x_graph + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn)
        x_graph = x_graph + res_graph

        # umlp
        x_graph = x_graph + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.umlp(self.norm2(x_graph)))

        x_att = x_att + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.umlp(self.norm2(x_att)))

        # # fusion
        # alpha = torch.cat((x_att, x_graph), dim=-1)
        # alpha = self.fusion(alpha)
        # alpha = alpha.softmax(dim=-1)
        # x = x_att * alpha[..., 0:1] + x_graph * alpha[..., 1:2]

        return x_graph, x_att

class GLCBlock(nn.Module):
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243, skeleton=get_skeleton(), layernum=0):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # ST Attention branch
        self.graph_former_spatial = GraphTransformerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames)
        
        self.graph_former_temporal = GraphTransformerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                    qk_scale, use_layer_scale, layer_scale_init_value,
                                    mode='temporal', mixer_type="attention",
                                    use_temporal_similarity=use_temporal_similarity,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames)


        self.token_num = 81
        self.graph_former_temporal_prune = GraphTransformerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                    qk_scale, use_layer_scale, layer_scale_init_value,
                                    mode='temporal', mixer_type="attention",
                                    use_temporal_similarity=use_temporal_similarity,
                                    neighbour_num=neighbour_num,
                                    n_frames=self.token_num)
       
        self.layernum = layernum

        ### for token pruning
        self.recover_num = n_frames

        self.layer_index = 10
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, dim))
        self.cross_attention = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.x_token = nn.Parameter(torch.zeros(1, self.recover_num, dim))

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        B, T, J, C = x.shape

        ##-----------------Clustering-----------------##
        if self.layernum == self.layer_index:
            x_knn = rearrange(x, 'b t j c -> b (t c) j')
            x_knn = self.pool(x_knn)
            x_knn = rearrange(x_knn, 'b (t c) 1 -> b t c', t=T)

            index, idx_cluster = cluster_dpc_knn(x_knn, self.token_num, 2)
            index, _ = torch.sort(index)

            batch_ind = torch.arange(B, device=x.device).unsqueeze(-1)
            x= x[batch_ind, index]

            x = rearrange(x, 'b t j c -> (b j) t c')
            x += self.pos_embed_token
            x= rearrange(x, '(b j) t c -> b t j c', j=J)
        
        x_graph = x.clone()
        x_att = x.clone()

        x_graph_spatial, x_att_spatial = self.graph_former_spatial(x_graph, x_att)

        # T = 243
        if self.layernum in [0, 1, 2, 3, 4, 5]:
            x_graph_spatial = rearrange(x_graph_spatial, 'B (K T) J C  -> (B K) T J C', T=27)
            
        elif self.layernum in [6, 7, 8, 9, 10]:
            x_graph_spatial = rearrange(x_graph_spatial, 'B (K T) J C  -> (B K) T J C', T=81)

        x_graph_spatial = rearrange(x_graph_spatial, '(B K) T J C  -> B (K T) J C', B=B)
        



        if self.layernum < self.layer_index:
            x_graph_temporal, x_att_temporal = self.graph_former_temporal(x_graph_spatial, x_att_spatial)
            # x_graph_temporal2, x_att_temporal2 = self.graph_former_temporal(x_graph, x_att)
        else:
            x_graph_temporal, x_att_temporal = self.graph_former_temporal_prune(x_graph_spatial, x_att_spatial)
            # x_graph_temporal2, x_att_temporal2 = self.graph_former_temporal_prune(x_graph, x_att)

        # x_graph_temporal, x_att_temporal = self.graph_former_temporal(x_graph_spatial, x_att_spatial)
        
        # x_graph_spatial2, x_att_spatial2 = self.graph_former_spatial(x_graph_temporal2, x_att_temporal2)

        alpha = torch.cat((x_graph_temporal, x_att_temporal), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_graph_temporal * alpha[..., 0:1] + x_att_temporal * alpha[..., 1:2]

        # beta = torch.cat((x_graph_spatial2, x_att_spatial2), dim=-1)
        # beta = self.fusion( beta)
        # beta = beta.softmax(dim=-1)
        # x_2 = x_graph_temporal *  beta[..., 0:1] + x_att_temporal *  beta[..., 1:2]

        # x = x_1 + x_2

        # Token Recovering Attention        
        if self.layernum == 15:
            x = rearrange(x, 'b t j c -> (b j) t c')
            x_token = repeat(self.x_token, '() t c -> b t c', b = B*J)
            x = x_token + self.cross_attention(x_token, x, x)
            x = rearrange(x, '(b j) t c -> b t j c', j=J)

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
        layers.append(GLCBlock(dim=dim,
                                mlp_ratio=mlp_ratio,
                                act_layer=act_layer,
                                attn_drop=attn_drop,
                                drop=drop_rate,
                                drop_path=drop_path_rate,
                                num_heads=num_heads,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                qkv_bias=qkv_bias,
                                qk_scale=qkv_scale,
                                use_adaptive_fusion=use_adaptive_fusion,
                                hierarchical=hierarchical,
                                use_temporal_similarity=use_temporal_similarity,
                                temporal_connection_len=temporal_connection_len,
                                use_tcn=use_tcn,
                                graph_only=graph_only,
                                neighbour_num=neighbour_num,
                                n_frames=n_frames,
                                skeleton=skeleton,
                                layernum=layer))
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

        print(lowlevel_feature.shape)
        print(intermediate_feature.shape)
        print(highlevel_feature.shape)
        intermediate_feature = torch.cat((intermediate_feature,intermediate_feature,intermediate_feature),dim=1) # 81 -> 243
        complex_feature = torch.cat((lowlevel_feature, intermediate_feature, highlevel_feature),dim=-1)  # B, T, J, 3*C          
        # print(complex_feature.shape)
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