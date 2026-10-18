# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple, Union, List, Dict, Any
from collections import OrderedDict
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertConfig
import torch.amp

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_B32_PATH,
    DWCONV3D_DISABLE_CUDNN,
)

# ==========================================================================
# 核心解耦工具：动态梯度缩放，用于防止 FC 头的特化梯度扭曲共享 CLIP 流形
# ==========================================================================
class GradientScale(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        return grad * ctx.scale, None

# --- [导入模块] ---
print("Attempting to import TokenClusterInter...")
try:
    from modules.cluster.cluster import TokenClusterInter
    print("SUCCESS: TokenClusterInter imported.")
except Exception as e:
    print(f"\n!!! CLUSTER IMPORT ERROR: {e}\n")
    TokenClusterInter = None

print("Attempting to import ObjectCentricDecoder...")
try:
    from modules.oad_decoder import ObjectCentricDecoder
    print("SUCCESS: ObjectCentricDecoder imported.")
except Exception as e:
    print(f"\n!!! OAD IMPORT ERROR: {e}\n")
    ObjectCentricDecoder = None
# -------------------------
print("Attempting to import TemporalMotionBranch...")
try:
    from modules.motion_branch import TemporalMotionBranch
    print("SUCCESS: TemporalMotionBranch imported.")
except Exception as e:
    print(f"\n!!! MOTION IMPORT ERROR: {e}\n")
    TemporalMotionBranch = None
# -------------------------

class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class Adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3), T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x) + offset
        x_id[:, 1:, :] += x
        return x_id

class Adapter1(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B, Ca, T) 

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(-1, Ca)
        x = self.fc2(x)
        x_id = x + x_id
        return x_id

class Adapter2(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3), T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, L, Ca).permute(0, 2, 3, 1).contiguous().view(B * L, Ca, T) 

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(B, L, T, Ca)
        x = x.permute(0, 2, 1, 3).contiguous().view(BT, L, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += offset
        #x_id = x_id.clone()
        #x_id[:, 1:, :] = x_id[:, 1:, :] + offset
        x_id = x + x_id
        return x_id

class OffsetAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        
        x_spatial = x[:, 1:, :].view(B, T, -1, C)
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x_spatial[:, former_id, :, :]
        
        offset = x_spatial - x_former
        offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
        
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        offset = self.conv(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        offset = self.fc2(offset)
        return offset

class TextAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.textad_fc1 = nn.Linear(in_channels, adapter_channels)
        self.textad_gelu = nn.GELU()
        self.textad_fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, adapter_width: int, adapter_kernel_size: Tuple[int, int, int], adapter_pre_attn: bool, adapter_pre_mlp: bool, num_frames: int, attn_mask: torch.Tensor = None, use_cluster: bool = False, cluster_config: Dict[str, Any] = None, use_oad: bool = False, oad_config: Dict[str, Any] = None, block_id: int = 0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.T = num_frames
        self.attn_mask = attn_mask

        self.grad_scale = oad_config.get('grad_scale', 0.3) if oad_config else 0.3

        adapter_class = functools.partial(Adapter2, in_channels=d_model, adapter_channels=adapter_width, kernel_size=3, T=self.T)
        text_adapter_class = functools.partial(TextAdapter, in_channels=d_model, adapter_channels=adapter_width)
        
        if num_frames == 0:
            self.adapter_pre_attn = text_adapter_class() if adapter_pre_attn else None
            self.adapter_pre_attn_off = None
            self.adapter_pre_mlp = text_adapter_class() if adapter_pre_mlp else None
            self.adapter_pre_mlp_off = None
        else:    
            self.adapter_pre_attn = adapter_class() if adapter_pre_attn else None
            self.adapter_pre_attn_off = None
            self.adapter_pre_mlp = adapter_class() if adapter_pre_mlp else None
            self.adapter_pre_mlp_off = None
        
        self.cluster = None
        enable_cluster = cluster_config.get('enable', True) if cluster_config else False
        if use_cluster and cluster_config is not None and enable_cluster and TokenClusterInter is not None:
            current_tokens = cluster_config.get('current_tokens', 197) 
            target_tokens = cluster_config.get('target_tokens', 49)
            self.cluster = TokenClusterInter(algorithm=cluster_config.get('algorithm', 'kmediods++'), block_id=block_id, cluster_num=target_tokens, before_cluster_num=current_tokens, before_block_frames=num_frames, after_block_frames=num_frames, distance='euclidean', transformer_width=d_model)

        self.oad_decoder = None
        self.oad_use_text_cond = bool(oad_config.get('use_text_cond', False)) if oad_config else False
        if use_oad and oad_config is not None and ObjectCentricDecoder is not None:
            self.oad_decoder = ObjectCentricDecoder(
                embed_dim=d_model,
                num_queries=oad_config.get('num_queries', 8),
                num_layers=oad_config.get('num_layers', 2),
                num_heads=n_head,
                num_frames=oad_config.get('num_frames', num_frames),
                dropout=0.0,
                num_spatial_tokens=oad_config.get('num_spatial_tokens', 196),
                pe_type=oad_config.get('pe_type', 'sincos'),
                use_text_cond=self.oad_use_text_cond,
                text_dim=oad_config.get('text_dim', 512),
            )

        # ===== 运动分支（替代冗余 cluster；显式帧间差分，喂闭集时序头）=====
        self.motion_branch = None
        if use_oad and oad_config is not None and oad_config.get('use_motion', False) and TemporalMotionBranch is not None:
            self.motion_branch = TemporalMotionBranch(
                dim=d_model,
                num_frames=oad_config.get('num_frames', num_frames),
                num_patches=oad_config.get('num_spatial_tokens', 196),
                grad_scale=oad_config.get('grad_scale', 0.3),
            )
        self.last_motion_feat = None
        self.last_motion_gate = 0.0

        # 全类别文本库（由 VisionTransformer.forward 下发，供文本条件化 query 使用；None 则不条件化）
        self._oad_text_bank = None

        self.actual_cluster_tokens = 0
        self.actual_oad_tokens = 0

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x.permute(1, 0, 2)
        x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return x.permute(1, 0, 2)

    def cross_attention(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        x = self.attn(x, k, k, need_weights=False, attn_mask=None)[0]
        return x.permute(1, 0, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_pre_attn_off is not None: x = self.adapter_pre_attn_off(x)
        if self.adapter_pre_attn is not None: x = self.adapter_pre_attn(x)
        
        x = x + self.attention(self.ln_1(x))
        
        if self.adapter_pre_mlp_off is not None: x = self.adapter_pre_mlp_off(x)
        if self.adapter_pre_mlp is not None: x = self.adapter_pre_mlp(x)
            
        x = x + self.mlp(self.ln_2(x))

        x_cluster = None
        x_oad = None
        self.last_oad_feat = None  

        if self.cluster is not None:
            x_in_c = x.permute(1, 0, 2)
            x_out_c, _ = self.cluster(x_in_c)
            x_cluster = x_out_c.permute(1, 0, 2)
            if x_cluster.dim() == 2:
                x_cluster = x_cluster.unsqueeze(1)

        if self.oad_decoder is not None:
            BT, L_total, C = x.shape
            B = BT // self.T
            # 应用梯度缩放系数，防止 OAD 反传破坏视觉 Encoder 特征
            vis_tokens = GradientScale.apply(x[:, 1:, :], self.grad_scale)
            
            # 兼容 OAD 返回元组或单一 Tensor
            oad_out = self.oad_decoder(vis_tokens, B=B, T=self.T,
                                       text_bank=getattr(self, '_oad_text_bank', None))
            oad_video = oad_out[0] if isinstance(oad_out, tuple) else oad_out
            
            self.last_oad_feat = oad_video
            Q = oad_video.size(1)
            x_oad = oad_video.unsqueeze(1).expand(B, self.T, Q, C).reshape(BT, Q, C)

        # 运动分支：从同一批 patch token 显式提取帧间差分，产出 video-level 运动向量（已 gated）
        self.last_motion_feat = None
        if self.motion_branch is not None:
            BT_m = x.size(0)
            B_m = BT_m // self.T
            motion_feat, motion_gate = self.motion_branch(x[:, 1:, :], B_m, self.T)
            self.last_motion_feat = motion_feat
            self.last_motion_gate = motion_gate

        self.actual_cluster_tokens = x_cluster.size(1) if x_cluster is not None else 0
        self.actual_oad_tokens = x_oad.size(1) if x_oad is not None else 0

        append_list = [x]
        if x_cluster is not None: append_list.append(x_cluster)
        if x_oad is not None: append_list.append(x_oad)

        if len(append_list) > 1: x = torch.cat(append_list, dim=1)

        return x

    def forward_cross(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.adapter_pre_attn is not None: x = self.adapter_pre_attn(x)
        x = x + self.cross_attention(self.ln_1(x), self.ln_1(k))
        if self.adapter_pre_mlp is not None: x = self.adapter_pre_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, adapter_width: int, adapter_layers: int, adapter_kernel_size: Tuple[int, int, int], adapter_pre_attn: bool, adapter_pre_mlp: bool, num_frames: int, attn_mask: torch.Tensor = None, cluster_config: Dict[str, Any] = None, oad_config: Dict[str, Any] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        
        cluster_target_layers = cluster_config.get('layers', []) if cluster_config else []

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(d_model=width, n_head=heads, adapter_width=adapter_width, adapter_kernel_size=adapter_kernel_size, adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers, adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers, num_frames=num_frames, attn_mask=attn_mask, use_cluster=(i in cluster_target_layers), cluster_config=cluster_config, use_oad=(i in cluster_target_layers), oad_config=oad_config, block_id=i + 1)
            for i in range(layers)
        ])
        self.use_grad_ckpt = False   # 由 main.py 按 --use_grad_ckpt 设置；仅 checkpoint 纯 backbone 层
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.resblocks):
            if i < 12:
                # 带 OAD/cluster/motion 的层有副作用输出(last_oad_feat 等)，不能 checkpoint
                has_side_effect = (getattr(block, 'oad_decoder', None) is not None
                                   or getattr(block, 'cluster', None) is not None
                                   or getattr(block, 'motion_branch', None) is not None)
                if self.use_grad_ckpt and self.training and not has_side_effect:
                    x = torch_checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, adapter_width: int, adapter_layers: int, adapter_kernel_size: Tuple[int, int, int], adapter_pre_attn: bool, adapter_pre_mlp: bool, num_classes: int, num_frames: int, class_fc: bool = True, cluster_config: Dict[str, Any] = None, oad_config: Dict[str, Any] = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, adapter_width, adapter_layers, adapter_kernel_size, adapter_pre_attn, adapter_pre_mlp, num_frames, cluster_config=cluster_config, oad_config=oad_config)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.t_adapter = Adapter1(in_channels=output_dim, adapter_channels=output_dim // 2, kernel_size=3, T=num_frames)
        self.adapter_dropout = nn.Dropout(0.5)
        
        self.adapter_fc_spatial = nn.Linear(width, num_classes) if class_fc else None
        self.adapter_fc_temporal = nn.Linear(width, num_classes) if class_fc else None
        
        if self.adapter_fc_spatial is not None:
            nn.init.normal_(self.adapter_fc_spatial.weight, std=0.02)
            nn.init.constant_(self.adapter_fc_spatial.bias, 0.)
            nn.init.normal_(self.adapter_fc_temporal.weight, std=0.002)
            nn.init.constant_(self.adapter_fc_temporal.bias, 0.)

        self.adapter_fc = self.adapter_fc_spatial

        self.spatial_t_adapter = TextAdapter(in_channels=width, adapter_channels=width // 4)
        
        self.gate_cluster = nn.Parameter(torch.tensor([0.2]))
        self.gate_oad = nn.Parameter(torch.tensor([0.3]))
        self.last_gate_oad = 0.0
        self.last_gate_cluster = 0.0  
        self.last_gate_motion = 0.0

        self.grad_scale = oad_config.get('grad_scale', 0.3) if oad_config else 0.3
        # FC 空间头对共享 backbone 的梯度缩放（默认=grad_scale；想给 FC 更强/更弱解耦时单独调）
        self.fc_grad_scale = oad_config.get('fc_grad_scale', self.grad_scale) if oad_config else self.grad_scale
        # 文本条件化 query 用的全类别文本库（外部按 epoch 设置；None 时 OAD 退化为无文本条件，行为同原版）
        self.text_bank = None

        # OAD 对比路径的开关与结构
        self.use_fine_grained_contrastive = oad_config.get('use_fine_grained_contrastive', False) if oad_config else False
        
        self.cluster_to_cls = nn.Linear(width, width, bias=False)
        self.oad_to_cls = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.cluster_to_cls.weight, std=0.01)
        nn.init.normal_(self.oad_to_cls.weight, std=0.01)

        self.gate_cluster_contrast = nn.Parameter(torch.tensor([0.1]))  
        self.gate_oad_contrast = nn.Parameter(torch.tensor([0.1]))      

        self.last_gate_cluster_contrast = 0.0
        self.last_gate_oad_contrast = 0.0

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # 把全类别文本库下发给携带 OAD 的 block（供文本条件化 query 读取；text_bank=None 时无副作用）
        for _blk in self.transformer.resblocks:
            if getattr(_blk, 'oad_decoder', None) is not None:
                _blk._oad_text_bank = self.text_bank

        x = self.transformer(x)

        if self.adapter_fc_spatial is not None:
            current_tokens = x.size(1)
            x_all = x.contiguous().view(B, T, current_tokens, x.size(-1))
            actual_cluster_tokens = 0
            oad_feat_video = None
            motion_feat_video = None
            for block in self.transformer.resblocks:
                if getattr(block, 'actual_cluster_tokens', 0) > 0:
                    actual_cluster_tokens = block.actual_cluster_tokens
                if getattr(block, 'last_oad_feat', None) is not None:
                    oad_feat_video = block.last_oad_feat  
                if getattr(block, 'last_motion_feat', None) is not None:
                    motion_feat_video = block.last_motion_feat
                    self.last_gate_motion = getattr(block, 'last_motion_gate', 0.0)

            # ===== Spatial Track =====
            spatial_seq = x_all[:, :, 0, :]  
            if actual_cluster_tokens > 0:
                actual_oad_tokens_local = 0
                for block in self.transformer.resblocks:
                    if getattr(block, 'actual_oad_tokens', 0) > 0:
                        actual_oad_tokens_local = block.actual_oad_tokens
                start_idx = -(actual_cluster_tokens + actual_oad_tokens_local) if actual_oad_tokens_local > 0 else -actual_cluster_tokens
                end_idx = -actual_oad_tokens_local if actual_oad_tokens_local > 0 else None
                cluster_seq = x_all[:, :, start_idx:end_idx, :].mean(dim=2)
                cluster_gate_val = torch.tanh(self.gate_cluster)
                self.last_gate_cluster = cluster_gate_val.detach().item()
                spatial_seq = spatial_seq + cluster_gate_val * cluster_seq
            
            B_, T_, C_ = spatial_seq.shape
            spatial_seq = self.spatial_t_adapter(spatial_seq.reshape(B_ * T_, C_)).reshape(B_, T_, C_)
            x2_spatial = spatial_seq.mean(dim=1)
            x2_spatial = self.ln_post(x2_spatial)

            x2_spatial_scaled = GradientScale.apply(x2_spatial, self.fc_grad_scale)

            # ===== Temporal Track (video-level OAD) =====
            if oad_feat_video is not None:
                x2_temporal = oad_feat_video.mean(dim=1)
                x2_temporal = self.ln_post(x2_temporal)
                gate_val = torch.tanh(self.gate_oad)
                self.last_gate_oad = gate_val.detach().item()
                x2_temporal_scaled = gate_val * GradientScale.apply(x2_temporal, self.grad_scale)
            else:
                x2_temporal_scaled = torch.zeros_like(x2_spatial_scaled)
                self.last_gate_oad = 0.0

            # 方案C：motion 独立注入时序头——绕过 gate_oad 的二次门控与共享 ln_post，
            # 仅由 motion 自身内部 gate 控制；motion_feat 已含 out_norm，无需再 ln。
            if motion_feat_video is not None:
                x2_temporal_scaled = x2_temporal_scaled + GradientScale.apply(motion_feat_video, self.grad_scale)

            logits_s = self.adapter_fc_spatial(self.adapter_dropout(x2_spatial_scaled))
            logits_t = self.adapter_fc_temporal(self.adapter_dropout(x2_temporal_scaled))
            x2 = logits_s + logits_t
        else:
            x2 = None
            
        x_norm = self.ln_post(x)
        cls_feat = x_norm[:, 0, :]
        
        # 【机制修正】：OAD 是否接入对比路径必须受 use_fine_grained_contrastive 严格控制！
        # 如果不开该开关，则保证对比路径的绝对纯净，防止零样本产生分布偏移
        if getattr(self, 'use_fine_grained_contrastive', False) and (oad_feat_video is not None):
            # 给 OAD 的反向传播梯度踩刹车 (0.1)，防止其破坏 CLIP 原生主干
            oad_mean = GradientScale.apply(oad_feat_video.mean(dim=1), 0.1)                     
            oad_contrib = self.oad_to_cls(oad_mean)                  
            oad_contrib = oad_contrib.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)  
            
            go_val = torch.tanh(self.gate_oad_contrast)
            self.last_gate_oad_contrast = go_val.detach().item()
            cls_feat = cls_feat + go_val * oad_contrib
        else:
            self.last_gate_oad_contrast = 0.0

        x1 = cls_feat @ self.proj
        x1 = self.t_adapter(x1) 
        x_contrastive = x1.view(B, T, -1).mean(dim=1, keepdim=False)
            
        return x_contrastive, x1.view(B, T, -1), x2, None, None

def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None and isinstance(attr, torch.Tensor):
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def copy_weights(source_module, target_module):
    source_params = dict(source_module.named_parameters())
    target_params = dict(target_module.named_parameters())
    for target_name, target_param in target_params.items():
        if target_name in source_params and source_params[target_name].data.shape == target_param.data.shape:
            target_param.data.copy_(source_params[target_name].data)

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, vision_adapter_width: int, vision_adapter_layers: int, vision_adapter_kernel_size: Tuple[int, int, int], vision_adapter_pre_attn: bool, vision_adapter_pre_mlp: bool, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int, text_adapter_width: int, text_adapter_layers: int, text_adapter_kernel_size: Tuple[int, int, int], text_adapter_pre_attn: bool, text_adapter_pre_mlp: bool, num_classes: int, num_frames: int, mlm_head_len: int, cluster_config: Dict[str, Any] = None, oad_config: Dict[str, Any] = None):
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, adapter_width=vision_adapter_width, adapter_layers=vision_adapter_layers, adapter_kernel_size=vision_adapter_kernel_size, adapter_pre_attn=vision_adapter_pre_attn, adapter_pre_mlp=vision_adapter_pre_mlp, output_dim=embed_dim, num_classes=num_classes, class_fc=True, num_frames=num_frames, cluster_config=cluster_config, oad_config=oad_config)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask(), adapter_width=text_adapter_width, adapter_layers=text_adapter_layers, adapter_kernel_size=text_adapter_kernel_size, adapter_pre_attn=text_adapter_pre_attn, adapter_pre_mlp=text_adapter_pre_mlp, num_frames=0)
        self.vocab_size = vocab_size
        self.mlm_head_len = mlm_head_len
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = BertConfig(vocab_size=mlm_head_len, hidden_size=transformer_width)
        self.initialize_parameters()
        self.mlm_adapter = MLMHead(self.config)
        self.mlm_adapter.apply(self.init_mlm_weights)

        self.text_prefix = nn.Parameter(torch.zeros(1, 4, transformer_width))
        # 【核心修正】：彻底废弃 Prompt Tuning，物理级防止 text_prefix 破坏零样本对齐空间！
        self.text_prefix.requires_grad = False 
        self.use_text_prefix = False

    def init_mlm_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)): module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): module.bias.data.zero_(); module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None: module.bias.data.zero_()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None: nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length).fill_(float("-inf")).triu_(1)
        return mask
        
    @property
    def dtype(self): 
        return self.visual.conv1.weight.dtype
    
    def encode_image(self, image): 
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text, force_no_prefix=False):
        x = self.token_embedding(text).type(self.dtype)
        eot_idx = text.argmax(dim=-1)
        
        # 即使 use_text_prefix 被设为 False，这里也加上 force_no_prefix 双保险
        if getattr(self, 'use_text_prefix', False) and not force_no_prefix:
            P = self.text_prefix.size(1)
            prefix = self.text_prefix.expand(x.size(0), -1, -1).type(self.dtype)
            x_new = torch.cat([x[:, :1], prefix, x[:, 1:77 - P]], dim=1)
            new_eot_idx = torch.clamp(eot_idx + P, max=76)
            out_of_bounds = (eot_idx + P) >= 77
            if out_of_bounds.any():
                x_new[out_of_bounds, 76, :] = x[out_of_bounds, eot_idx[out_of_bounds], :]
            x = x_new
            eot_idx = new_eot_idx
            
        x = x + self.positional_embedding[:x.size(1)].type(self.dtype)
        x1 = self.transformer(x)
        x = self.ln_final(x1).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x, x1.type(self.dtype)

    def compute_mlm(self, text_feats, mlm_labels, visual_feats):
        text_feats = self.transformer.resblocks[12].forward_cross(text_feats, visual_feats)
        text_feats = self.ln_final(text_feats)
        mlm_logits = self.mlm_adapter(text_feats)
        mlm_loss = F.cross_entropy(mlm_logits.view(-1, self.mlm_head_len), mlm_labels.view(-1), ignore_index=-100)
        return {"mlm_loss": mlm_loss}
    
    def forward(self, image, text, force_no_prefix=False):
        image_features, image_mlm_inputs, image_fc, cluster_norm, oad_norm = self.encode_image(image)
        text_features, text_mlm_inputs = self.encode_text(text, force_no_prefix=force_no_prefix)
        
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, image_features, text_mlm_inputs, image_mlm_inputs, image_fc, cluster_norm, oad_norm

def clip_vit_base_patch16_adapter24x384(**kwargs):
    model = VisionTransformer(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, adapter_width=384, adapter_layers=12, adapter_kernel_size=(3, 1, 1), adapter_pre_attn=True, adapter_pre_mlp=True, **kwargs)
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu'); model.load_state_dict(checkpoint.visual.state_dict(), strict=False); return model

def clip_vit_base_patch16_adapter12x384(**kwargs):
    model = VisionTransformer(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, adapter_width=384, adapter_layers=12, adapter_kernel_size=(3, 1, 1), adapter_pre_attn=False, adapter_pre_mlp=True, **kwargs)
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu'); model.load_state_dict(checkpoint.visual.state_dict(), strict=False); return model

def clip_vit_base_patch32_adapter12x384(**kwargs):
    model = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, adapter_width=384, adapter_layers=12, adapter_kernel_size=(3, 1, 1), adapter_pre_attn=False, adapter_pre_mlp=True, **kwargs)
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu'); model.load_state_dict(checkpoint.visual.state_dict(), strict=False); return model

def clip_vit_base_patch32_adapter24x384(**kwargs):
    model = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, adapter_width=384, adapter_layers=12, adapter_kernel_size=(3, 1, 1), adapter_pre_attn=True, adapter_pre_mlp=True, **kwargs)
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu'); model.load_state_dict(checkpoint.visual.state_dict(), strict=False); return model

def clip_vit_base_patch16_multimodal_adapter24x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model = CLIP(embed_dim=512, image_resolution=224, vision_patch_size=16, vision_width=768, vision_layers=12, vision_adapter_width=384, vision_adapter_layers=12, vision_adapter_kernel_size=(3, 1, 1), vision_adapter_pre_attn=True, vision_adapter_pre_mlp=True, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=13, text_adapter_width=384, text_adapter_layers=13, text_adapter_kernel_size=(3, 1, 1), text_adapter_pre_attn=True, text_adapter_pre_mlp=True, **kwargs)
    model.load_state_dict(checkpoint.state_dict(), strict=False); copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12]); convert_weights(model); return model

def clip_vit_base_patch32_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model = CLIP(embed_dim=512, image_resolution=224, vision_patch_size=32, vision_width=768, vision_layers=12, vision_adapter_width=384, vision_adapter_layers=12, vision_adapter_kernel_size=(3, 1, 1), vision_adapter_pre_attn=False, vision_adapter_pre_mlp=True, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=13, text_adapter_width=384, text_adapter_layers=1, text_adapter_kernel_size=(3, 1, 1), text_adapter_pre_attn=False, text_adapter_pre_mlp=True, **kwargs)
    model.load_state_dict(checkpoint.state_dict(), strict=False); copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12]); convert_weights(model); return model

def clip_vit_base_patch16_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model = CLIP(embed_dim=512, image_resolution=224, vision_patch_size=16, vision_width=768, vision_layers=12, vision_adapter_width=384, vision_adapter_layers=12, vision_adapter_kernel_size=(3, 1, 1), vision_adapter_pre_attn=False, vision_adapter_pre_mlp=True, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=13, text_adapter_width=384, text_adapter_layers=2, text_adapter_kernel_size=(3, 1, 1), text_adapter_pre_attn=False, text_adapter_pre_mlp=True, **kwargs)
    model.load_state_dict(checkpoint.state_dict(), strict=False); copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12]); convert_weights(model); return model
    
def clip_vit_base_patch32_multimodal_adapter24x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model = CLIP(embed_dim=512, image_resolution=224, vision_patch_size=32, vision_width=768, vision_layers=12, vision_adapter_width=384, vision_adapter_layers=12, vision_adapter_kernel_size=(3, 1, 1), vision_adapter_pre_attn=True, vision_adapter_pre_mlp=True, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=13, text_adapter_width=384, text_adapter_layers=13, text_adapter_kernel_size=(3, 1, 1), text_adapter_pre_attn=False, text_adapter_pre_mlp=True, **kwargs)
    model.load_state_dict(checkpoint.state_dict(), strict=False); copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12]); convert_weights(model); return model

