import math
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


# ==========================================================================
# Sin-Cos 位置编码工具（硬编码，O(1) 量级，与归一化后的特征同尺度）
# ==========================================================================
def build_1d_sincos(length: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, f"1d sincos dim must be even, got {dim}"
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def build_2d_sincos(h: int, w: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, f"2d sincos dim must be even, got {dim}"
    half = dim // 2
    pe_h = build_1d_sincos(h, half)
    pe_w = build_1d_sincos(w, half)
    pe = torch.zeros(h, w, dim, dtype=torch.float32)
    pe[:, :, :half] = pe_h.unsqueeze(1).expand(h, w, half)
    pe[:, :, half:] = pe_w.unsqueeze(0).expand(h, w, half)
    return pe.reshape(h * w, dim)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.fc2(x); x = self.drop2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, return_attn=False):
        B, N, C = x.shape
        _, L, _ = y.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_used = self.attn_drop(attn)
        x = (attn_used @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        if return_attn:
            return x, attn.mean(dim=1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_y = norm_layer(dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, return_attn=False):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        if return_attn:
            cross_x, attn = self.cross_attn(self.norm2(x), self.norm2_y(y), return_attn=True)
            x = x + self.drop_path2(cross_x)
            x = x + self.drop_path3(self.mlp(self.norm3(x)))
            return x, attn
        else:
            x = x + self.drop_path2(self.cross_attn(self.norm2(x), self.norm2_y(y)))
            x = x + self.drop_path3(self.mlp(self.norm3(x)))
            return x


# ==========================================================================
# 【改造版】Video-level Spatio-Temporal Object Decoder
#   修复 1: PE 注入前先 LayerNorm，避免位置信号被深层特征的大幅值淹没（关键）
#   新增 2: 因子化时空 PE（temporal[每帧] + spatial[每patch]），默认 sin-cos
#   新增 3: 文本条件化 Query（query 通过 cross-attn 读取「全类别文本库」，无 GT 泄漏）
# ==========================================================================
class ObjectCentricDecoder(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 num_queries=8,
                 num_layers=2,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 num_frames=8,
                 num_spatial_tokens=196,    # ViT-B/16@224 -> 14x14=196；ViT-L/14@224 -> 16x16=256
                 pe_type='sincos',          # 'sincos'(推荐，O(1)量级) | 'learnable'
                 use_text_cond=False,       # 是否启用文本条件化 query（要对零样本生效还需在 adapter 里把 OAD 接入迁移路径）
                 text_dim=512):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_spatial_tokens = num_spatial_tokens
        self.pe_type = pe_type
        self.use_text_cond = use_text_cond

        # 关键修复：把 memory 特征先归一化，PE 才能真正起作用
        self.pe_norm = nn.LayerNorm(embed_dim)

        # Video-level learnable queries
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.normal_(self.query_embed, std=0.02)

        # ---------- 因子化时空位置编码 ----------
        H = W = int(round(math.sqrt(num_spatial_tokens)))
        assert H * W == num_spatial_tokens, \
            f"num_spatial_tokens={num_spatial_tokens} 不是完全平方数，无法构造 2D 空间 PE"
        if pe_type == 'sincos':
            self.register_buffer('temporal_pe',
                                 build_1d_sincos(num_frames, embed_dim).view(1, num_frames, 1, embed_dim),
                                 persistent=False)
            self.register_buffer('spatial_pe',
                                 build_2d_sincos(H, W, embed_dim).view(1, 1, num_spatial_tokens, embed_dim),
                                 persistent=False)
            self.temporal_pos = None
            self.spatial_pos = None
        elif pe_type == 'learnable':
            # 归一化后特征 ~单位尺度，PE 用稍大的 std 才有意义（0.02 太小）
            self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
            self.spatial_pos = nn.Parameter(torch.zeros(1, 1, num_spatial_tokens, embed_dim))
            nn.init.trunc_normal_(self.temporal_pos, std=0.2)
            nn.init.trunc_normal_(self.spatial_pos, std=0.2)
            self.temporal_pe = None
            self.spatial_pe = None
        else:
            raise ValueError(f"unknown pe_type: {pe_type}")

        # ---------- 文本条件化模块（leakage-safe：只读全类别文本库，不读 GT） ----------
        if use_text_cond:
            self.text_proj = nn.Linear(text_dim, embed_dim)
            self.norm_q_pre = nn.LayerNorm(embed_dim)
            self.norm_text = nn.LayerNorm(embed_dim)
            self.text_cross = CrossAttention(dim=embed_dim, num_heads=num_heads,
                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
            # gate 初始为 0：开局无影响，可从旧 ckpt 平滑恢复，逐步学到文本引导
            self.text_cond_gate = nn.Parameter(torch.zeros(1))
        else:
            self.text_proj = None

        self.blocks = nn.ModuleList([
            CrossAttnBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=True, drop=dropout, attn_drop=dropout, drop_path=0.0)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _get_pe(self, T, L, dtype):
        if self.pe_type == 'sincos':
            return self.temporal_pe[:, :T].to(dtype), self.spatial_pe[:, :, :L].to(dtype)
        else:
            return self.temporal_pos[:, :T], self.spatial_pos[:, :, :L]

    def forward(self, x_encoder, B, T, text_bank=None):
        """
        Args:
            x_encoder: [BT, L, C]  视觉 token（不含 CLS，L=num_spatial_tokens）
            B, T: 视频数 / 帧数
            text_bank: [Ncls, text_dim] 已归一化的全类别文本特征（启用文本条件化时传入）
        Returns:
            q: [B, Q, C]  video-level object-centric features
            last_attn: [B, Q, T*L]
            x_flat: [B, T*L, C]  注入 PE 后的归一化 memory（若外部一致性约束用到，请用这个版本）
        """
        BT, L, C = x_encoder.shape
        assert BT == B * T, f"Expected BT={B*T}, got {BT}"
        assert L <= self.num_spatial_tokens, \
            f"输入 token 数 L={L} 超过 num_spatial_tokens={self.num_spatial_tokens}，请按 backbone 调整该参数"

        # 关键：先归一化，再注入 PE（否则 PE 被深层特征大幅值淹没）
        x_video = self.pe_norm(x_encoder).view(B, T, L, C)
        t_pos, s_pos = self._get_pe(T, L, x_video.dtype)
        x_video = x_video + t_pos + s_pos                          # [B, T, L, C]
        x_flat = x_video.reshape(B, T * L, C)                      # [B, T*L, C]

        q = self.query_embed.expand(B, -1, -1)                     # [B, Q, C]

        # 文本条件化：query 读取全类别文本库（class-agnostic，无 GT 泄漏）
        if self.use_text_cond and (text_bank is not None) and (self.text_proj is not None):
            t = self.text_proj(text_bank.to(q.dtype))              # [Ncls, C]
            t = t.unsqueeze(0).expand(B, -1, -1)                   # [B, Ncls, C]
            cond = self.text_cross(self.norm_q_pre(q), self.norm_text(t))
            q = q + torch.tanh(self.text_cond_gate) * cond

        last_attn = None
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                q, attn = block(q, x_flat, return_attn=True)
                if attn is not None:
                    last_attn = attn
            else:
                q = block(q, x_flat)

        return self.norm(q), last_attn, x_flat

