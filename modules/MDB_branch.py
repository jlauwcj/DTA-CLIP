import torch
import torch.nn as nn


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class TemporalMotionBranch(nn.Module):
    """
    替换冗余 cluster：显式提取帧间差分(运动)，产出 video-level 运动向量，
    gated 加到【闭集时序头】的输入上（不进对比/零样本路径）。
      - 隔离：只服务 temporal 闭集 track
      - 自禁用：gate=tanh(param)，init 0 → K400 若用不上会学到 ~0，零副作用；SSv2 激活
      - 输入端 GradientScale 限制对 backbone 的扰动
    """
    def __init__(self, dim, num_frames=8, num_patches=196, hidden=None, grad_scale=0.3):
        super().__init__()
        hidden = hidden or dim
        self.grad_scale = grad_scale
        self.in_norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)  # 空间深度卷积去噪
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden, dim)
        self.temporal_attn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))  # 时间注意力池化
        self.out_norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor([0.1]))   # 配合方案C(绕过 gate_oad)：init 0.1 即可，结构上已放大 motion 话语权

    def forward(self, x_tokens, B, T):
        # x_tokens: [B*T, L, C]  layer-11 patch token(不含 CLS)
        x = GradientScale.apply(x_tokens, self.grad_scale)
        x = self.in_norm(x)
        BT, L, C = x.shape
        H = W = int(round(L ** 0.5))
        x = x.view(B, T, L, C)
        motion = x[:, 1:] - x[:, :-1]                      # 帧间差分 [B, T-1, L, C]
        Tm = T - 1
        m = self.proj_in(motion)                           # [B, Tm, L, hid]
        m = m.reshape(B * Tm, H, W, -1).permute(0, 3, 1, 2)
        m = self.act(self.dwconv(m))                       # 空间 dwconv
        m = m.permute(0, 2, 3, 1).reshape(B, Tm, L, -1)
        m = self.proj_out(m).mean(dim=2)                   # 空间池化 [B, Tm, C]
        w = self.temporal_attn(m).softmax(dim=1)           # 时间注意力 [B, Tm, 1]
        motion_feat = self.out_norm((w * m).sum(dim=1))    # video-level [B, C]
        g = torch.tanh(self.gate)
        return g * motion_feat, g.detach().item()
