import math
import sys

sys.path.insert(0, "")

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import activation_factory


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
        residual=True,
        residual_kernel_size=1,
        activation="relu",
    ):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, (
            "# out channels should be multiples of # branches"
        )

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    activation_factory(activation),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
                for dilation in dilations
            ]
        )

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(branch_channels),
            )
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride,
            )

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out


class SelfAttentionTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_head=8,
        stride=1,  # 添加步长参数
        dropout=0.1,
        residual=True,
        activation="relu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_head = n_head
        self.stride = stride  # 保存步长值
        assert out_channels % n_head == 0, "out_channels must be divisible by n_head"

        # Projections
        self.c_attn = nn.Conv2d(in_channels, 3 * out_channels, kernel_size=1)
        self.c_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 如果需要下采样，添加池化层
        self.pool = None
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
                if stride > 1
                else nn.Identity(),
            )

        self.act = activation_factory(activation)

    def forward(self, x):
        # x: (N,C,T,V)
        N, C, T, V = x.size()
        res = self.residual(x)

        # 下采样输入（如果需要）
        if self.pool is not None:
            x = self.pool(x)
            _, _, T, _ = x.size()  # 更新T为下采样后的值

        # Project and reshape for attention
        # Using conv2d with kernel_size=1 instead of Linear
        qkv = self.c_attn(x)  # (N,3C',T,V)
        q, k, v = qkv.chunk(3, dim=1)  # Each (N,C',T,V)

        # Reshape for multi-head attention
        head_dim = self.out_channels // self.n_head
        q = q.view(N, self.n_head, head_dim, T, V)
        k = k.view(N, self.n_head, head_dim, T, V)
        v = v.view(N, self.n_head, head_dim, T, V)

        # Transpose for attention dot product: (N, n_head, T, V, head_dim)
        q = q.permute(0, 1, 3, 4, 2)
        k = k.permute(0, 1, 3, 4, 2)
        v = v.permute(0, 1, 3, 4, 2)

        # Reshape keys and queries for attention
        q_flat = q.reshape(N, self.n_head, T * V, head_dim)
        k_flat = k.reshape(N, self.n_head, T * V, head_dim)
        v_flat = v.reshape(N, self.n_head, T * V, head_dim)

        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        attn = (
            torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        )  # (N, n_head, T*V, T*V)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v_flat)  # (N, n_head, T*V, head_dim)
        out = out.reshape(N, self.n_head, T, V, head_dim)
        out = out.permute(0, 1, 4, 2, 3).reshape(N, self.out_channels, T, V)

        # Output projection
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        # Residual connection and activation
        out = out + res
        out = self.act(out)

        return out


class LinearSelfAttentionTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_head=8,
        stride=1,  # 添加步长参数
        dropout=0.1,
        residual=True,
        activation="relu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_head = n_head
        self.head_dim = out_channels // n_head
        self.stride = stride  # 保存步长值

        # Projections
        self.c_attn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.c_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 如果需要下采样，添加池化层
        self.pool = None
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))

        # TSSA specific parameters
        self.temp = nn.Parameter(torch.ones(n_head, 1))
        self.denom_bias = nn.Parameter(torch.zeros(n_head, 1, 1))

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
                if stride > 1
                else nn.Identity(),
            )

        self.act = activation_factory(activation)

    def forward(self, x):
        # x: (N,C,T,V)
        N, C, T, V = x.size()
        res = self.residual(x)

        # 下采样输入（如果需要）
        if self.pool is not None:
            x = self.pool(x)
            _, _, T, _ = x.size()  # 更新T为下采样后的值

        # Project inputs
        w = self.c_attn(x)  # (N,C',T,V)

        # Reshape for multi-head attention
        w = w.view(N, self.n_head, self.head_dim, T, V)
        w = w.permute(0, 1, 3, 4, 2)  # (N, n_head, T, V, head_dim)

        # Flatten the time and vertex dimensions for linear attention
        w = w.reshape(N, self.n_head, T * V, self.head_dim)

        # Linear attention calculations
        w_sq = w**2
        denom = torch.cumsum(w_sq, dim=2).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias

        tmp = torch.sum(w_normed, dim=-1) * self.temp  # (N, n_head, T*V)
        Pi = F.softmax(tmp, dim=2)

        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=2) / (
            Pi.cumsum(dim=2) + 1e-8
        ).unsqueeze(-1)
        attn = 1.0 / (1 + dots)
        attn = self.attn_dropout(attn)

        y = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        # Reshape back to original format
        y = y.reshape(N, self.n_head, T, V, self.head_dim)
        y = y.permute(0, 1, 4, 2, 3).reshape(N, self.out_channels, T, V)

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        # Residual connection and activation
        y = y + res
        y = self.act(y)

        return y


if __name__ == "__main__":
    mstcn = MultiScale_TemporalConv(288, 288)
    x = torch.randn(32, 288, 100, 20)
    mstcn.forward(x)
    for name, param in mstcn.named_parameters():
        print(f"{name}: {param.numel()}")
    print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))
