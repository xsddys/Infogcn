import math

import torch
import torch.nn as nn


class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.heads = num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(dropout)

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.denom_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        h = self.heads

        # 投影并重新排列为多头格式
        w = self.qkv(x)  # (B, C, T, V)
        w = w.reshape(B, h, C // h, T, V)  # (B, h, C//h, T, V)
        w = w.permute(0, 1, 3, 4, 2)  # (B, h, T, V, C//h)

        # 扁平化时间和顶点维度
        w_flat = w.reshape(B, h, T * V, C // h)  # (B, h, T*V, C//h)

        # 线性注意力计算
        w_sq = w_flat**2
        denom = torch.cumsum(w_sq, dim=2).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias

        # 计算注意力权重
        tmp = torch.sum(w_normed, dim=-1) * self.temp  # (B, h, T*V)
        Pi = self.attend(tmp)  # (B, h, T*V)

        # 计算加权表示
        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=2) / (
            Pi.cumsum(dim=2) + 1e-8
        ).unsqueeze(-1)
        attn = 1.0 / (1 + dots)
        attn = self.attn_drop(attn)

        # 应用注意力
        y = -torch.mul(w_flat.mul(Pi.unsqueeze(-1)), attn)

        # 重塑回原始格式
        y = y.reshape(B, h, T, V, C // h)
        y = y.permute(0, 1, 4, 2, 3).reshape(B, C, T, V)

        # 输出投影
        y = self.proj(y)
        y = self.proj_drop(y)

        return y


class ClassAttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.heads = num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_cls, x_patch):
        # x_cls: (B, C, 1, V)  - CLS令牌
        # x_patch: (B, C, T, V) - 骨架特征

        B, C, _, V = x_cls.size()
        _, _, T, _ = x_patch.size()
        h = self.heads
        head_dim = C // h

        # 展平和变换以进行线性投影
        x_cls_flat = x_cls.reshape(B, C, V).permute(0, 2, 1)  # (B, V, C)
        x_patch_flat = x_patch.reshape(B, C, T * V).permute(0, 2, 1)  # (B, T*V, C)

        # 只对CLS令牌生成查询
        q = (
            self.q(x_cls_flat).reshape(B, V, h, head_dim).permute(0, 2, 1, 3)
        )  # (B, h, V, head_dim)

        # 为所有标记生成键值
        k = (
            self.k(x_patch_flat).reshape(B, T * V, h, head_dim).permute(0, 2, 1, 3)
        )  # (B, h, T*V, head_dim)
        v = (
            self.v(x_patch_flat).reshape(B, T * V, h, head_dim).permute(0, 2, 1, 3)
        )  # (B, h, T*V, head_dim)

        # 注意力计算：q与k的点积
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            head_dim
        )  # (B, h, V, T*V)

        # 应用温度参数
        attn = attn * self.temp.unsqueeze(-1)

        # 归一化和dropout
        attn = self.attend(attn)
        attn = self.attn_drop(attn)

        # 聚合值向量
        x = torch.matmul(attn, v)  # (B, h, V, head_dim)

        # 重组多头结果
        x = x.permute(0, 2, 1, 3).reshape(B, V, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        # 重塑回原始格式
        x = x.permute(0, 2, 1).reshape(B, C, 1, V)

        return x


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2) * (-math.log(10000.0) / channels)
        )

        pe = torch.zeros(1, channels, max_len, 1)
        pe[0, 0::2, :, 0] = torch.sin(position * div_term)
        pe[0, 1::2, :, 0] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, channels, time_steps, vertices]
        """
        _, _, T, _ = x.size()
        x = x + self.pe[:, :, :T]
        return self.dropout(x)


class ClassAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_head=8, dropout=0.1):
        super(ClassAttentionBlock, self).__init__()

        # 归一化层
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # 类注意力层
        self.cls_attn = ClassAttentionTSSA(
            in_channels, num_heads=n_head, dropout=dropout
        )

        # 输出投影
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_cls, x_patch):
        # 归一化
        x_cls_norm = self.norm1(x_cls)
        x_patch_norm = self.norm1(x_patch)

        # 应用类注意力
        cls_out = self.cls_attn(x_cls_norm, x_patch_norm)

        # 残差连接
        cls_out = x_cls + cls_out

        # 输出投影
        cls_out = self.proj(cls_out)
        cls_out = self.norm2(cls_out)
        cls_out = self.act(cls_out)
        cls_out = self.dropout(cls_out)

        return cls_out
