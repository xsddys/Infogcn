import numpy as np
import torch
from torch import nn

from model.cls_attn import ClassAttentionBlock, TemporalPositionalEncoding
from model.modules import EncodingBlockWithLinearAttention, import_class


class InfoGCN(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        in_channels=3,
        drop_out=0,
        num_head=3,
        noise_ratio=0.1,
        k=0,
        gain=1,
        use_temporal_pos=False,  # 新增：是否使用时间位置编码
    ):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel * 4)
        self.A_vector = self.get_A(graph, k)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        # 可选的时间位置编码
        self.use_temporal_pos = use_temporal_pos
        if use_temporal_pos:
            self.temporal_pos_embedding = TemporalPositionalEncoding(base_channel)

        # 原始编码块保持不变
        self.l1 = EncodingBlockWithLinearAttention(base_channel, base_channel, A)
        self.l2 = EncodingBlockWithLinearAttention(base_channel, base_channel, A)
        self.l3 = EncodingBlockWithLinearAttention(base_channel, base_channel, A)
        self.l4 = EncodingBlockWithLinearAttention(
            base_channel, base_channel * 2, A, stride=2
        )
        self.l5 = EncodingBlockWithLinearAttention(
            base_channel * 2, base_channel * 2, A
        )
        self.l6 = EncodingBlockWithLinearAttention(
            base_channel * 2, base_channel * 2, A
        )
        self.l7 = EncodingBlockWithLinearAttention(
            base_channel * 2, base_channel * 4, A, stride=2
        )

        # CLS令牌
        self.cls_token = nn.Parameter(torch.randn(1, base_channel * 4, 1, num_point))

        # 类注意力块
        self.cls_attn1 = ClassAttentionBlock(base_channel * 4, base_channel * 4)
        self.cls_attn2 = ClassAttentionBlock(base_channel * 4, base_channel * 4)

        # 分类器
        self.fc = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_mu = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_logvar = nn.Linear(base_channel * 4, base_channel * 4)
        self.decoder = nn.Linear(base_channel * 4, num_class)

        # 初始化
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(
            self.fc_logvar.weight, gain=nn.init.calculate_gain("relu")
        )

    def get_A(self, graph, k):
        if isinstance(graph, str):
            Graph = import_class(graph)()
        else:
            Graph = graph

        A_outward = Graph.A_outward_binary
        eye = np.eye(Graph.num_node)
        return torch.from_numpy(eye - np.linalg.matrix_power(A_outward, k)).float()

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        # 原始代码部分 - 处理输入
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V, C, T)
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M * V, T, C)

        x = self.to_joint_embedding(x)
        x = x.view(N * M, V, T, -1).permute(0, 3, 2, 1).contiguous()

        # 应用可选的时间位置编码
        if self.use_temporal_pos:
            x = self.temporal_pos_embedding(x)

        # 正常处理层 l1-l7
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # 添加CLS令牌
        B, C, T, V = x.size()
        cls_token = self.cls_token.expand(B, -1, -1, -1)

        # 应用类注意力块
        cls_out = self.cls_attn1(cls_token, x)
        cls_out = self.cls_attn2(cls_out, x)

        # 处理CLS令牌进行分类
        out = cls_out.mean(-1).squeeze(-1)  # 聚合顶点维度 (B, C)

        # 后续的处理与原始代码保持一致
        out = self.fc(out)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        y = self.decoder(z)
        return y, z

    def reparameterize(self, mu, logvar):
        # 原始代码保持不变
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
