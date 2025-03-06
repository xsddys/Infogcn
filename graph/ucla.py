import sys

import networkx as nx  # 添加NetworkX依赖
import numpy as np

sys.path.extend(["../"])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2),
    (2, 3),
    (4, 3),
    (5, 3),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 3),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode="spatial", scale=1, use_bone_dual=False):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        # 如果使用骨骼对偶图，则先生成骨骼对偶图，再基于它构建邻接矩阵
        if use_bone_dual:
            self.bone_dual_edges = self.create_bone_dual_graph()

            # 定义对偶图的inward和outward
            self.bone_dual_inward = self.bone_dual_edges
            self.bone_dual_outward = [(j, i) for (i, j) in self.bone_dual_edges]

            # 对于对偶图，节点数等于原始图中的边数
            self.num_node = len(self.inward)  # 更新为骨骼数量

            self.A = self.get_dual_adjacency_matrix()

            # 添加A_outward_binary属性，用于对偶图
            self.A_outward_binary = tools.get_adjacency_matrix(
                self.bone_dual_outward, self.num_node
            )

            # 更新其他骨骼对偶图的属性
            self.A_binary = tools.edge2mat(
                self.bone_dual_edges + self.bone_dual_outward, self.num_node
            )
            self.A_norm = tools.normalize_adjacency_matrix(
                self.A_binary + 2 * np.eye(self.num_node)
            )
            self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)
        else:
            # 原始图处理逻辑
            self.A = self.get_adjacency_matrix(labeling_mode)
            self.A_outward_binary = tools.get_adjacency_matrix(
                self.outward, self.num_node
            )
            self.A_binary = tools.edge2mat(neighbor, self.num_node)
            self.A_norm = tools.normalize_adjacency_matrix(
                self.A_binary + 2 * np.eye(self.num_node)
            )
            self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def create_bone_dual_graph(self):
        """创建骨骼对偶图（线图）并定义基于根骨骼的向心/离心关系"""
        # 构建原始关节图
        G = nx.Graph()
        for i in range(num_node):
            G.add_node(i)

        # 添加边（即骨骼）
        for i, j in inward:
            G.add_edge(i, j)

        # 生成线图
        L = nx.line_graph(G)

        # 创建边到索引的映射
        edge_to_idx = {}
        idx_to_edge = {}
        for idx, (i, j) in enumerate(inward):
            edge = tuple(sorted([i, j]))
            edge_to_idx[edge] = idx
            idx_to_edge[idx] = edge

        # 构建骨骼对偶图的边列表
        dual_edges = []
        for e1, e2 in L.edges():
            idx1 = edge_to_idx[tuple(sorted(e1))]
            idx2 = edge_to_idx[tuple(sorted(e2))]
            dual_edges.append((idx1, idx2))

        # 定义骨骼对偶图中的根节点（对应于第一条边，即头部-躯干连接）
        root_bone_idx = 0  # 第一条骨骼通常是连接头部和躯干的骨骼

        # 创建对偶图结构用于计算距离
        DG = nx.Graph()
        for i in range(len(inward)):
            DG.add_node(i)
        for i, j in dual_edges:
            DG.add_edge(i, j)

        # 计算每个骨骼节点到根骨骼的距离
        distances = nx.shortest_path_length(DG, source=root_bone_idx)

        # 基于距离定义向心和离心关系
        bone_dual_inward = []  # 向心（朝向根部）
        bone_dual_outward = []  # 离心（远离根部）

        for i, j in dual_edges:
            if distances[j] < distances[i]:  # j比i更接近根部
                bone_dual_inward.append((i, j))
                bone_dual_outward.append((j, i))
            elif distances[i] < distances[j]:  # i比j更接近根部
                bone_dual_inward.append((j, i))
                bone_dual_outward.append((i, j))
            else:  # 距离相同，使用索引比较或其他规则
                if i < j:
                    bone_dual_inward.append((j, i))
                    bone_dual_outward.append((i, j))
                else:
                    bone_dual_inward.append((i, j))
                    bone_dual_outward.append((j, i))

        # 存储定义的向心和离心关系
        self.bone_dual_inward = bone_dual_inward
        self.bone_dual_outward = bone_dual_outward

        return dual_edges

    def get_dual_adjacency_matrix(self):
        """为骨骼对偶图生成基于层次距离的邻接矩阵"""
        num_bones = self.num_node

        # 创建自连接
        I = np.eye(num_bones)

        # 创建并归一化向心和离心连接
        In = tools.normalize_digraph(
            tools.get_adjacency_matrix(self.bone_dual_inward, num_bones)
        )
        Out = tools.normalize_digraph(
            tools.get_adjacency_matrix(self.bone_dual_outward, num_bones)
        )

        # 堆叠三种连接
        A = np.stack((I, In, Out))

        return A

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
