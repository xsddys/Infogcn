import os
import sys
import unittest

import networkx as nx
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.ucla import Graph, inward, num_node


class TestUCLABoneDualGraph(unittest.TestCase):
    def setUp(self):
        # 创建启用骨骼对偶图的Graph实例
        self.graph = Graph(use_bone_dual=True)

    def test_node_count(self):
        """测试骨骼对偶图的节点数量是否等于骨骼数量"""
        self.assertEqual(self.graph.num_node, len(inward))
        print(f"骨骼对偶图的节点数: {self.graph.num_node}")

    def test_dual_edge_correctness(self):
        """测试骨骼对偶图的边是否正确生成"""
        # 手动构建原始骨架图
        G = nx.Graph()
        for i in range(num_node):
            G.add_node(i)
        for i, j in inward:
            G.add_edge(i, j)

        # 生成线图
        L = nx.line_graph(G)

        # 创建同样的边到索引映射
        edge_to_idx = {}
        for idx, (i, j) in enumerate(inward):
            edge = tuple(sorted([i, j]))
            edge_to_idx[edge] = idx

        # 验证每条生成的边是否在NetworkX的结果中
        expected_edges = []
        for e1, e2 in L.edges():
            idx1 = edge_to_idx[tuple(sorted(e1))]
            idx2 = edge_to_idx[tuple(sorted(e2))]
            expected_edges.append((idx1, idx2))

        # 排序后比较
        sorted_expected = sorted([tuple(sorted(e)) for e in expected_edges])
        sorted_actual = sorted([tuple(sorted(e)) for e in self.graph.bone_dual_edges])

        self.assertEqual(sorted_actual, sorted_expected)
        print(f"骨骼对偶图的边数: {len(self.graph.bone_dual_edges)}")

    def test_adjacency_matrix_shape(self):
        """测试邻接矩阵的形状是否正确"""
        self.assertEqual(
            self.graph.A.shape, (3, self.graph.num_node, self.graph.num_node)
        )
        print(f"邻接矩阵形状: {self.graph.A.shape}")

    def test_self_connections(self):
        """测试自连接是否正确（矩阵第一层应该是单位矩阵）"""
        np.testing.assert_array_equal(self.graph.A[0], np.eye(self.graph.num_node))

    def test_edge_representation(self):
        """测试边在邻接矩阵中的表示是否正确"""
        # 验证所有的边都在邻接矩阵中
        for i, j in self.graph.bone_dual_edges:
            self.assertEqual(self.graph.A[1, i, j], 1, f"边 {i}->{j} 在邻接矩阵中缺失")
            self.assertEqual(self.graph.A[2, j, i], 1, f"边 {j}->{i} 在邻接矩阵中缺失")

        # 验证邻接矩阵中的边数与骨骼对偶图边列表中的边数一致
        edge_count_in_matrix = int(np.sum(self.graph.A[1]))
        self.assertEqual(edge_count_in_matrix, len(self.graph.bone_dual_edges))
        print(f"邻接矩阵中的边数: {edge_count_in_matrix}")

    def test_A_outward_binary(self):
        """测试A_outward_binary是否正确生成"""
        # 验证维度
        self.assertEqual(
            self.graph.A_outward_binary.shape,
            (self.graph.num_node, self.graph.num_node),
        )

        # 验证边的表示
        for i, j in self.graph.bone_dual_outward:
            self.assertEqual(
                self.graph.A_outward_binary[i, j],
                1,
                f"outward边 {i}->{j} 在A_outward_binary中缺失",
            )

    def test_matrix_consistency(self):
        """测试邻接矩阵的各表示形式是否一致"""
        # A_binary应该是将A[1]和A[2]合并后的结果
        expected_binary = (self.graph.A[1] + self.graph.A[2] > 0).astype(int)
        np.testing.assert_array_equal(self.graph.A_binary, expected_binary)


if __name__ == "__main__":
    unittest.main()
