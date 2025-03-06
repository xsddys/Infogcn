import unittest

import networkx as nx


class TestBoneDualGraph(unittest.TestCase):
    def test_edge_to_index_mapping(self):
        """测试骨骼对偶图的边到索引映射"""
        # 设置一个简单的测试图，类似UCLA骨架的简化版
        # 0 -- 1 -- 2
        #      |    |
        #      3 -- 4
        # 边: (0,1), (1,2), (1,3), (3,4), (4,2)

        test_inward = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 2)]

        # 1. 创建原始图
        G = nx.Graph()
        for i in range(5):  # 5个节点
            G.add_node(i)

        for i, j in test_inward:
            G.add_edge(i, j)

        # 2. 创建线图
        L = nx.line_graph(G)

        # 3. 创建边到索引的映射，与代码中相同
        edge_to_idx = {}
        for idx, (i, j) in enumerate(test_inward):
            edge = tuple(sorted([i, j]))
            edge_to_idx[edge] = idx

        # 4. 构建骨骼对偶图的边列表
        dual_edges = []
        for e1, e2 in L.edges():
            idx1 = edge_to_idx[tuple(sorted(e1))]
            idx2 = edge_to_idx[tuple(sorted(e2))]
            dual_edges.append((idx1, idx2))

        # 5. 验证结果
        # 预期的对偶图边
        # 边(0,1)与边(1,2)相邻 => 对偶图中节点0与节点1相连
        # 边(0,1)与边(1,3)相邻 => 对偶图中节点0与节点2相连
        # 边(1,2)与边(1,3)相邻 => 对偶图中节点1与节点2相连
        # 边(1,2)与边(4,2)相邻 => 对偶图中节点1与节点4相连
        # 边(1,3)与边(3,4)相邻 => 对偶图中节点2与节点3相连
        # 边(3,4)与边(4,2)相邻 => 对偶图中节点3与节点4相连
        expected_edges = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4)]

        # 对结果排序以便比较
        dual_edges_sorted = sorted([tuple(sorted(e)) for e in dual_edges])
        expected_edges_sorted = sorted([tuple(sorted(e)) for e in expected_edges])

        self.assertEqual(
            dual_edges_sorted, expected_edges_sorted, "骨骼对偶图边映射不正确"
        )

        # 额外测试每条边的索引
        for i, j in test_inward:
            edge = tuple(sorted([i, j]))
            self.assertIn(edge, edge_to_idx, f"边{edge}没有被映射")


if __name__ == "__main__":
    unittest.main()
