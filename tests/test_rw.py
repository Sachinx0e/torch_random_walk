import torch
from torch_rw import utils
from torch_rw import rw
import networkx as nx
import unittest
from loguru import logger

class MainTest(unittest.TestCase):
    def test_add(self):
        graph = nx.DiGraph()

        # add edge
        graph.add_edge("A","B")
        graph.add_edge("A","C")
        graph.add_edge("B","D")
        graph.add_edge("D","C")
        graph.add_edge("E","A")
        graph.add_edge("E","D")

        edge_index = utils.edge_tensor(graph)
        nodes = utils.nodes_tensor(graph)

        walks = rw.walk(edge_indices=edge_index,target_nodes=nodes,p=1.0,q=1.0,walk_length=10)
        print(walks)

        # test that 1 + 1 = 2
        self.assertEqual(5, 5)


if __name__ == '__main__':
    unittest.main()

