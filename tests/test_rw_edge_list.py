from networkx.classes.function import neighbors
import torch
from torch_rw import utils
from torch_rw import rw
import networkx as nx
import unittest
from loguru import logger


class MainTest(unittest.TestCase):

    def test_uniform_walk_edge_list_cpu(self):
        graph = nx.DiGraph()

        # add edge
        graph.add_edge("A","B")
        graph.add_edge("A","C")
        graph.add_edge("B","C")
        graph.add_edge("B","D")
        graph.add_edge("D","C")
        graph.add_edge("E","A")
        graph.add_edge("E","D")

        # get indexed edge list
        edge_list_indexed, node_idx_map = utils.to_edge_list_indexed(graph)
        target_nodes = torch.Tensor(list(node_idx_map.values())).to(int).contiguous()
        
        # get node_edge_index
        node_edge_index, edge_list_indexed = utils.build_node_edge_index(edge_list_indexed)
        
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1


        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx
                                )

        print(walks)

        # define actual walks
        walk_actual =torch.Tensor([[0, 1, 2, 5, 5, 5, 5],
                                    [1, 2, 5, 5, 5, 5, 5],
                                    [2, 5, 5, 5, 5, 5, 5],
                                    [3, 2, 5, 5, 5, 5, 5],
                                    [4, 0, 1, 2, 5, 5, 5]]).to(int)

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")

if __name__ == '__main__':
    unittest.main()

