from networkx.classes.function import neighbors
import torch
from torch_rw import utils
from torch_rw import rw
import networkx as nx
import unittest
from loguru import logger

def get_neighbors(node, nodes, row_ptr, col_idx):
    node_idx = nodes.index(node)
    
    row_start = node_idx
    row_end = node_idx + 1

    # column indices
    index_start = row_ptr[row_start]
    index_end = row_ptr[row_end]

    neighbors = col_idx[index_start:index_end]

    neighbors_name = []
    for n in neighbors:
        neighbors_name.append(nodes[n])

    return neighbors_name    

class MainTest(unittest.TestCase):


    def test_uniform_walk(self):
        graph = nx.Graph()

        # add edge
        graph.add_edge("A","B")
        graph.add_edge("A","C")
        graph.add_edge("B","C")
        graph.add_edge("B","D")
        graph.add_edge("D","C")
        graph.add_edge("E","A")
        graph.add_edge("E","D")

        # get csr
        row_ptr, col_idx = utils.to_csr(graph)
        nodes = utils.nodes_tensor(graph)

        walks = rw.walk(row_ptr=row_ptr,col_idx=col_idx,target_nodes=nodes,p=1.0,q=1.0,walk_length=6,seed=10)

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [3, 2, 1, 2, 1, 2, 1],
        [4, 3, 2, 1, 2, 1, 2]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks do not match")


if __name__ == '__main__':
    unittest.main()

