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

class CSRTest(unittest.TestCase):


    def test_uniform_walk_cpu(self):
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
        walk_actual =torch.Tensor([[0, 2, 1, 3, 4, 0, 4],
        [1, 3, 2, 3, 4, 3, 4],
        [2, 0, 1, 3, 2, 0, 2],
        [3, 4, 0, 1, 2, 1, 2],
        [4, 0, 4, 0, 2, 1, 0]]).to(int)

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks do not match")


    def test_uniform_walk_gpu(self):
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

        # move all tensors to gpu
        row_ptr = row_ptr.to("cuda")
        col_idx = col_idx.to("cuda")
        nodes = nodes.to("cuda")

        walks = rw.walk(row_ptr=row_ptr,col_idx=col_idx,target_nodes=nodes,p=1.0,q=1.0,walk_length=6,seed=10)

        # define actual walks
        if torch.version.cuda:
            walk_actual =torch.Tensor([[0, 4, 0, 1, 3, 4, 3],
                                        [1, 3, 4, 0, 4, 0, 4],
                                        [2, 0, 4, 3, 1, 0, 1],
                                        [3, 4, 0, 2, 3, 1, 3],
                                        [4, 3, 4, 3, 2, 3, 1]]).to(int).to("cuda")
        else:
            walk_actual =torch.Tensor([[0, 4, 3, 4, 3, 4, 3],
                                    [1, 3, 2, 0, 4, 3, 1],
                                    [2, 0, 2, 3, 4, 3, 2],
                                    [3, 2, 1, 0, 2, 0, 1],
                                    [4, 0, 4, 3, 1, 0, 2]]).to(int).to("cuda")

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks do not match for gpu")


    def test_biased_walk(self):
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

        walks = rw.walk(row_ptr=row_ptr,col_idx=col_idx,target_nodes=nodes,p=0.7,q=0.5,walk_length=6,seed=10)

        walk_actual = torch.Tensor([[0, 2, 3, 4, 3, 4, 3],
                                    [1, 2, 1, 2, 1, 0, 4],
                                    [2, 0, 2, 3, 4, 3, 2],
                                    [3, 2, 0, 4, 3, 4, 3],
                                    [4, 0, 4, 0, 2, 3, 4]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks do not match")

    def test_biased_walk_gpu(self):
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

        # move tensors to gpu
        row_ptr = row_ptr.to("cuda")
        col_idx = col_idx.to("cuda")
        nodes = nodes.to("cuda")

        walks = rw.walk(row_ptr=row_ptr,col_idx=col_idx,target_nodes=nodes,p=0.7,q=0.5,walk_length=6,seed=10)

        if torch.version.cuda:
            walk_actual = torch.Tensor([[0, 4, 0, 1, 0, 2, 0],
                                            [1, 3, 4, 0, 4, 0, 2],
                                            [2, 0, 4, 0, 1, 2, 0],
                                            [3, 4, 0, 4, 3, 1, 3],
                                            [4, 3, 2, 0, 4, 0, 4]]).to(int).to("cuda")
        else:
            walk_actual = torch.Tensor([[0, 4, 3, 1, 0, 4, 0],
                                        [1, 3, 2, 0, 4, 0, 1],
                                        [2, 0, 2, 3, 2, 0, 2],
                                        [3, 2, 1, 2, 0, 1, 0],
                                        [4, 0, 1, 2, 1, 0, 1]]).to(int).to("cuda")

        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks do not match")


if __name__ == '__main__':
    unittest.main()

