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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1


        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 0, 1, 2, 5],
                                    [1, 3, 2, 5, 1, 2, 5],
                                    [2, 5, 2, 5, 2, 5, 2],
                                    [3, 2, 5, 3, 2, 5, 3],
                                    [4, 3, 2, 5, 4, 3, 2]]).to(int)

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")

    
    def test_uniform_walk_edge_list_cpu_no_restart(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1


        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx,
                                  restart=False
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 5, 5, 5, 5],
                                    [1, 2, 5, 5, 5, 5, 5],
                                    [2, 5, 5, 5, 5, 5, 5],
                                    [3, 2, 5, 5, 5, 5, 5],
                                    [4, 0, 2, 5, 5, 5, 5]]).to(int)

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")


    def test_uniform_walk_edge_list_gpu(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 0, 1, 2, 5],
                                    [1, 3, 2, 5, 1, 2, 5],
                                    [2, 5, 2, 5, 2, 5, 2],
                                    [3, 2, 5, 3, 2, 5, 3],
                                    [4, 3, 2, 5, 4, 3, 2]]).to(int).cuda()

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")


    def test_uniform_walk_edge_list_gpu_restart(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx,
                                  restart=False
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 5, 5, 5, 5],
                                    [1, 3, 2, 5, 5, 5, 5],
                                    [2, 5, 5, 5, 5, 5, 5],
                                    [3, 2, 5, 5, 5, 5, 5],
                                    [4, 3, 2, 5, 5, 5, 5]]).to(int).cuda()

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")


    def test_uniform_walk_edge_list_cpu_undirected(self):
        graph = nx.Graph()

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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)

        node_edge_index_gt = torch.Tensor([[ 0,  2],
                                            [ 3,  5],
                                            [ 6,  8],
                                            [ 9, 11],
                                            [12, 13]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1


        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 0, 4, 3, 4, 3],
                                    [1, 0, 2, 1, 0, 4, 3],
                                    [2, 3, 4, 0, 2, 3, 1],
                                    [4, 3, 4, 0, 2, 0, 2],
                                    [3, 1, 0, 2, 0, 2, 3]]).to(int)

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks on edge list do not match")

    def test_uniform_walk_edge_list_gpu_undirected(self):
        graph = nx.Graph()

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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)

        node_edge_index_gt = torch.Tensor([[ 0,  2],
                                            [ 3,  5],
                                            [ 6,  8],
                                            [ 9, 11],
                                            [12, 13]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1


        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=1.0,
                                  q=1.0,
                                  walk_length=6,
                                  seed=10,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 4, 0, 1, 0, 4, 3],
                                    [1, 0, 4, 0, 4, 0, 4],
                                    [2, 3, 2, 3, 4, 0, 1],
                                    [4, 0, 1, 3, 2, 3, 2],
                                    [3, 2, 1, 3, 1, 0, 1]]).to(int).cuda()

        self.assertTrue(torch.equal(walks,walk_actual),"Uniform sampling walks undirected on edge list gpu do not match")

    def test_biased_walk_edge_list_cpu(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 0, 1, 3, 2, 0],
                                    [1, 3, 2, 1, 3, 2, 1],
                                    [2, 5, 2, 5, 2, 5, 2],
                                    [3, 2, 3, 2, 3, 2, 3],
                                    [4, 0, 1, 3, 2, 4, 0]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks do not match")


    def test_biased_walk_edge_list_cpu_restart(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx,
                                  restart=False
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 5, 5, 5, 5],
                                    [1, 3, 2, 5, 5, 5, 5],
                                    [2, 5, 5, 5, 5, 5, 5],
                                    [3, 2, 5, 5, 5, 5, 5],
                                    [4, 0, 2, 5, 5, 5, 5]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks do not match")


    def test_biased_walk_edge_list_gpu(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 0, 2, 0, 1, 2],
                                    [1, 3, 2, 1, 2, 1, 2],
                                    [2, 5, 2, 5, 2, 5, 2],
                                    [3, 2, 3, 2, 3, 2, 3],
                                    [4, 3, 2, 4, 3, 2, 4]]).to(int).cuda()
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks on gpu do not match")


    def test_biased_walk_edge_list_gpu_restart(self):
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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        
        node_edge_index_gt = torch.Tensor([[ 0,  1],
                                            [ 2,  3],
                                            [-1, -1],
                                            [ 4,  4],
                                            [ 5,  6]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx,
                                  restart=False
                                )

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 5, 5, 5, 5, 5],
                                    [1, 3, 2, 5, 5, 5, 5],
                                    [2, 5, 5, 5, 5, 5, 5],
                                    [3, 2, 5, 5, 5, 5, 5],
                                    [4, 3, 2, 5, 5, 5, 5]]).to(int).cuda()
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks on gpu do not match")

    def test_biased_walk_edge_list_cpu_undirected(self):
        graph = nx.Graph()

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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  2],
                                            [ 3,  5],
                                            [ 6,  8],
                                            [ 9, 11],
                                            [12, 13]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx
                                )

        

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 3, 4, 3, 2, 0],
                                    [1, 3, 2, 0, 4, 3, 2],
                                    [2, 0, 4, 3, 1, 0, 4],
                                    [4, 3, 1, 0, 4, 3, 4],
                                    [3, 4, 0, 1, 0, 4, 3]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling walks do not match")

    def test_biased_walk_edge_list_gpu_undirected(self):
        graph = nx.Graph()

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
        nodes_tensor = torch.unique(edge_list_indexed.view(-1))
        node_edge_index, edge_list_indexed_sorted = utils.build_node_edge_index(edge_list_indexed,nodes_tensor)
        node_edge_index_gt = torch.Tensor([[ 0,  2],
                                            [ 3,  5],
                                            [ 6,  8],
                                            [ 9, 11],
                                            [12, 13]]).to(int)

        self.assertTrue(torch.equal(node_edge_index,node_edge_index_gt),"Node edge index does not match the ground truth")

        # create a padding index
        padding_idx = sorted(target_nodes.tolist())[-1] + 1

        walks = rw.walk_edge_list(edge_list_indexed=edge_list_indexed_sorted,
                                  node_edge_index=node_edge_index,
                                  target_nodes=target_nodes,
                                  p=0.7,
                                  q=0.2,
                                  walk_length=6,
                                  seed=20,
                                  padding_idx=padding_idx
                                )

        # move to GPU
        edge_list_indexed_sorted = edge_list_indexed_sorted.cuda()
        node_edge_index = node_edge_index.cuda()
        target_nodes = target_nodes.cuda()

        # define actual walks
        walk_actual =torch.Tensor([[0, 2, 3, 4, 3, 2, 0],
                                    [1, 3, 2, 0, 4, 3, 2],
                                    [2, 0, 4, 3, 1, 0, 4],
                                    [4, 3, 1, 0, 4, 3, 4],
                                    [3, 4, 0, 1, 0, 4, 3]]).to(int)
        
        self.assertTrue(torch.equal(walks,walk_actual),"Biased sampling undirected walks on gpu do not match")


if __name__ == '__main__':
    unittest.main()

