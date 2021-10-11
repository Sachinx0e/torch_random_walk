from networkx.classes.function import neighbors
import torch
from torch_rw import utils
from torch_rw import rw
import networkx as nx
import unittest
from loguru import logger


class TriplesTest(unittest.TestCase):

    def test_uniform_walk_edge_triples_cpu(self):
        
        # entity index
        A = 0
        B = 1
        C = 2
        D = 3
        E = 4

        # relation index
        r1 = 5
        r2 = 6
        r3 = 7

        # triples
        tripels_list = [
            (A,r1,B),
            (B,r2,D),
            (A,r1,C),
            (C,r2,E),
            (C,r3,B),
            (A,r2,D),
            (D,r3,A),
            (D,r2,C)
        ]

        triples_tensor = torch.Tensor(tripels_list)

        # target nodes
        target_nodes_list = list(set(triples_tensor[:,0].tolist()+triples_tensor[:,2].tolist()))
        target_nodes = torch.Tensor(target_nodes_list).to(int)
        
        # build relation_tail index
        relation_tail_index,triples_tensor_sorted = utils.build_relation_tail_index(triples_tensor)

        relation_tail_index_actual = torch.Tensor([[ 0,  2],
                                                [ 3,  3],
                                                [ 4,  5],
                                                [ 6,  7],
                                                [-1, -1]]).to(int)

        self.assertTrue(torch.equal(relation_tail_index,relation_tail_index_actual),"relation_tail_index does not match the ground truth")

        # last entity will be the padding index
        padding_idx = r3 + 1

        target_nodes = target_nodes.repeat_interleave(2,0)

        # perform walk
        walks = rw.walk_triples(triples_indexed=triples_tensor_sorted,
                                relation_tail_index=relation_tail_index,
                                target_nodes=target_nodes,
                                walk_length=6,
                                seed=10,
                                padding_idx=padding_idx,
                                restart=False
                                )
        
        walks_gt = torch.Tensor([[0, 5, 2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8],
                                [0, 6, 3, 6, 2, 6, 4, 8, 8, 8, 8, 8, 8],
                                [1, 6, 3, 6, 2, 7, 1, 6, 3, 6, 2, 7, 1],
                                [1, 6, 3, 6, 2, 7, 1, 6, 3, 6, 2, 6, 4],
                                [2, 7, 1, 6, 3, 7, 0, 5, 2, 6, 4, 8, 8],
                                [2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                [3, 6, 2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8],
                                [3, 7, 0, 5, 2, 7, 1, 6, 3, 6, 2, 6, 4],
                                [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]).to(int)

        self.assertTrue(torch.equal(walks,walks_gt),"Triple walks on cpu do not match the ground truth")


    def test_uniform_walk_edge_triples_gpu(self):
        
        # entity index
        A = 0
        B = 1
        C = 2
        D = 3
        E = 4

        # relation index
        r1 = 5
        r2 = 6
        r3 = 7

        # triples
        tripels_list = [
            (A,r1,B),
            (B,r2,D),
            (A,r1,C),
            (C,r2,E),
            (C,r3,B),
            (A,r2,D),
            (D,r3,A),
            (D,r2,C)
        ]

        triples_tensor = torch.Tensor(tripels_list)

        # target nodes
        target_nodes_list = list(set(triples_tensor[:,0].tolist()+triples_tensor[:,2].tolist()))
        target_nodes = torch.Tensor(target_nodes_list).to(int)
        
        # build relation_tail index
        relation_tail_index,triples_tensor_sorted = utils.build_relation_tail_index(triples_tensor)

        relation_tail_index_actual = torch.Tensor([[ 0,  2],
                                                [ 3,  3],
                                                [ 4,  5],
                                                [ 6,  7],
                                                [-1, -1]]).to(int)

        self.assertTrue(torch.equal(relation_tail_index,relation_tail_index_actual),"relation_tail_index does not match the ground truth")

        # last entity will be the padding index
        padding_idx = r3 + 1

        target_nodes = target_nodes.repeat_interleave(2,0)

        # move to gpu
        target_nodes = target_nodes.cuda()
        relation_tail_index = relation_tail_index.cuda()
        triples_tensor_sorted = triples_tensor_sorted.cuda()

        # perform walk
        walks = rw.walk_triples(triples_indexed=triples_tensor_sorted,
                                relation_tail_index=relation_tail_index,
                                target_nodes=target_nodes,
                                walk_length=6,
                                seed=10,
                                padding_idx=padding_idx,
                                restart=False
                                )
        
        print(walks)

        walks_gt = torch.Tensor([[0, 5, 2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8],
                                [0, 6, 3, 6, 2, 6, 4, 8, 8, 8, 8, 8, 8],
                                [1, 6, 3, 6, 2, 7, 1, 6, 3, 6, 2, 7, 1],
                                [1, 6, 3, 6, 2, 7, 1, 6, 3, 6, 2, 6, 4],
                                [2, 7, 1, 6, 3, 7, 0, 5, 2, 6, 4, 8, 8],
                                [2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                [3, 6, 2, 6, 4, 8, 8, 8, 8, 8, 8, 8, 8],
                                [3, 7, 0, 5, 2, 7, 1, 6, 3, 6, 2, 6, 4],
                                [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]).to(int)

        self.assertTrue(torch.equal(walks,walks_gt),"Triple walks on gpu do not match the ground truth")
        

if __name__ == '__main__':
    unittest.main()