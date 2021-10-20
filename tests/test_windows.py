import torch
from torch_rw import rw

def test_to_windows_cpu():
    # create an array of walks
    torch.manual_seed(20)
    walks = torch.randint(low=0,high=30,size=(3,10))

    # to windows
    target_nodes, pos_windows, neg_windows = rw.to_windows(walks=walks,window_size=5,num_nodes=30,seed=20)
 
    assert target_nodes.size(0) == 6*3
    
    target_nodes_expected = torch.Tensor([27, 13, 24, 20, 13,  6]).to(int)
    pos_windows_expected = torch.Tensor([[11, 10, 13, 24],
                                [10, 27, 24, 20],
                                [27, 13, 20, 13],
                                [13, 24, 13,  6],
                                [24, 20,  6, 27],
                                [20, 13, 27,  0]]).to(int)

    neg_windows_expected = torch.Tensor([[ 1, 18, 17,  9],
                                        [26,  1, 22, 11],
                                        [10,  1, 20,  4],
                                        [17,  9, 14,  9],
                                        [25, 17, 29, 29],
                                        [15, 16, 11, 11]]).to(int)
    
    assert torch.equal(target_nodes[:6],target_nodes_expected)
    assert torch.equal(pos_windows[:6],pos_windows_expected)
    assert torch.equal(neg_windows[:6],neg_windows_expected)


def test_to_windows_gpu():
    # create an array of walks
    torch.manual_seed(20)
    walks = torch.randint(low=0,high=30,size=(3,10)).to("cuda")

    # to windows
    target_nodes, pos_windows, neg_windows = rw.to_windows(walks=walks,window_size=5,num_nodes=30,seed=20)
 
    assert target_nodes.size(0) == 6*3
    
    target_nodes_expected = torch.Tensor([27, 13, 24, 20, 13,  6]).to(int).to("cuda")
    pos_windows_expected = torch.Tensor([[11, 10, 13, 24],
                                        [10, 27, 24, 20],
                                        [27, 13, 20, 13],
                                        [13, 24, 13,  6],
                                        [24, 20,  6, 27],
                                        [20, 13, 27,  0]]).to(int).to("cuda")

    if torch.version.cuda:
        neg_windows_expected = torch.Tensor([[11, 27, 29, 14],
                                            [ 1, 12, 23, 24],
                                            [20, 22, 10,  7],
                                            [23, 29, 17, 19],
                                            [11, 27,  8,  4],
                                            [23,  6,  0,  8]]).to(int).to("cuda")
    else:
        neg_windows_expected = torch.Tensor([[16,  8, 18, 28],
                                            [18,  2, 14, 12],
                                            [28,  1, 20, 23],
                                            [ 1, 29, 29, 16],
                                            [28, 16, 10, 16],
                                            [ 0,  2,  7, 14]]).to(int).to("cuda")

    
    
    assert torch.equal(target_nodes[:6],target_nodes_expected)
    assert torch.equal(pos_windows[:6],pos_windows_expected)
    assert torch.equal(neg_windows[:6],neg_windows_expected)


def test_to_windows_triples_cpu():
    # create an array of walks
    torch.manual_seed(20)
    walk_length = (10*2)+1
    walks = torch.randint(low=0,high=30,size=(3,walk_length))
    triples = torch.randint(low=0,high=30,size=(10,3))

    # to window
    target_triples, pos_windows, neg_windows = rw.to_windows_triples(walks=walks,
                                                                     window_size=4,
                                                                     num_nodes=30,
                                                                     padding_idx=-1,
                                                                     triples=triples,
                                                                     seed=20)
    
    
    target_triples_expected = torch.Tensor([[11, 10, 27],
                                            [27, 13, 24]]).to(int)

    pos_windows_expected = torch.Tensor([[[-1, -1, 11],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [27, 13, 24],
                                        [24, 20, 13],
                                        [13,  6, 27],
                                        [27,  0,  7]],

                                        [[10, 10, 27],
                                        [-1, -1, 11],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [24, 20, 13],
                                        [13,  6, 27],
                                        [27,  0,  7],
                                        [ 7, 14, 20]]]).to(int)

    neg_windows_expected = torch.Tensor([[[18,  5, 19],
                                        [ 7, 25, 24],
                                        [10,  4, 14],
                                        [16, 24, 21],
                                        [20, 23, 10],
                                        [18,  5, 19],
                                        [20,  5, 14],
                                        [18,  5, 19]],

                                        [[29,  9, 17],
                                        [18,  5, 19],
                                        [29,  9, 17],
                                        [ 1,  8,  6],
                                        [10,  4, 14],
                                        [16, 24, 21],
                                        [ 1,  8,  6],
                                        [16, 24, 21]]]).to(int)


    assert torch.equal(target_triples[:2],target_triples_expected)
    assert torch.equal(pos_windows[:2],pos_windows_expected)
    assert torch.equal(neg_windows[:2],neg_windows_expected)


def test_to_windows_triples_cuda():
    # create an array of walks
    torch.manual_seed(20)
    walk_length = (10*2)+1
    walks = torch.randint(low=0,high=30,size=(3,walk_length)).cuda()
    triples = torch.randint(low=0,high=30,size=(10,3)).cuda()

    # to window
    target_triples, pos_windows, neg_windows = rw.to_windows_triples(walks=walks,
                                                                     window_size=4,
                                                                     num_nodes=30,
                                                                     padding_idx=-1,
                                                                     triples=triples,
                                                                     seed=20)
    
    
    target_triples_expected = torch.Tensor([[11, 10, 27],
                                            [27, 13, 24]]).to(int).cuda()

    pos_windows_expected = torch.Tensor([[[-1, -1, 11],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [27, 13, 24],
                                        [24, 20, 13],
                                        [13,  6, 27],
                                        [27,  0,  7]],

                                        [[10, 10, 27],
                                        [-1, -1, 11],
                                        [-1, -1, -1],
                                        [-1, -1, -1],
                                        [24, 20, 13],
                                        [13,  6, 27],
                                        [27,  0,  7],
                                        [ 7, 14, 20]]]).to(int).cuda()

    neg_windows_expected = torch.Tensor([[[18,  5, 19],
                                        [10,  4, 14],
                                        [16, 24, 21],
                                        [ 1,  8,  6],
                                        [18,  5, 19],
                                        [20,  5, 14],
                                        [26, 20, 23],
                                        [ 1,  8,  6]],

                                        [[29,  9, 17],
                                        [20,  5, 14],
                                        [29,  9, 17],
                                        [10,  4, 14],
                                        [26, 20, 23],
                                        [16, 24, 21],
                                        [10,  4, 14],
                                        [16, 24, 21]]]).to(int).cuda()

    assert torch.equal(target_triples[:2],target_triples_expected)
    assert torch.equal(pos_windows[:2],pos_windows_expected)
    assert torch.equal(neg_windows[:2],neg_windows_expected)


def test_to_windows_triples_cbow_cpu():
    # create an array of walks
    torch.manual_seed(20)
    walk_length = (10*2)+1
    walks = torch.randint(low=0,high=30,size=(3,walk_length))
    triples = torch.randint(low=0,high=30,size=(10,3))

    # to window
    pos_triples, neg_triples, windows = rw.to_windows_triples_cbow(walks=walks,
                                                                     window_size=4,
                                                                     num_nodes=30,
                                                                     padding_idx=-1,
                                                                     triples=triples,
                                                                     seed=20)


    pos_triples_expected = torch.Tensor([[11, 10, 27],
                                        [27, 13, 24]]).to(int)

    neg_triples_expected = torch.Tensor([[18,  5, 19],
                                        [ 7, 25, 24]]).to(int)

    windows_expected = torch.Tensor([[[-1, -1, 11],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [27, 13, 24],
                                    [24, 20, 13],
                                    [13,  6, 27],
                                    [27,  0,  7]],

                                    [[10, 10, 27],
                                    [-1, -1, 11],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [24, 20, 13],
                                    [13,  6, 27],
                                    [27,  0,  7],
                                    [ 7, 14, 20]]]).to(int)

    assert torch.equal(pos_triples[:2],pos_triples_expected)
    assert torch.equal(neg_triples[:2],neg_triples_expected)
    assert torch.equal(windows[:2],windows_expected)


def test_to_windows_triples_cbow_gpu():
    # create an array of walks
    torch.manual_seed(20)
    walk_length = (10*2)+1
    walks = torch.randint(low=0,high=30,size=(3,walk_length)).cuda()
    triples = torch.randint(low=0,high=30,size=(10,3)).cuda()

    # to window
    pos_triples, neg_triples, windows = rw.to_windows_triples_cbow(walks=walks,
                                                                     window_size=4,
                                                                     num_nodes=30,
                                                                     padding_idx=-1,
                                                                     triples=triples,
                                                                     seed=20)

    pos_triples_expected = torch.Tensor([[11, 10, 27],
                                        [27, 13, 24]]).to(int).cuda()

    neg_triples_expected = torch.Tensor([[18,  5, 19],
                                        [10,  4, 14]]).to(int).cuda()

    windows_expected = torch.Tensor([[[-1, -1, 11],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [27, 13, 24],
                                    [24, 20, 13],
                                    [13,  6, 27],
                                    [27,  0,  7]],

                                    [[10, 10, 27],
                                    [-1, -1, 11],
                                    [-1, -1, -1],
                                    [-1, -1, -1],
                                    [24, 20, 13],
                                    [13,  6, 27],
                                    [27,  0,  7],
                                    [ 7, 14, 20]]]).to(int).cuda()

    assert torch.equal(pos_triples[:2],pos_triples_expected)
    assert torch.equal(neg_triples[:2],neg_triples_expected)
    assert torch.equal(windows[:2],windows_expected)
    




