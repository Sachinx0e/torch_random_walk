import torch
from torch_rw import rw

def test_to_windows_cpu():
    # create an array of walks
    torch.manual_seed(20)
    walks = torch.randint(low=0,high=30,size=(3,10))

    # to windows
    target_nodes, pos_windows, neg_windows = rw.to_windows(walks=walks,window_size=5,num_nodes=30)
 
    assert target_nodes.size(0) == 6*3
    
    target_nodes_expected = torch.Tensor([27, 13, 24, 20, 13,  6]).to(int)
    pos_windows_expected = torch.Tensor([[11, 10, 13, 24],
                                [10, 27, 24, 20],
                                [27, 13, 20, 13],
                                [13, 24, 13,  6],
                                [24, 20,  6, 27],
                                [20, 13, 27,  0]]).to(int)

    neg_windows_expected = torch.Tensor([[19, 16,  2,  2],
                                        [ 9, 17,  6,  3],
                                        [21, 24, 21, 29],
                                        [ 7, 19, 21, 26],
                                        [14, 26, 28, 17],
                                        [ 4,  7, 12, 29]]).to(int)

    
    assert torch.equal(target_nodes[:6],target_nodes_expected)
    assert torch.equal(pos_windows[:6],pos_windows_expected)
    assert torch.equal(neg_windows[:6],neg_windows_expected)


