import torch
import torch_rw

arr = torch.Tensor([0,1,3,4])
walks = torch_rw.walk(arr)
print(walks)