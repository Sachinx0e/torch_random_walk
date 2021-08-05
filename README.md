# Pytorch Random Walk
Author : Sachin Gavali


## Requirements
```
1. Pytorch >= 1.9.0
2. NVIDIA-GPU (Cuda Toolkit >= 10.1)
3. AMD-GPU (ROCM == 4.0.1)
4. Python == 3.8
```

## Peform Random Walk

```
import torch
from torch_rw import utils
from torch_rw import rw
import networkx as nx

# create graph
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

# perform walks
walks = rw.walk(row_ptr=row_ptr,col_idx=col_idx,target_nodes=nodes,p=1.0,q=1.0,walk_length=6,seed=10)
```