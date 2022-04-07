# Trajectory Identification (TI)
> DifferenceNet as ann alternative to Siamese Net for TI


<p align="middle">
  <img src="https://github.com/FKTechLab/DifferenceNet/blob/main/ti/images/traj_half.png" width="50%" />
  <img src="https://github.com/FKTechLab/DifferenceNet/blob/main/ti/images/traj_full.png" width="50%" />
</p>

## How to use
`python -m ti.train -m DN -e 10 -p 1`

usage: train.py [-h] [-m {SN,DN}] [-e EPOCH] [-p PRINT]

optional arguments:

    -h, --help            show this help message and exit

    -m {SN,DN}, --model {SN,DN}
    Choose model, SN : Siamese Net, DN : Difference Net

    -e EPOCH, --epoch EPOCH
    Number of Epochs

    -p PRINT, --print PRINT
    Print at every p step, p must not be greater than e

### Transformer

```python
import pandas as pd
from ti.prep import Transformer

X = pd.read_csv('../data/sim/0.csv')
transformer = Transformer()
x1, x2 = transformer.transform(X)
print(len(x1), len(x2))
print(len(x1[0]), len(x2[0]))
```

    9 9
    4 4


### Model

#### Difference Net
This module helps create the difference features using (simple sub, abs diff or squared diff) for:
- (pos, anchor) or
- (neg, anchor)

```python
from torch.utils.data import DataLoader

from ti.dataloader import DatasetTraj, zero_padding
from ti.prep import Transformer
from ti.model import Difference

# Data generator
params = {
    'batch_size': 3,
    'shuffle': True,
    'collate_fn': zero_padding
}
train_g = DataLoader(DatasetTraj(range(0,4), [[0,1], [2,3], [1,3], [0, 1], [2,4], [6,0], [5,1], [9,3], [11,7], [8, 1], [10,4], [4,10]]), **params)
diff_s = Difference(mode='simple')
diff_a = Difference(mode='abs')
diff_sq = Difference(mode='square')
for x1, x2, y, x_seq_lens, max_seq_len in train_g:
    print(x1.shape)
    print('simple')
    print(diff_s(x1, x2[0])[0])
    print('abs')
    print(diff_a(x1, x2[0])[0])
    print('square')
    print(diff_sq(x1, x2[0])[0])
```

    torch.Size([3, 5, 4])
    simple
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0444, -0.0222, -0.1000, -0.1000],
            [ 0.0000, -0.0222,  0.0000,  0.2000]])
    abs
    tensor([[0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0444, 0.0222, 0.1000, 0.1000],
            [0.0000, 0.0222, 0.0000, 0.2000]])
    square
    tensor([[0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0020, 0.0005, 0.0100, 0.0100],
            [0.0000, 0.0005, 0.0000, 0.0400]])
    torch.Size([1, 3, 4])
    simple
    tensor([[ 0.0222, -0.0056,  0.0000,  0.0000],
            [ 0.0111, -0.0111,  0.5000,  0.0000],
            [ 0.0444, -0.0056,  0.4000,  0.5000]])
    abs
    tensor([[0.0222, 0.0056, 0.0000, 0.0000],
            [0.0111, 0.0111, 0.5000, 0.0000],
            [0.0444, 0.0056, 0.4000, 0.5000]])
    square
    tensor([[4.9383e-04, 3.0864e-05, 0.0000e+00, 0.0000e+00],
            [1.2346e-04, 1.2346e-04, 2.5000e-01, 0.0000e+00],
            [1.9753e-03, 3.0864e-05, 1.6000e-01, 2.5000e-01]])


#### Trajectory Identtification: Difference Net

![alt text](https://github.com/FKTechLab/DifferenceNet/blob/main/ti/images/TrajectoryDN.jpeg)

```python
from ti.model import TrajectoryDN
import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_net = Difference(mode='simple')
net = TrajectoryDN(diff_net, n_features=len(transformer.features_traj)*2) # 2x for org and dest 
# Forward Pass
count = 0
for x1, x2, y, x_seq_lens, max_seq_len in train_g:
    x1, y, x_seq_lens = torch.Tensor(x1).to(device), torch.Tensor(y).to(device), torch.Tensor(x_seq_lens).to(device)
    org = x2[0]
    dst = x2[1]
    org = torch.Tensor(org).to(device)
    dst = torch.Tensor(dst).to(device)
    x2 = [org, dst]
    print('Batch')
    print(x1.shape)
    print(x2[0].shape, x2[0].shape)
    print(y)
    print(x_seq_lens)
    print(max_seq_len)
    print(net(x1, x2, x_seq_lens))
    if count >= 0:
        break
    count+=1
```

    Batch
    torch.Size([3, 5, 4])
    torch.Size([3, 5, 4]) torch.Size([3, 5, 4])
    tensor([0., 1., 1.])
    tensor([5., 5., 4.])
    5
    tensor([[0.4676],
            [0.4676],
            [0.4675]], grad_fn=<SigmoidBackward>)


#### Trajectory Identtification: Siamese Net

![alt text](https://github.com/FKTechLab/DifferenceNet/blob/main/ti/images/TrajectorySN.jpeg)

```python
import torch
from ti.model import TrajectorySN, ContrastiveLoss

net = TrajectorySN(n_features=len(transformer.features_traj)) 
criterion = ContrastiveLoss()
# Forward Pass
count = 0
for x1, x2, y, x_seq_lens, max_seq_len in train_g:
    print('Batch')
    print(x1.shape)
    print(x2[0].shape, x2[0].shape)
    print(y)
    y = torch.Tensor(y)
    print(x_seq_lens)
    print(max_seq_len)
    x1, x2 = net(x1, x2, x_seq_lens)
    print(x1.shape, x2.shape)
    print(criterion(x1, x2, y).item())
    if count >= 0:
        break
    count+=1
```

    Batch
    torch.Size([3, 6, 4])
    torch.Size([3, 6, 4]) torch.Size([3, 6, 4])
    [0, 1, 1]
    [6, 3, 5]
    6
    torch.Size([3, 16]) torch.Size([3, 16])
    0.2942207455635071

