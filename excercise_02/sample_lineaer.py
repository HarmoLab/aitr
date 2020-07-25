import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.rand(1, 3)
print("x")
print(x)
print()

W1 = nn.Linear(3, 2, bias=False)
print("W1 weight")
print(W1.weight)
print()
h = F.relu(W1(x))
print("h")
print(h)
print()

W2 = nn.Linear(2, 1, bias=False)
print("W2 weight")
print(W2.weight)
print()
y = F.relu(W2(h))
print("y")
print(y)