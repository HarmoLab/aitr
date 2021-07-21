import torch
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.randn(3, 3)
print("x")
print(x)
print()

# relu
y = F.relu(x)
print("y = F.relu(x)")
print(y)
print()

# sigmoid
y = torch.sigmoid(x)
print("y = torch.nn.sigmoid(x)")
print(y)
print()

# max
y = torch.max(x)
print("y = torch.max(x)")
print(y)
print()

y, yi = torch.max(x, dim=0)
print("y, yi = torch.max(x, dim=0)")
print(y, yi)
print()

y, yi = torch.max(x, dim=1)
print("y, yi = torch.max(x, dim=1)")
print(y, yi)
print()

print()
# softmax
x = torch.zeros(3, 3)
x[0][0], x[0][1], x[0][2] = 2.0, 0.4, -2.0
x[0][0], x[1][0], x[2][0] = 2.0, 0.4, -2.0
print("x")
print(x)
print()

y = F.softmax(x, dim=0)
print("y = F.softmax(x, dim=0)")
print(y)
print()

y = F.softmax(x, dim=1)
print("y = F.softmax(x, dim=1)")
print(y)
print()
