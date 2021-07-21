import torch

#
# テンソル操作について
#

# 4 * 6 のテンソル
x = torch.tensor([[i + j * 6 for i in range(6)] for j in range(4)])
print("x")
print(x)
print(x.shape)
print()

#
# view
#

# 2 * 12 のテンソルに変換
y = x.view(2, 12)
print("y = x.view(2, 12)")
print(y)
print(y.shape)
print()

# -1 を指定すると良い感じにやってくれる
y = x.view(2, -1)
print("y = x.view(2, -1)")
print(y)
print(y.shape)
print()

# 次元(軸)を増やすことも可能
y = x.view(2, 2, 6)
print("y = x.view(2, 2, 6)")
print(y)
print(y.shape)
print()

#
# unsqueeze
#

y = x.unsqueeze(0)
print("y = x.unsqueeze(0)")
print(y)
print(y.shape)
print()

y = x.unsqueeze(1)
print("y = x.unsqueeze(1)")
print(y)
print(y.shape)
print()

#
# max
#