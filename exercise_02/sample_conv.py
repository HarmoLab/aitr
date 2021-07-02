import torch
import torch.nn as nn

# バッチの次元 = 1, チャネルの次元 = 1, 画像の次元(縦) = 5, 画像の次元(横)  = 5
x = torch.zeros(1, 1, 5, 5)
x[:, :, 2, :] = 1
x[:, :, :, 2] = 1
print("image")
print(x)
print()

# この x を conv に入力するとエラーになります。
# バッチの次元 と チャネルの次元 が必要です。
# x = torch.zeros(5, 5)
# x[2, :] = 1
# x[:, 2] = 1

conv = nn.Conv2d(1, 1, 3, bias=False,
                 padding=1, padding_mode="zeros",
                 stride=2)
conv.weight[0][0][:, 0] = -1
conv.weight[0][0][:, 1] = 0
conv.weight[0][0][:, 2] = 1
print("filter")
print(conv.weight)
print()

y = conv(x)
print("image after convolution")
print(y)