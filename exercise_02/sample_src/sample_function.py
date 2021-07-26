import torch
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.randn(3, 3)
print("x")
print(x)
print()
print()

# relu関数
y = F.relu(x)
print("relu関数は0未満の値を0にする関数．ニューラルネットワークの活性化関数としてよく使用される．")
print("y = F.relu(x)")
print(y)
print()
print()

# max関数
y = torch.max(x)
print("torch.max()は入力tensorの最大値を取得．")
print("y = torch.max(x)")
print(y)
print()

y, yi = torch.max(x, dim=0)
print("次元dim を指定すると指定した次元に沿った最大値とそのインデックスを取得")
print("y, yi = torch.max(x, dim=0)")
print(y, yi)
print()

y, yi = torch.max(x, dim=1)
print("y, yi = torch.max(x, dim=1)")
print(y, yi)
print()

print()
# softmax関数
x = torch.zeros(3, 3)
x[0][0], x[0][1], x[0][2] = 2.0, 0.4, -2.0
x[0][0], x[1][0], x[2][0] = 2.0, 0.4, -2.0
print("x")
print(x)
print()

y = F.softmax(x, dim=0)
print("softmax関数は「ベクトルの総和が1」かつ「各値が0以上」になるようにする関数．")
print("ニューラルネットワークの出力を確率として扱いたい場合に使用する．")
print("指定した次元dim に沿ってsoftmax関数を適用．")
print("y = F.softmax(x, dim=0)")
print(y)
print()

y = F.softmax(x, dim=1)
print("y = F.softmax(x, dim=1)")
print(y)
print()
