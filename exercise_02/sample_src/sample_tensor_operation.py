import torch

#
# テンソル操作について
#

# 4 * 6 のテンソルを作成
x = torch.tensor([[i + j * 6 for i in range(6)] for j in range(4)])
print("x")
print(x)
print(x.shape)
print()

# tensor と for文
print("listと同様に，2次元テンソルをfor文で回すと行ごとのテンソルを取得可能")
for row in x:
    print(row)
print()
print("listと同様に，1次元テンソルをfor文で回すと各要素を取得可能")
for element in x[0]:
    print(element)
print()
print()

# view関数について
print("view関数はテンソルのshapeを変更する関数．")
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

# 次元(軸)を増やすことも可能
y = x.view(1, 4, 6)
print("y = x.view(2, 2, 6)")
print(y)
print(y.shape)
print()

print()
# unsqueeze関数について
print("unsqueeze関数は，引数で指定した位置に次元を追加する関数．")
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