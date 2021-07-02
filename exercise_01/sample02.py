import torch

from PIL import Image
import torchvision

# tensor同士は四則演算可能です
# 各要素ごとに計算されます

x = torch.rand(3, 2)
y = torch.rand(3, 2)
print(x)
print(y)
print("加算", x + y)
print("原産", x - y)
print("積", x * y)
print("除算", x / y)

# いずれかをtensorではないただの数字にした場合，ブロードキャストされます
print(torch.zeros(2, 5) + 3)


# これらの演算子ではサイズが異なる場合は計算できません
# x = torch.rand(3, 2)
# y = torch.rand(5, 5)
# print(x + y)  # エラーで実行できない

# 和の計算や最大値の計算など集計用のメソッドも実装されています
z = torch.ones(2, 3)
print(z.sum())


x = torch.rand(10, 10)
# 要素へのアクセスは配列のようにできます
print(x[0, 1])

# また，0列目をすべて取り出す，のような操作も可能です
print(x[:, 0])

# 長方形状にアクセスすることも可能です
print(x[2:5, 3:8])
print(x[2:5, 3:8].size())  # サイズはこのように確認できます

# tensorの形を変形する操作は以下のように行います
# 軸の順番の入れ替えはtransposeで行います
x = torch.rand(3, 2)
print(x)
print(x.transpose(0, 1))  # 0軸目と1軸目を入れ替えます
