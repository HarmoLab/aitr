# インストールしたパッケージをコード内で使用できるようにインポートします
import torch

# pytorchはディープラーニングに関する数値計算のためのライブラリとなっています．
# まずは初期化されていない行列(配列として表現される)を生成してみましょう
# この数値が格納された配列のことをpytorchではtensorと呼びます

x = torch.empty(5, 3)
print(x)

# ほかにも，0で初期化したものや，0-1で初期化したものなども作成できます

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3)
print(x)

# pythonのリストから直接生成することもできます
x = torch.tensor([3, 2])
print(x)

# pytorch上でディープラーニングをする際には入力データをtensor型にする必要があります
