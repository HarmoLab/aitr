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
# print(x + y)

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
print(x.transpose(0, 1))  # 0軸目と1軸めを入れ替えます



# 演習：次のimg_tensorの縦と横を変えてください
# 読み込んだ画像は(3,2268,4032)のtensorに変換されています
from PIL import Image
import torchvision
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)

# ここに処理を記述

img_changed = torchvision.transforms.functional.to_pil_image(img_tensor)
img_changed.save('car_changed.jpg')


# 演習：img_tensorの縦300から1499番目，横800番目xから1200番目の領域を値を0に0に変更してください
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)

# ここに処理を記述

img_changed = torchvision.transforms.functional.to_pil_image(img_tensor)
img_changed.save('car_mask.jpg')

# 演習：２つのtensorをそれぞれ0.5倍した上で足してください
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)
img2 = Image.open("cleaner.jpg")
img_tensor2 = torchvision.transforms.functional.to_tensor(img2)

# ここに処理を記述

img_mixup = torchvision.transforms.functional.to_pil_image(img_output)
img_mixup.save('mixup.jpg')

