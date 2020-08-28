import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

# シード値の固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    mixupを行う関数

    facebokkの公式実装を利用しています．
    https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    mixupされた画像に対する予測結果のロス計算を行う関数
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


transform = transforms.Compose(
    [
        # 画像のリサイズ
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # tensorに変換
        transforms.ToTensor(),
    ]
)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 画像読み込みおよび前処理
img_1 = Image.open("../excercise_03/data/train/cat.1.jpg")
img_2 = Image.open("../excercise_03/data/train/dog.1.jpg")
img_3 = Image.open("../excercise_03/data/train/cat.2.jpg")

img_1 = transform(img_1)
img_2 = transform(img_2)
img_3 = transform(img_3)

# バッチの作成（trainloaderをfor文で回したときのx, labelに相当）
label = torch.tensor([0, 1, 0])
x = torch.stack((img_1, img_2, img_3))

# mixup
x, label_a, label_b, lam = mixup_data(x, label, 1, use_cuda=False)

# mixupの後でcnnの順伝播処理，ここでは省略して適当な予測結果を設定
pred = torch.tensor([
    [0.2, 0.8],
    [0.7, 0.3],
    [0.1, 0.9]
])

# ロス計算
loss = mixup_criterion(criterion, pred, label_a, label_b, lam)

# 通常はこの後誤差逆伝播等をおこなう．今回はmixupの結果を見たいので省略

# 画像の表示
for i in range(3):
    img_pil = torchvision.transforms.ToPILImage()(x[i, :, :, :])
    img_pil.show()

# ラベルと混合比
print(label_a, label_b, lam)
