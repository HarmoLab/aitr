import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.models as models


# 各種設定
BATCH_SIZE = 50
MAX_EPOCH = 5
IMAGE_SIZE = 224
# シード値の固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# 自作のデータセットの処理
class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, loader=default_loader):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir  # 画像が入っているディレクトリのパス
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    # dataloaderで読み込むときの処理
    def __getitem__(self, idx):
        # filename取得
        img_name = self.df.iloc[idx, 0]
        # 画像のパス設定
        img_path = os.path.join(self.root_dir, img_name)
        # 画像読み込み
        image = self.loader(img_path)
        # user_id = self.df.iloc[idx, 2]

        label = self.df.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, img_path


# データの前処理を定義
transform = transforms.Compose(
    [
        # 画像のリサイズ
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # tensorに変換
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]
)
# 訓練データを取得
train_dataset = MyDataset(
    csv_file="./data/train.csv",
    root_dir='./data/train/',  # 画像を保存したディレクトリ(適宜書き換えて)
    transform=transform,
)
# 訓練データの一部を検証データとして使用
num_train = len(train_dataset)
print(num_train)
train_dataset, valid_dataset = torch.utils.data.random_split(
    train_dataset,
    [int(num_train * 0.8), int(num_train * 0.2)],
)
# DataLoaderを作成
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
# テストデータも同様
test_dataset = MyDataset(
    "./data/test.csv",
    root_dir='./data/train/',
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ニューラルネットワーク
model = models.resnet18()
# print(model)
model.fc = nn.Linear(512, 2)
# 損失関数
loss_function = nn.CrossEntropyLoss()
# 勾配降下法を行うoptimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

start = time.time()
print('processing train data ...')
for epoch in range(MAX_EPOCH):
    model.train()
    train_loss_list = []
    # DataLoaderをfor文で回すと入力と正解ラベルが得られる
    for x, label, img_path in train_loader:
        # 勾配を0に初期化
        optimizer.zero_grad()
        # 順伝播
        output = model(x)
        # 誤差の計算
        loss = loss_function(output, label)
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # ミニバッチの訓練誤差をlistに追加
        train_loss_list.append(loss.item())
        # print(loss.item(), time.time()-start)
    # 各ミニバッチでの訓練誤差の平均を取り，本エポックでの訓練誤差とする
    train_loss_mean = np.mean(train_loss_list)

    # 検証データでも同様に誤差を計算
    # モデルを評価する時は model.eval() とする
    model.eval()
    valid_loss_list = []
    valid_correct, valid_total = 0, 0
    for x, label, img_path in valid_loader:
        output = model(x)
        loss = loss_function(output, label)
        valid_loss_list.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)
        # 正解ラベルと比較，一致している数を加算
        valid_correct += pred.eq(label.view_as(pred)).sum().item()
        # 正解率(accuracy)を出すためにテストデータの数も加算
        valid_total += label.size()[0]

    valid_loss_mean = np.mean(valid_loss_list)

    valid_accuracy = valid_correct / valid_total

    print(epoch, train_loss_mean, valid_loss_mean, "Valid accuracy :", valid_accuracy,
          'processed time:', time.time()-start)

# モデル保存
model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)
save_path = model_dir + "cnn.pt"
torch.save(model.state_dict(), save_path)
# モデル読込
# model.load_state_dict(torch.load(save_path))

# モデルの評価(テストデータを使用)
print('processing test data ...')
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
for x, label, img_path in test_loader:
    output = model(x)
    loss = loss_function(output, label)
    # 出力値が最大のインデックスを取得
    pred = output.argmax(dim=1, keepdim=True)
    # 正解ラベルと比較，一致している数を加算
    test_correct += pred.eq(label.view_as(pred)).sum().item()
    # 正解率(accuracy)を出すためにテストデータの数も加算
    test_total += label.size()[0]

test_accuracy = test_correct / test_total
print("Test accuracy :", test_accuracy)
