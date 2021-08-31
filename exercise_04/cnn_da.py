import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.models as models

#import matplotlib.pyplot as plt


# 各種設定
BATCH_SIZE = 50
MAX_EPOCH = 10
IMAGE_SIZE = 224
# シード値の固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# 学習を途中から再開する
model_read = True


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

# 訓練用のデータ前処理を定義
train_transform = transforms.Compose(
    [
        # 画像のリサイズ
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # ここにData augmentation処理を記述する



        # tensorに変換
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]
)


# 訓練データを取得
train_dataset = MyDataset(
    csv_file="./data/train.csv",
    root_dir='../exercise_03/data/train/',  # 画像を保存したディレクトリ(適宜書き換えて)
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

# 訓練時のtransformを設定
train_dataset.dataset.transform = train_transform

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
    root_dir='../exercise_03/data/train/',
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

# GPU利用設定
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

# 学習途中モデルの読み出し
if model_read:
    if not(torch.cuda.is_available()):
        checkpoint = torch.load("./model/cnn40.pt", map_location=device)
    else:
        checkpoint = torch.load("./model/cnn40.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
# 損失関数
loss_function = nn.CrossEntropyLoss()
# 勾配降下法を行うoptimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if model_read:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start = time.time()
print('processing train data ...')

v_acc = []
t_acc = []
best_v_acc = 0
v_accuracy = []
for epoch in range(MAX_EPOCH):
    model.train()
    train_loss_list = []
    # DataLoaderをfor文で回すと入力と正解ラベルが得られる
    for x, label, img_path in train_loader:
        x = x.to(device)
        label = label.to(device)
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
        x = x.to(device)
        label = label.to(device)

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
    t_acc.append(train_loss_mean)
    v_acc.append(valid_loss_mean)
    v_accuracy.append(valid_accuracy)
    # モデル保存
    if valid_accuracy > best_v_acc:
        best_v_acc = valid_accuracy
        model_dir = "./model/"
        os.makedirs(model_dir, exist_ok=True)
        save_path = model_dir + "cnn.pt"
        torch.save(model.state_dict(), save_path)

if not(model_read):
    # 事前学習用
    model_dir = "./model/"
    os.makedirs(model_dir, exist_ok=True)
    save_path_ckpt = model_dir + "cnn{}.pt".format(MAX_EPOCH)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        }, save_path_ckpt
    )


# モデル読込
model.load_state_dict(torch.load("./model/cnn.pt"))

# 可視化コード
# fig, ax = plt.subplots(1,1)
# ax.plot(t_acc, label="train loss", marker="o")
# ax.plot(v_acc, label="valid loss", marker="*")
# ax.legend()
# plt.savefig("losscurve.png", bbox_inches="tight")

# fig, ax = plt.subplots(1,1)
# ax.plot(v_accuracy, label="valid acc", marker="o")
# ax.legend()
# plt.savefig("acccurve.png", bbox_inches="tight")

# モデルの評価(テストデータを使用)
print('processing test data ...')
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
for x, label, img_path in test_loader:
    x = x.to(device)
    label = label.to(device)
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
