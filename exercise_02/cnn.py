import os
import random
import time
import warnings
warnings.simplefilter('ignore', UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# 各種設定
BATCH_SIZE = 128
MAX_EPOCH = 10
# シード値の固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# データの前処理を定義
transform = transforms.Compose(
    [
        # tensorに変換
        transforms.ToTensor(),
    ]
)
# 訓練データを取得
train_dataset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform,
)
# 訓練データの一部を検証データとして使用
train_dataset, valid_dataset = torch.utils.data.random_split(
      train_dataset,
      [48000, 12000],
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
test_dataset = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ニューラルネットワークの定義
class CNN(nn.Module):
    def __init__(self):
        '''
        中間層を定義
        '''
        super(CNN, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 8, 3) # 28x28x1 -> 26x26x8
        # 畳み込み層
        self.conv2 = nn.Conv2d(8, 16, 3) # 26x26x8 -> 24x24x16
        # プーリング層
        self.pool = nn.MaxPool2d(2, 2) # 24x24x16 -> 12x12x16
        # 全結合層
        self.fc1 = nn.Linear(12 * 12 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        '''
        ネットワークの（順伝播）の定義
        '''
        # 畳み込み層 + 活性化関数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 最大プーリング
        x = self.pool(x)

        # 全結合層
        x = x.view(-1, 12 * 12 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ニューラルネットワーク
model = CNN()
# 損失関数
loss_function = nn.CrossEntropyLoss()
# 勾配降下法を行うoptimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 学習にかかった時間の計測用
start_time = time.time()

print("\ntraining start !!!\n")
# train_lossは訓練データに対する誤差, valid_lossは検証データに対する誤差
print("epoch\ttrain_loss\tvalid_loss")
for epoch in range(MAX_EPOCH):
    # モデルを訓練する時は model.train() とする
    model.train()
    train_loss_list = []
    # DataLoaderをfor文で回すとバッチサイズ分の入力(x)と正解ラベル(label)が得られる
    for x, label in train_loader:
        # 勾配を0に初期化
        optimizer.zero_grad()
        # モデルの出力
        output = model(x)
        # 出力(output)と正解(label)の訓練誤差を計算
        loss = loss_function(output, label)
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # ミニバッチの訓練誤差をlistに追加
        train_loss_list.append(loss.item())
    # 各ミニバッチでの訓練誤差の平均を取り，本エポックでの訓練誤差とする
    train_loss_mean = np.mean(train_loss_list)

    # 検証データでも同様に誤差を計算
    # モデルを評価する時は model.eval() とする
    model.eval()
    valid_loss_list = []
    for x, label in valid_loader:
        output = model(x)
        loss = loss_function(output, label)
        valid_loss_list.append(loss.item())
    valid_loss_mean = np.mean(valid_loss_list)

    print("{}\t{:.8}\t{:.8}".format(epoch, train_loss_mean, valid_loss_mean))

# モデル保存
model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)
save_path = model_dir + "cnn.pt"
torch.save(model.state_dict(), save_path)
# モデル読込
# model.load_state_dict(torch.load(save_path))

# モデルの評価(テストデータを使用)
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
for x, label in test_loader:
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
