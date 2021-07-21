import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from PIL import Image
import torchvision

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

# TODO CNNモデル読込

# TODO 画像読込

# TODO 画像をtensorに変換

# TODO tensorにバッチの次元を作る

# TODO 推論(CNNモデルに画像tensorを入力)

# TODO ソフトマックス関数で確率として扱える値に変換

# TODO 確率が最も高い 数字 を分類結果にする

# TODO 各数字の確率と分類結果を出力

