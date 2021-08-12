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
from sklearn.metrics import confusion_matrix


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

print(len(test_dataset))

# ニューラルネットワーク
model = models.resnet18()
# print(model)
model.fc = nn.Linear(512, 2)

start = time.time()
# モデル読み込み
model_dir = "./model/"
save_path = model_dir + "cnn1.pt"
model.load_state_dict(torch.load(save_path))

# モデルの評価(テストデータを使用)
print('processing test data ...')
model.eval()
softmax = nn.Softmax()
for n, (x, label, img_path) in enumerate(test_loader):
    output = model(x)
    pred = output.argmax(dim=1)
    probs = softmax(output)

    preds = pred if n == 0 else torch.cat([preds, pred])
    labels = label if n == 0 else torch.cat([labels, label])

    if n == 0:
        for i in range(50):
            print(img_path[i], probs[i], label[i])

cm = confusion_matrix(labels, preds)
print(cm)
