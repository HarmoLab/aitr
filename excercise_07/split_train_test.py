import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def split_train_test(watch_kind, fold):

    # データの読み込み
    paths = []
    labels = []
    for idx, kind in enumerate(watch_kind):
        p = Path("./data/{}/".format(kind))
        imgs = list(p.glob("*"))

        for imgpath in imgs:
            paths.append(imgpath)
            labels.append(idx)

    paths = np.array(paths)
    labels = np.array(labels)

    # 分割処理
    kf = KFold(n_splits=fold, random_state=0, shuffle=True)

    for idx, (train_index, test_index) in enumerate(kf.split(paths)):

        # 訓練データ
        with open("./data/train_kfold_{:02}.csv".format(idx), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for path, label in zip(paths[train_index], labels[train_index]):
                writer.writerow([path, label])

        # テストデータ
        with open("./data/test_kfold_{:02}.csv".format(idx), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for path, label in zip(paths[test_index], labels[test_index]):
                writer.writerow([path, label])


def main():
    watch_kind = ["daytona", "gmtmaster"]
    fold = 5

    split_train_test(watch_kind, fold)


if __name__ == "__main__":
    main()
