from sklearn.model_selection import KFold
import numpy as np

x = np.array(range(20, 40))  # 訓練データ
y = np.array(range(30, 50))  # テストデータ

kf = KFold(n_splits=2, random_state=None, shuffle=False)

for train_index, test_index in kf.split(x):
    print("訓練インデックス", train_index)
    print("訓練データ", x[train_index])
    print("テストインデックス", test_index, "\n")

    print("テストデータ", x[test_index])

