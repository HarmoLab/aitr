import pandas as pd

# csvの読み込み
df = pd.read_csv('./data/train.csv')
# dfの最初の数行を表示
print(df.head(10))
# データのサイズを確認
print(df.shape)
# 特定の列を所得
print(df['filename'])
# 特定の列のある条件を満たしたデータだけ所得
print(df[df['label'] == 0])
# ラベル0のデータの数を数える
print(df[df['label'] == 0].shape)


# 演習：テストデータに対しても同様にデータ数を確認してください
