"""
学習・評価に使用する画像のみ抽出してくるスクリプト
"""
import shutil
import os

from_data_dir = "../excercise_03/data/train/"
to_data_dir = "./data/train/"
os.makedirs(to_data_dir, exist_ok=True)

start_no = 1000
num_cat = 3000
num_dog = 300
cat_list = ['cat.{}.jpg'.format(start_no + x) for x in range(num_cat)]
dog_list = ['dog.{}.jpg'.format(start_no + x) for x in range(num_dog)]


for img_path in cat_list:
    # 画像をコピー
    shutil.copy(from_data_dir + img_path, to_data_dir + img_path)
for img_path in dog_list:
    # 画像をコピー
    shutil.copy(from_data_dir + img_path, to_data_dir + img_path)

# to_data_dir = "./data/test/"
# os.makedirs(to_data_dir, exist_ok=True)

# start_no = 10000
# num_test = 500
# cat_list = ['cat.{}.jpg'.format(start_no + x) for x in range(num_test)]
# dog_list = ['dog.{}.jpg'.format(start_no + x) for x in range(num_test)]
# for img_path in cat_list:
#     # 画像をコピー
#     shutil.copy(from_data_dir + img_path, to_data_dir + img_path)
# for img_path in dog_list:
#     # 画像をコピー
#     shutil.copy(from_data_dir + img_path, to_data_dir + img_path)
