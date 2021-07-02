import imagehash
import os
from pathlib import Path
from PIL import Image

# DELETE = True とすると同じクラス内での同一画像候補を自動で削除
DELETE = True

# 画像を格納しているディレクトリのパスを記入
path_list = [
    "豚肉+生肉/pork_cleaned/",
    "牛肉+生肉/beef_cleaned/",
]

all_imgs = []
all_img_hash = []
for path in path_list:
    p = Path(path)
    imgs = sorted(list(p.glob("*")))
    img_num = len(imgs)
    img_hash = [imagehash.phash(Image.open(img)) for img in imgs]

    print()
    print(path)
    print()
    for i in range(img_num):
        if i + 1 <= img_num:
            for j in range(i+1, img_num):
                hashdiff = img_hash[i] - img_hash[j]

                if hashdiff < 5:
                    print("同じクラス内での同一の画像組候補")
                    print(imgs[i])
                    print(imgs[j])
                    print()
                    if DELETE:
                        os.remove(imgs[j])

    # 削除したから再度取り直し
    p = Path(path)
    imgs = sorted(list(p.glob("*")))
    img_num = len(imgs)
    img_hash = [imagehash.phash(Image.open(img)) for img in imgs]
    # 全クラスまとめ
    all_imgs.extend(imgs)
    all_img_hash.extend(img_hash)

img_num = len(all_imgs)
for i in range(img_num):
    if i + 1 <= img_num:
        for j in range(i+1, img_num):
            hashdiff = all_img_hash[i] - all_img_hash[j]

            if hashdiff < 5:
                print("異なるクラス間での同一の画像組候補")
                print(all_imgs[i])
                print(all_imgs[j])
                print()
