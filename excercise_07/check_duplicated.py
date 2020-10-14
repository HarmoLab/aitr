import imagehash
from pathlib import Path
from PIL import Image

img_num = 300

p = Path("data/daytona")

imgs = sorted(list(p.glob("*")))
img_num = len(imgs)

img_hash = [imagehash.phash(Image.open(img)) for img in imgs]

for i in range(img_num):
    if i + 1 <= img_num:
        for j in range(i+1, img_num):
            hashdiff = img_hash[i] - img_hash[j]

            if hashdiff < 5:
                print("同一の画像組候補", i, j, hashdiff)
