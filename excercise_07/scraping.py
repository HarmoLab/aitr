import time
import datetime

import requests
from bs4 import BeautifulSoup
from pathlib import Path

# HTML接続・パース
root = "http://127.0.0.1:5000"
res = requests.get(root)
soup = BeautifulSoup(res.content, "html.parser")

# classタグが2Dの画像を取得
img_2d = soup.find("div", class_="2D").find_all("img")


nowdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
p = Path("download_imgs{}/".format(nowdate))
p.mkdir()
for img_tag in img_2d:
    # 画像の取得
    img = requests.get(root + img_tag["src"])

    # 保存
    with open(p / img_tag["src"].split("/")[-1], "wb") as f:
        f.write(img.content)

    # アクセス後5秒間待機
    time.sleep(5)
