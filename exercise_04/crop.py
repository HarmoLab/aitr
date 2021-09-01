import torchvision
from PIL import Image

# 画像の読み込み
img = Image.open("../exercise_03/data/train/cat.1.jpg")

# 元画像の表示
img.show()

# 画像のクロップ
transform = torchvision.transforms.RandomCrop((224, 224))

# クロップ
img_output = transform(img)

# 画像の表示
img_output.show()

# 画像の保存
img_output.save("output/crop.jpg")
