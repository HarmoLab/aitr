import torchvision
from PIL import Image


# 画像の読み込み
img = Image.open("../exercise_03/data/train/cat.1.jpg")

# 元画像の表示
img.show()

# 画像の回転
transform = torchvision.transforms.RandomRotation((-30, 30))
img_output = transform(img)

# 画像の表示
img_output.show()

# 画像の保存
# img_output.save("output/rotate.png")
