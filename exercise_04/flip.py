import torchvision
from PIL import Image


# 画像の読み込み
img = Image.open("../excercise_03/data/train/cat.1.jpg")

# 元画像の表示
img.show()

# 画像の左右反転
transform = torchvision.transforms.RandomHorizontalFlip(p=1)
img_output = transform(img)

# 画像の表示
img_output.show()

# 画像の保存
# img_output.save("output/vflip.jpg")
