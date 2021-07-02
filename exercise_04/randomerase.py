import torchvision
from PIL import Image
from torchvision import transforms

# 画像の読み込み
img = Image.open("../excercise_03/data/train/cat.1.jpg")

# Tensor化
img = torchvision.transforms.ToTensor()(img)

# RandomErasing
transform = torchvision.transforms.RandomErasing(p=1)
img_output = transform(img)

# 表示用にPIL形式に戻す
img_output = transforms.ToPILImage()(img_output)

# 画像の表示
img_output.show()

# 画像の保存
# img_output.save("output/re.jpg")
