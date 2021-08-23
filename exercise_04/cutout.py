import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np

# Cutoutのコードは以下URLのものを利用
# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask_value = img.mean()

        for n in range(self.n_holes):
            top = np.random.randint(0 - self.length // 2, h)
            left = np.random.randint(0 - self.length // 2, w)
            bottom = top + self.length
            right = left + self.length

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left

            img[:, top:bottom, left:right].fill_(mask_value)

        return img


# 画像の読み込み
img = Image.open("../exercise_03/data/train/cat.1.jpg")

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    Cutout(1, 50),
    transforms.ToPILImage()
])

# Tensor化
img_output = transform(img)

# 画像の表示
img_output.show()

# 画像の保存
# img_output.save("output/re.jpg")
