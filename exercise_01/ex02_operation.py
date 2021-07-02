# 1.次のimg_tensorの縦と横を変えてください
# 読み込んだ画像は(3,2268,4032)のtensorに変換されています
from PIL import Image
import torchvision
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)

# ここに処理を記述．処理後のtensorはimg_tensorに代入する


img_changed = torchvision.transforms.functional.to_pil_image(img_tensor)
img_changed.save('car_changed.jpg')


# 2.img_tensorの縦301から1500，横801番目から1200の1200*400の
# 長方形領域を値を0に変更して保存してください
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)

# ここに処理を記述


img_changed = torchvision.transforms.functional.to_pil_image(img_tensor)
img_changed.save('car_mask.jpg')

# 3.２つのtensorをそれぞれ0.5倍した上で足してください
img = Image.open("car.jpg")
img_tensor = torchvision.transforms.functional.to_tensor(img)
img2 = Image.open("cleaner.jpg")
img_tensor2 = torchvision.transforms.functional.to_tensor(img2)

# ここに処理を記述．足し合わせたtensorはimg_outputに代入


img_mixup = torchvision.transforms.functional.to_pil_image(img_output)
img_mixup.save('mixup.jpg')
