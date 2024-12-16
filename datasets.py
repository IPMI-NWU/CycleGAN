import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

'''
Image.new()方法，顾名思义，是用来new一个新的图像，具体参数如下：

Image.new(mode, size, color=0)

mode:模式，通常用"RGB"这种模式，如果需要采用其他格式，可以参考博文：PIL的mode参数

size：生成的图像大小

color：生成图像的颜色，默认为0，即黑色。



python中PIL库中的paste函数的作用为将一张图片覆盖到另一张图片的指定位置去。函数的声明如下：

def paste(self, im, box=None, mask=None):
该函数是图像的一个方法，调用方式为image.paste(...)。

定义中im是要黏贴到image上面去的图片，box是要黏贴到的区域。

'''


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        # print('itemA', item_A)
        # print('itemB', item_B)
        # print('type(itemA)', type(item_A))
        # print('type(itemB)', type(item_B))
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
