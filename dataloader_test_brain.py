from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import os
import pdb
import cv2

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def to_grey(image):
    grey_image = Image.new("grey", image.size)
    grey_image.paste(image)
    return grey_image

def default_loader(path, IorM = 'rgb'):
    if IorM == 'rgb':
        return Image.open(path)
    else:
        return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, dirroot, txt, loader=default_loader, mode='train', wName=False, transforms_=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()  # chars -- 指定删除的字符（默认为空格）
            # print('line', line)  # HGG\Brats18_TCIA08_406_1 145.png 1
            words = line.split(' ')
            # print('words', words)  # words ['HGG\\Brats18_TCIA08_406_1', '145.png', '1']

            # window10 上运行
            imgs.append((words[0], words[1], words[-1]))

            '''
            # ubuntu 上运行需要修改路径的表示形式
            num = words[0].split('\\')
            f = num[0]+'/'+num[1]+'/'+num[2]
            imgs.append((f, words[1]))
            '''

        self.imgs = imgs
        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=2)
        self.degrees = random.uniform(0, 10)
        self.RandomAffine = transforms.RandomAffine(degrees=self.degrees)
        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.resize = transforms.Resize(320)
        self.loader = loader
        self.mode = mode
        self.wName = wName
        self.dirroot = dirroot
        # self.dirroot = '../../NII2PNG'
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):

        f, pic_name, label = self.imgs[index]
        words = f.split('\\')
        subpath1 = words[-1] + '_t1.gz'
        subpath2 = words[-1] + '_t2.gz'
        # subpath1 = words[-1] + '_t1ce.gz'
        # subpath2 = words[-1] + '_flair.gz'


        fn1 = os.path.join(self.dirroot, words[0], words[1], subpath1, pic_name)
        fn2 = os.path.join(self.dirroot, words[0], words[1], subpath2, pic_name)

        image_A = Image.open(fn1)
        # print('image_A', image_A)  # image_A <PIL.PngImagePlugin.PngImageFile image mode=L size=155x240 at 0x1F518F49BE0>
        image_B = Image.open(fn2)
        # print('image_B', image_B)


        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        # print('imageA', image_A)  # imageA <PIL.Image.Image image mode=RGB size=155x240 at 0x271FC799B00>


        # if self.mode == 'train':
        #     if random.random() >=0.5: #水平翻转
        #         image_A = self.HorizontalFlip(image_A)
        #         image_B = self.HorizontalFlip(image_B)
        #     if random.random() >=0.6: #仿射变换
        #         image_A = self.RandomAffine(image_A)
        #         image_B = self.RandomAffine(image_B)
        #     if random.random() >=0.6: #对比度变换
        #         image_A = self.ColorJitter(image_A)
        #         image_B = self.ColorJitter(image_B)
#
        # image_A = self.resize(image_A)
        # image_A = self.toTensor(image_A)
        # image_B = self.resize(image_B)
        # image_B = self.toTensor(image_B)

        item_a = self.transform(image_A)
        item_b = self.transform(image_B)


        # img = self.resize(img)
        # img = self.toTensor(img)
        # img0 = self.resize(img0)
        # img0 = self.toTensor(img0)

        # if self.wName == True:
        #     return img, img0, patientName, fn
        # else:
        #     return img, img0

        return {"A": item_a, "B": item_b, "label": int(label)}

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dirroot = r'../../NII2PNG/'
    train_path = os.path.join(dirroot, 'train_Brain.txt')

    # pdb.set_trace()  # 程序运行到这里就会暂停。
    train_datasets = MyDataset(dirroot, train_path, mode='train', wName=True, transforms_=None)
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=1, shuffle=True, num_workers=0)
    imgs = next(iter(trainloader))
    print(imgs)

    for step, (imgs, true_mask, patientName, fn) in enumerate(trainloader):
        print(imgs)
        imgs = imgs.float()
        true_mask = true_mask.float()
        print(imgs)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(imgs[0][0])
        plt.subplot(1,2,2)
        plt.imshow(true_mask[0][0], cmap='Greys_r')
        plt.show()

