from torch.utils.data import Dataset
import torch.utils.data as Data
from PIL import Image
import random
import os
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

def default_loader(path, IorM = 'rgb'):
    '''
    # RGB三通道读取图像
    if IorM == 'rgb':
        return Image.open(path)
    else:
        return Image.open(path).convert('L')
    '''
    return Image.open(path).convert('L')  # 读取形式为灰度图


# 直接展示tensor形式存储的图片
def imshow(tensor, title=None):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
      plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class MyDataset_train(Dataset):
    def __init__(self, dirroot, flag_num, loader=default_loader, mode='train'):
        imgs = []
        path = dirroot
        path_list = os.listdir(path)
        for line in path_list:
            words = line.split('_')
            if words[0] == 'data' and words[1] == str(flag_num):
                pic_name = line
                label = words[-1].split('.')[0]
                imgs.append((pic_name, label))
        # print('train_imgs', imgs)
        self.imgs = imgs
        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=2)
        self.degrees = random.uniform(0, 10)
        self.RandomAffine = transforms.RandomAffine(degrees=self.degrees)
        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.resize = transforms.Resize((200, 100))  # resize的参数顺序是h, w
        self.mode = mode
        # self.dirroot = dirroot
        self.dirroot = dirroot
        self.loader = loader
        # self.Norm = transforms.Normalize((0.1307, ), (0.3081, ))
        self.Norm = transforms.Normalize((0.1307, ), (0.3081, ))

    def __getitem__(self, index):

        pic_name, label = self.imgs[index]

        image = self.loader(os.path.join(self.dirroot, pic_name))

        if self.mode == 'train':
            if random.random() >=0.5: #水平翻转
                image = self.HorizontalFlip(image)
            if random.random() >=0.6: #仿射变换
                image = self.RandomAffine(image)
            if random.random() >=0.6: #对比度变换
                image = self.ColorJitter(image)

        image = self.resize(image)
        image = self.toTensor(image)
        image = self.Norm(image)

        return image, int(label)

    def __len__(self):
        return len(self.imgs)


class MyDataset_test(Dataset):
    def __init__(self, dirroot, txt, loader=default_loader, mode='test'):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            '''
            # window10 上运行
            imgs.append((words[0], words[1]))
            '''

            # ubuntu 上运行需要修改路径的表示形式
            num = words[0].split('\\')
            f = num[0] + '/' + num[1] + '/' + num[2]
            imgs.append((f, words[1]))
        # print('imgs', imgs)
        self.imgs = imgs
        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=2)
        self.degrees = random.uniform(0, 10)
        self.RandomAffine = transforms.RandomAffine(degrees=self.degrees)
        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.resize = transforms.Resize((100, 100))
        self.mode = mode
        self.dirroot = dirroot
        self.loader = loader
        # self.Norm = transforms.Normalize((0.1307, ), (0.3081, ))
        self.Norm = transforms.Normalize((0.1307, ), (0.3081, ))

    def __getitem__(self, index):

        f, label = self.imgs[index]
        slice = os.path.split(f)[1]
        fn = os.path.join(self.dirroot, f)
        prefix_name = os.path.split(fn)[1]
        image_A = self.loader(os.path.join(fn, 'cut_DWI.png'))
        image_B = self.loader(os.path.join(fn, 'cut_shrink.png'))
        # image_merge = Image.merge("RGB", (image_A, image_B))
        # image_merge = image_A + image_B

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

        image_A = self.resize(image_A)
        image_B = self.resize(image_B)

        image_merge = Image.blend(image_A, image_B, 0.5)

        image_merge = self.resize(image_merge)
        image_A = self.toTensor(image_A)
        image_B = self.toTensor(image_B)
        image_merge = self.toTensor(image_merge)
        image_concat = torch.cat([image_B, image_A], 1)  # 图像的高度变为原来的二倍 100*200
        image_A = self.Norm(image_A)
        image_B = self.Norm(image_B)
        image_merge = self.Norm(image_merge)
        image_concat = self.Norm(image_concat)

        # print('imgA5', image_A)
        # print('imgA.size', image_A.shape)
        # print('label', label)

        return image_concat, int(label)

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    dirroot_train = r'./images_testdef/chest'

    train_datasets = MyDataset_train(dirroot_train, flag_num=0, mode='train')
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=1, shuffle=False, num_workers=0)
    # for batch_idx, (data, target) in enumerate(trainloader):
    #     print(data.shape)  # 1*3*512*256/1*1*512*256
    #     print(target)
    #     imshow(data, 'pic')
    print('len(traindata_loader)', len(trainloader))

    dirroot_test = r'D:\PythonProgram\PyTorch-GAN\chest_data'
    train_path = os.path.join(dirroot_test, 's_test0.txt')
    # pdb.set_trace()  # 程序运行到这里就会暂停。
    test_datasets = MyDataset_test(dirroot_test, train_path, mode='test')
    testloader = Data.DataLoader(dataset=test_datasets, batch_size=1, shuffle=True, num_workers=0)
    # for batch_idx, (data, target) in enumerate(trainloader):
    #     print(data)
    #     print(target)
    #     imshow(data, 'pic')
    print('len(testdata_loader)', len(testloader))

