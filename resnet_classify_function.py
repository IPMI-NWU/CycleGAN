from __future__ import print_function
# 使得我们能够手动输入命令行参数
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import os
import torch.utils.data as Data
import numpy as np

from cal_dataloader import *
from Resnet_model import *


# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--img_height', type=int, default=100)
parser.add_argument('--img_width', type=int, default=100)
parser.add_argument('--n_residual_blocks', type=int, default=7)
# 跑多少次batch进行一次日志记录
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


torch.cuda.set_device(0)
# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置一个随机数种子
torch.manual_seed(args.seed)
if args.cuda:
    # 为GPU设置一个随机数种子
    torch.cuda.manual_seed(args.seed)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307, ), (0.3081, ))  # 灰度图只有一个通道 所以只有（*，，）*位置上的一个数据
#                                                   # 前边表示该通道标准化的均值为0.1307，后边表示该通道标准化的标准差为0.3081
# ])
#
# train_set = datasets.MNIST(root='./dataset/mnist', train=True, transform=transform, download=True)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# test_set = datasets.MNIST(root='./dataset/mnist', train=False, transform=transform, download=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

input_shape = (args.channels, args.img_height, args.img_width)
# model = Net()
model = Net(input_shape, args.n_residual_blocks)
# 输出torch模型每一层的输出
print('model_summary', summary(model, input_size=(1, 200, 100), batch_size=64, device='cpu'))

# 判断是否调用GPU模式
if args.cuda:
    model.cuda()
# 初始化优化器 model.train()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, Train_dirroot, flag_num):
    """
    定义每个epoch的训练细节
    """
    # Training data loader
    dirroot = Train_dirroot
    print('train_dirroot', dirroot)
    train_datasets = MyDataset_train(dirroot, flag_num, mode='train')
    train_loader = Data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('len(dataloader)', len(train_loader))

    # 设置为training模式
    model.train()
    for batch_idx, (data, target0) in enumerate(train_loader):
        target_n = np.array(target0)
        target = torch.from_numpy(target_n)
        # print('data', data.shape)  # 64*1*28*28
        # print('target.size', target.shape) # torch.Size([64])
        # 如果要调用GPU模式,就把数据转存到GPU
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # 优化器梯度初始化为零
        optimizer.zero_grad()
        output = model(data)
        # output = torch.tensor(output)
        output = torch.squeeze(output)

        # 输出训练的准确率
        train_correct = 0
        pred = output.data.max(1, keepdim=True)[1]  # 获取最大对数概率值的索引
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的个数进行累加
        train_correct = 100. * train_correct / 20

        # 负对数似然函数损失
        # print('output', output)
        # print('output.shape', output.shape)
        # print('target', target)
        # print('target.shape', target.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}  Accuracy:({:.0f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_correct,
            ))


def test(Test_txt):
    # Test data loader
    dirroot = r'../../chest_data'
    # train_path = os.path.join(dirroot, 's_test0.txt')
    train_path = os.path.join(dirroot, Test_txt)
    # pdb.set_trace()  # 程序运行到这里就会暂停。
    test_datasets = MyDataset_test(dirroot, train_path, mode='test')
    test_loader = Data.DataLoader(dataset=test_datasets, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    # 设置为test模式
    model.eval()
    # 初始化测试损失值为0
    test_loss = 0
    # 初始化预测正确的数据个数为0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)

        output = torch.squeeze(output)
        # 把所有loss值进行累加
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # 获取最大对数概率值的索引
        pred = output.data.max(1, keepdim=True)[1]
        # 对预测正确的个数进行累加
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 因为把所有loss值进行累加,所以最后要除以总的数据长度才能得到平均loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))


# 进行每个epoch的训练
for k in range(4):
    Train_dirroot = r'./images_testdef_norevolve/chest'
    Test_txt = 's_test' + str(k) + '.txt'
    print('-----------------第', k, '折-----------------------')
    for epoch in range(1, args.epochs + 1):
        train(epoch, Train_dirroot, k)
        test(Test_txt)
