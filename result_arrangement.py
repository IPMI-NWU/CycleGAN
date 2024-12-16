
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import os


def Get_Max(list):
    return max(list)

#最小数
def Get_Min(list):
    return min(list)

#极差
def Get_Range(list):
    return max(list) - min(list)

#中位数
def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
       median = (data[size//2]+data[size//2-1])/2
   if size % 2 == 1: # 判断列表长度为奇数
       median = data[(size-1)//2]
   return median

#众数(返回多个众数的平均值)
def Get_Most(list):
    most=[]
    item_num = dict((item, list.count(item)) for item in list)
    for k,v in item_num.items():
        if v == max(item_num.values()):
           most.append(k)
    return sum(most)/len(most)

#获取平均数
def Get_Average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)

#获取方差
def Get_Variance(list):
	sum = 0
	average = Get_Average(list)
	for item in list:
		sum += (item - average)**2
	return sum/len(list)

#获取n阶原点距
def Get_NMoment(list,n):
    sum=0
    for item in list:
        sum += item**n
    return sum/len(list)


if __name__ == '__main__':

    dirroot = r'D:\PythonProgram\PyTorch-GAN\implementations\cyclegan\results'
    train_path = os.path.join(dirroot, 'log_chest_256_changeloss_changesuper.txt')

    fh = open(train_path, 'r')
    imgs = []
    imageA_name = []
    adv = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        name = line.split('SSIM:')
        # print('name', name)  # # ['0', '0chenxueli2765868', '94 0']
        if len(name) >= 2:
            # print('name[1]', name[1])
            name_name = name[1].split(',')
            # print('name_name[0]', name_name[0])
            adv.append(float(name_name[0]))
        # number = name[2].split(' ')
        # imageA_name.append((name[1], '_', number[0]))
    print('最大值', Get_Max(adv))
    print('最小值', Get_Min(adv))
    print('中位数', get_median(adv))
    print('极差', Get_Range(adv))
    print('众数', Get_Most(adv))
    print('平均数', Get_Average(adv))
    print('方差', Get_Variance(adv))


    time = 0
    for f, label in imgs:
        slice = os.path.split(f)[1]
        fn = os.path.join(dirroot, f)  # D:\PythonProgram\PyTorch-GAN\chest_data\0\0WANGWENXINxianliu\79
        image_A = Image.open(os.path.join(fn, 'cut_DWI.png'))
        image_B = Image.open(os.path.join(fn, 'cut_shrink.png'))

        d_path_A = '../data/train3_A'
        d_path_B = '../data/train3_B'

        new_A_name = imageA_name[time][0] + imageA_name[time][1] + imageA_name[time][2] + '_' + 'cut_DWI.png'
        new_B_name = imageA_name[time][0] + imageA_name[time][1] + imageA_name[time][2] + '_' + 'cut_shrink.png'

        time = time + 1
        dst_A = os.path.join(d_path_A, new_A_name)
        dst_B = os.path.join(d_path_B, new_B_name)

        image_A.save(dst_A)
        image_B.save(dst_B)
    print('ok!')