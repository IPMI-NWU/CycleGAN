import os
import cv2

# 定位图像的裁减位置
def locate(path):
    prefix_name = os.path.split(path)[1]
    image = cv2.imread(path + '\\' + 'mask_' + prefix_name + '.png')  # 512*512
    img_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row = len(img_gry)
    col = len(img_gry[0])
    x_start = 0
    y_start = 0
    x_end = 0
    y_end = 0
    break2 = False
    for i in range(row):
        if break2 == False:
            for j in range(col):
                if img_gry[i][j] == 255:
                    if (i - 100) >= 0:
                        x_start = i - 100
                    else:
                        x_start = 0
                    x_end = i + 100
                    break2 = True
                    break
        else:
            break

    break2 = False
    for i in range(col):
        if break2 == False:
            for j in range(row):
                if img_gry[j][i] == 255:
                    if (i - 100) >= 0:
                        y_start = i - 100
                    else:
                        y_start = 0
                    y_end = i + 100
                    break2 = True
                    break
        else:
            break

    return x_start, x_end, y_start, y_end


# 获取某路径下文件名包含特定字符串的文件路径
def file_name(file_dir,str):
    L = []
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            filename = file
            # filename = os.path.splitext(file)[0]
            # 如果在filename中包含str,则将该文件的路径加入到L列表中
            if str in filename:
                L.append(os.path.join(root,file))
    return L


# 将单通道图像合成3通道图像
def cropImage(path):
    prefix_name = os.path.split(path)[1]

    # CPU
    # image = cv2.imread(path + '\\' + prefix_name + '_1.png')
    # image1 = cv2.imread(path + '\\' + 'DWI_' + prefix_name + '.png')

    # GPU
    image = cv2.imread(path + '/' + prefix_name + '_1.png')
    image1 = cv2.imread(path + '/' + 'DWI_' + prefix_name + '.png')


    sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
    sz1 = sp[0]  # 图像的高度（行 范围）
    sz2 = sp[1]  # 图像的宽度（列 范围）
    # sz3 = sp[2]                #像素值由【RGB】三原色组成

    a, b, c, d = locate(path)
    a1 = int(a / 2)
    b1 = int(b / 2)
    c1 = int(c / 2)
    d1 = int(d / 2)

    # 你想对文件的操作
    # a = int(sz1 / 2 - 64)  # x start
    # b = int(sz1 / 2 + 64)  # x end
    # c = int(sz2 / 2 - 64)  # y start
    # d = int(sz2 / 2 + 64)  # y end
    cropImg = image[a:b, c:d]  # 裁剪图像
    cropImg1 = image1[a1:b1, c1:d1]  # 裁剪图像

    size = (int(b1 - a1), int(d1 - c1))
    shrink = cv2.resize(cropImg, size, interpolation=cv2.INTER_AREA)

    # CPU
    # mergePath = path + '\\' + 'cut_1.png'
    # # mergePath1 = path + '\\' + 'cut_DWI.png'
    # mergePath2 = path + '\\' + 'cut_shrink_1.png'

    # GPU
    mergePath = path + '/' + 'cut_1.png'
    # mergePath1 = path + '\\' + 'cut_DWI.png'
    mergePath2 = path + '/' + 'cut_shrink_1.png'

    # mergePath_gray = path + '\\' + 'cut_0_gray.png'
    # mergePath1_gray = path + '\\' + 'cut_DWI_gray.png'
    # mergePath2_gray = path + '\\' + 'cut_shrink_gray.png'

    # print(mergePath)
    # cv2.imwrite(filepath,img,flag,[int(cv2.IMWRITE_PNG_COMPRESSION),9]),第三个参数默认为3
    # 判断文件是否存在，不存在时保存
    # if not os.path.exists(mergePath):

    # 将处理的图片转化为灰度图保存
    # cropImg_gray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    # cropImg1_gray = cv2.cvtColor(cropImg1, cv2.COLOR_BGR2GRAY)
    # shrink_gray = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(mergePath, cropImg)
    # cv2.imwrite(mergePath1, cropImg1)
    cv2.imwrite(mergePath2, shrink)

    # cv2.imwrite(mergePath_gray, cropImg_gray)
    # cv2.imwrite(mergePath1_gray, cropImg1_gray)
    # cv2.imwrite(mergePath2_gray, shrink_gray)

if __name__ == '__main__':

    # file_dir = r'D:\PythonProgram\PyTorch-GAN\chest_data\1'
    file_dir = r'../chest_data/0'

    str = '_1.'
    file_list = file_name(file_dir, str)

    # 裁减图像
    for index, file in enumerate(file_list):
        filepath = os.path.dirname(file)
        cropImage(filepath)