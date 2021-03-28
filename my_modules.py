import os
import torch
import torchvision
from torchvision.utils import save_image
import math
import inspect
import re
import random

import matplotlib.pyplot as plt
import numpy as np



# #########################################
# A library for self-practicing with python.
# Build date: 03/15/2021
# #########################################


# ######### Index ##########
# mkdir (make tree directories, and return the final root.)
# save_images (保存任何形式的图片数据到指定文件夹下)
# check_images_from_train_loader_32_to_32 (保存train_loader到指定文件夹下以便检查) (32 -> 32)
# check_images_from_train_loader_32_to_1 (保存train_loader到指定文件夹下以便检查) (32 -> 1)
# show_dataloader_shape (打印dataloader的shape)
# return_free_GPU (返回设备中最后一块GPU的device)
# draw_easy_plot (Draw an easy plot to test the connect between local and remote connection for image communication.)
# check_cuda_env (运行一个最简单的GPU运算，检查CUDA的环境是否已经坏掉了)
# nms (底层实现NMS算法，返回scores经过nms算法满足条件的scores数列的索引值 )
# seed_everything (设置全局种子)
# show_loss (在训练的过程中自动打印loss信息)
#
#
#
#
#

# ####### Build-in ########
# print_info (打印函数的处理状态， begin or done)
# varname (返回变量的名称)


def mkdir(root='', *dir_name_list):
    """
    # make tree directories, and return the final root.
    :param root: 要创建文件夹的起始目录, 一般为 './'
    :param dir_name_list: 链式表示要生成的所有文件夹名称，不用顾及中间文件夹是否存在
    :return: 返回最终的子文件夹名称
    """
    if len(dir_name_list) == 0:
        raise ValueError('Please input the directory names to be built.')

    for dir_name in dir_name_list:
        root = os.path.join(root, dir_name)
        if os.path.exists(root) is not True:
            print('Directory "{}" not exists. Building "{}"...'.format(root, dir_name))
            try:
                os.mkdir(root)
                print_info()
            except:
                # 以后这种方式尽量少用，尽可能使用python的原生报错以得到最准确的错误原因，不要自己手动更改
                raise ValueError('The new directory build not successfull.')
            
    return root




def save_images(images, dir_name, epoch=None, step=None, token=None, pytorch=True):
    """
    # 保存包含channel维度的任何形式的图片数据，并放在指定的文件夹下
    # 图片的shape可以是[32, 64, 64, 3], [1, 64, 64, 3], [64, 64, 3]
    # 注意如果图片是单张图片（不是一个batch的图片）的话，建议传入token参数来区分
    :param images:
    :param dir_name:
    :param epoch:
    :param step:
    :param token:
    :return: 成功与否
    """
    # images = gen_image.to('cpu').clone().detach()
    # images = images.numpy().transpose(0, 2, 3, 1) # => [b,  H, W, c]
    batch_shape = images.dim()
    if batch_shape == 4:
        batch_size = images.size(0)
        if batch_size == 1: # e.g. [1, 3, 64, 64]
            images = images.squeeze(0)
            name = os.path.join(dir_name, 'image{}{}{}.png'.format("_epoch_{}".format(epoch+1) if epoch != None else "", "_step_{}".format(step) if step != None else "", "_{}".format(token) if token != None else ""))
            print_info('begin', 'save image "{}"'.format(name))
            save_image(images, name)
            print_info()
            return True
        if batch_size != 1: # e.g. [32, 3, 64, 64]
            # 默认显示4行
            rows = 4
            # colums 限幅10
            columns = min(math.floor(batch_size / rows), 10)
            imgs = images.to('cpu').clone().detach().numpy()
            # 暂且认为用matplotlib储存32to1的图片需要的格式是[b, H, W, c]
            if pytorch:
                imgs = imgs.transpose(0, 2, 3, 1)
            fig = plt.figure(figsize=(25, 16))
            name = os.path.join(dir_name, 'image_to_one{}{}{}.png'.format("_epoch_{}".format(epoch+1) if epoch != None else "",
                                                                   "_step_{}".format(step) if step != None else "",
                                                                   "_{}".format(token) if token != None else ""))
            for ii, img in enumerate(imgs):
                ax = fig.add_subplot(rows, columns, ii + 1, xticks=[], yticks=[])
                plt.imshow(img)
                plt.savefig(name)
                if ii == rows * columns - 1: break
            print('saved {}'.format(name))
            return True

    if batch_shape == 3: # e.g. [3, 64, 64]
        name = os.path.join(dir_name, 'image{}{}{}.png'.format("_epoch_{}".format(epoch+1) if epoch != None else "", "_step_{}".format(step) if step != None else "", "_{}".format(token) if token != None else ""))
        print_info('begin', 'save image "{}"'.format(name))
        save_image(images, name)
        print_info()
        return True

    raise ValueError('The input tensor shape not handlable.')




# check images from train_loader
def check_images_from_train_loader_to_all(train_loader, img_dir='./', terminate=True, pytorch=True):
    """
    save the fitst batch of train_loader to the img_dir, don't merge them to one single image
    :param train_loader: train_loader
    :param img_dir: img_dir = mkdir()
    :param pytorch: pytorch的保存图片的默认格式是[b, c, H, W], 而tf是[b, H, W, c]
    :param terminate: 是否在保存图片之后终止程序
    """
    imgs = next(iter(train_loader))
    if isinstance(imgs, list):
        imgs = imgs[0]
        print('The dataloader is a "tuple" instance, displaying imgs[0] by default')
    print('batch shape: {}'.format(imgs.shape))
    batch_size = imgs.size(0)
    # if pytorch:
    #     imgs = imgs.permute((0,2,3,1))
    for ii in range(batch_size):
        print_info('begin', 'saving sample_{}.png'.format(ii))
        save_image(imgs[ii, ...], os.path.join(img_dir, f'sample_{ii}.png'))
        print_info()
    # terminate the program
    exit() if terminate else None # 不行就换这个：terminate and exit()




# # check images from train_loader (32 to 1)
def check_images_from_train_loader_to_1(train_loader, img_dir='./', rows=4, terminate=True, pytorch=True):
    """
    save the fitst batch of train_loader to the img_dir, don't merge them to one single image
    :param train_loader: train_loader
    :param img_dir: img_dir = mkdir()
    :param pytorch: pytorch的保存图片的默认格式是[b, c, H, W], 而tf是[b, H, W, c]
    :param terminate: 是否在保存图片之后终止程序
    """
    imgs =next(iter(train_loader))
    if isinstance(imgs, list):
        imgs = imgs[0]
        print('The dataloader is a "tuple" instance, displaying imgs[0] by default')
    print('batch shape: {}'.format(imgs.shape))
    batch_size = imgs.size(0)
    # colums 限幅10
    columns = min(math.floor(batch_size / rows), 10)
    imgs = imgs.to('cpu').clone().detach().numpy()
    if pytorch:
        imgs = imgs.transpose(0, 2, 3, 1)
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(imgs):
        ax = fig.add_subplot(rows, columns, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
        plt.savefig(os.path.join(img_dir, '32_to_1.png'))
        if ii == rows*columns-1: break
    print_info('begin', 'saving 32_to_1.png')
    print_info()
    # terminate the program
    exit() if terminate else None #
    # 不行就换这个：terminate and exit()




def return_free_GPU():
    """
    return the last GPU device. Because it is always free for training.
    :return: the last GPU device.
    """
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device = torch.device('cuda:{}'.format(gpu_num-1))
        print('Using GPU:[{}]/[{}] for training...'.format(gpu_num-1,gpu_num-1))
        return device
        
    raise ValueError('GPU not available for training. Check CUDA env with function "check_cuda_env"')
    




def draw_easy_plot(terminate=True):
    """
    Draw an easy plot to test the connect between local and remote connection for image communication.
    :return:
    """
    print_info('begin', 'draw a eazy plot to check the connection with remote server.')
    x = np.linspace(-1, 1, 50)
    y = 2 * x + 1
    plt.plot(x, y)
    plt.show()
    print_info()
    # terminate the program
    exit() if terminate else None # 不行就换这个：terminate and exit()




def print_info(status='done', begin_item=None):
    """
    (Build in)
     打印函数的处理状态， begin or done
    :param status:
    :param begin_item:
    :return: 'begin' or 'done'
    """
    if status == 'done':
        print('Done.')
        return 'done'
    if status == 'begin':
        chars = 'Begin to {}...'.format(begin_item)
        print(chars)
        return 'begin'
    raise ValueError("'status' parameter not given('begin' or 'done').")




def check_cuda_env(terminate=True):
    """
    运行一个最简单的GPU运算，检查CUDA环境是否已经坏掉了
    :param terminate: 是否在执行完成之后结束程序
    :return: None
    """
    x = torch.randn(1,1,10,10)
    model = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[3,3], padding=1)
    device = return_free_GPU()
    model.to(device)
    x = x.to(device)
    print_info('begin', 'test the CUDA env')
    z= model(x)
    if z.shape:
        print('The CUDA env is available.')
    exit() if terminate else None



def nms(bboxes, scores, threshold=0.5):
    """
    实现nms算法的底层原理
    :param bboxes: [N, 4]
    :param scores: [N,]
    :param threshold:
    :return: 返回scores经过nms算法满足条件的scores数列的索引值
    """
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1) # [N,] 每个box的面积
    _, order = scores.sort(0, descending=True) # order返回排序过后scores的索引

    keep = []
    while order.numel() > 0:
        if order.numel() == 0: # box只剩一个时
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        # 计算box[i]与其余各框的IoU, 下面的计算的方法就是求交集面积的一个典型的例子，很巧妙
        xx1 = x1[order[1:]].clamp(min=x1[i])  #
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter) # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze()  # 返回idx：所有iou大于阈值threshold的索引
        if idx.numel() == 0:
            break
        order = order[idx+1] # 修补索引之间的差值

    return torch.LongTensor(keep)

# 打印dataloader的shape
def show_dataloader_shape(dataloader=None):
    """
    打印dataloader的shape
    :param dataloader:
    :return:
    """
    raise ValueError('No dataloader input') if dataloader == None else None
    # 获取第一个迭代值
    step = next(iter(dataloader))
    # 如果是有标签的数组对象
    if isinstance(step, list):
        img = step[0]
        label = step[1]
        print("Dataloader is 'list' instance")
        print("{}[0] batch shape: {}".format(varname(dataloader), img.shape))
        print("{}[1] batch shape: {}".format(varname(dataloader), label.shape))
        return
    # 如果是没有标签的tensor对象
    if isinstance(step, torch.Tensor):
        print("Dataloader is 'torch.Tensor' instance")
        print("{} batch shape: {}".format(varname(dataloader), step.shape))
        return
    raise ValueError('The input dataloader type is not suitable')




# 获取变量的变量名
def varname(p):
    """
    获取变量的变量名
    :param p:
    :return:
    """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


# 设置全局得种子
def seed_everything(seed=23):
    """
    设置全局随机种子
    :param seed: 默认23
    :return: seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torchvision.datasets.folder.has_file_allowed_extension()
    return seed



# 在训练的过程中自动打印loss信息
def show_loss(epoch, epochs, step, steps, GAN:bool, *loss):
    """
    在训练的过程中自动打印loss信息
    **All params is required**
    :param epoch:
    :param epochs:
    :param step:
    :param steps:
    :param GAN: bool, if True, loss num >=2
    :param loss: loss tuple to be printed
    :return:
    """
    if loss:
        if (not GAN) & (len(loss) ==1):
            print("[{}/{}][{}/{}] Loss:{}"
                .format(
                epoch + 1, epochs, step, steps,
                loss[0]
            ))
            return
        if GAN & (len(loss) >=2):
            print("[{}/{}][{}/{}] Loss D:{} Loss G: {}"
                .format(
                epoch + 1, epochs, step, steps,
                loss[0], loss[1]
            ))
            return
        raise ValueError("param 'GAN' not match num(loss), check the status instead")

    raise ValueError('List "loss" is null')

