Python
===

* 返回一个错误提示
  `raise ValueError('我是错误信息')`

* 循环遍历一行写
  `item for item in [1,2,3,4,5]`

---

* 判断某路径下是否存在某文件
  `os.path.exists(os.path.join(VOC_root, "VOCdevkit"))`

* 拼接目录名
  `os.path.join('', '')`

* 创建文件夹
  `os.makedirs("save_weights")`
  
* 列出路径下得的目录
  `os.listdir('../dataset')`

---

* 打印可用gpu数
  `torch.cuda.device_count()`

* 指定GPU
  `device = torch.device("cuda:0")`

* 搬到GPU上去
  `model.cuda()`, `model.to(device)`

* 查看模型的位置
  `next(model.parameters()).device`

* 查看变量位置
  `data.device`

* 默认使用最后一块GPU
  `device = torch.device("cuda:{}".format(torch.cuda.device_count() - 1))`

* 设备类型
  `torch.cuda.type`
---


* Pycharm debug 'loading timeout'
  >在PyCharm，打开Setting界面，在如下设置项中勾选“Gevent compatible”即可

* 显示所有的断点
  `ctrl shift F8`

* 打印一个模型里面的所有layer
  `for layer in self.children()`

* 初始化模型的参数
  ```python
  for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
  ```

* 减少pycharm启动时Connecting to XX.XX.XX 的时间过长
  >开启的时候断网，报警告后连上，即可 直接调试

* 迭代器
  ```python
  arr = [1,2,3,4,5]
  arr = iter(arr)
  x = next(arr)
  print(x)
  x = next(arr)
  print(x)
  x = next(arr)
  print(x)
  ```

* 设置pytorch的随机种子
  `torch.manual_seed(23)`

* 设置pytorch的随机种子
  `np.random.seed(23)`

* pycharm Upload
  `Ctrl+Alt+x` 上传到默认server
  `Ctrl+Alt+Shift+x` 上传到指定server

* 在每一次更新w前需要：
  `optim.zero_grad()`

* model.parameters()的内容
  返回的是一个generator，每个迭代器里有两个元素
  分别是：[0]: W, [1]: b

* 获得迭代器所有元素
  `[item for item in model.parameters()]`
---

* 获得迭代器的第一个元素
  `next(model.parameters())`
---
  
* `loss.item()`
  获得的是loss的float的形式


* argparse
  
    ```python
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--family', default='../dataset', type=str,help='姓')
        parser.add_argument('--name', type=str,help='名')
        args = parser.parse_args()

        #打印姓名
        print(args.family+args.name)

    ```
    在命令行中输入:

    `python demo.py --family=张 --name=三`

    运行结果

    `张三`

* 解决`Can't get remote credentials for deployment server`
  `原因是有另一个项目和该项目共享同一个pycharm的解释器，删除该项目，再重新设置解释器即可`

* `torch.nn`
  * `nn.Linear(in_size, out_size)` # `in_size = w * h`
  * `nn.Conv2D(in_channel, out_channel, kernel_size, stride, padding)`

---
* 现将依赖的环境冷冻起来：`pip freeze > requirements.txt`
* 创建一个新的空虚拟环境：`mkvirtualenv3 blog`
* 选择新的虚拟环境：`workon blog`
* 安装相关依赖包：`pip install -r requirements.txt`
---

* >cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
  这类错误无端发生的原因往往是环境出错了
  所以重新创建一个虚拟环境然后运行即可


---
* torch.transforms
  * Resize(int) # 将短边缩放到这个数值，长边根据原本的比例缩放
  * CenterCrop(int) # 从中心剪出一个[int, int]大小的图片
  * ToTensor() # 将一个PIL Image格式的图片转换为tensor格式的图片
  * Normalize(mean, std) # e.g. mean = (0.5, 0.5, 0.5)

---
* torch常用模块的目录树
```
|-- torch
    |-- nn
        |-- Conv2D() class
        |-- ConvTranspose2d() class
        |-- BatchNorm2d() class
        |-- functional
            |-- F
                |-- leaky_relu() function

    |-- optim
    |-- utils
        |-- data
            |-- Dataloader
    |-- tanh() function
    |-- sigmoid() function


|-- torchvision
    |-- datasets
        |-- ImageFolder
    |-- transforms
    |-- utils
        save_images

```

* pytorch 支持的图片类型
  .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp

* Error: Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp
  需要将image下再建立一个文件夹，然后把图片放里面就行了

* size和shape
  * tesor.size 是函数 batch_size = a.size(0)
  * tensor.shape 是属性

* _方法
  * a.fill_(1) # 讲tensor a填充元素1 


* 求batch的总个数
  `len(train_loader)`

* 求一个batch的长度
  `images.size(1)`

---
* 保存和读取网络训练的模型
  ```
    一、

    1. 先建立一个字典，保存三个参数：
    state = {‘net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}

    2.调用torch.save():

    torch.save(state, dir)

    其中dir表示保存文件的绝对路径+保存文件名，如'/home/qinying/Desktop/modelpara.pth'

    二、

    当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。

    checkpoint = torch.load(dir)

    model.load_state_dict(checkpoint['net'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch'] + 1
  ```


* 维度变换
  ```python
  >>> x = torch.randn(2, 3, 5) 
  >>> x.size() 
  torch.Size([2, 3, 5]) 
  >>> x.permute(2, 0, 1).size() 
  torch.Size([5, 2, 3])

  ```
  
 * 返回所有满足条件的索引
    `(tensor>=threshold).nonzero().squeeze()`
    squeeze可能的作用时防止报错，也可以不加
    
* csv返回的df制作成矩阵

   `df = pd.read_csv`
   
   `df.iloc[[1,3],0:3].values.reshape(28, 28).astype(np.uint8)`
   
   
* 关于Dataset，Dataloader的构造
    * Dataset的构造范例
    ```python
    class DigitDataset(Dataset):
    # init, len, getitem
    def __init__(self, path, transform=None):
        self.df = pd.read_csv(root)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.df.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8)
        label = self.df.iloc[idx, 0].astype(np.uint8)

        # transform
        image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.df)
    ```
    返回的数据格式：
    一共有N个tuple，每一个tuple是__getitem__里面的return，针对是每一个图片的矩阵和标签
    
    * DataLoader的范例
    ```python
    dataloaders = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    ```
    返回的数据格式：
    有标签的数据：
    \# Dataloader is 'list' instance
    
    \# dataloader[0] batch shape: torch.Size([32, 1, 28, 28])
    
    \# dataloader[1] batch shape: torch.Size([32])
    
    无标签的数据：
    \# dataloader batch shape: torch.Size([32, 1, 28, 28])



* visdom remote的解决方案
  `xshell设置隧道转发`
  `把主机配置的隧道的目标设置成‘131.123.39.102’`
  `在Xshell中启动ssh链接`

* Visdom remote的步骤
  1. 打开Xshell，连接配置好的guanguan
  2. 一直开着Xshell，不能关闭
  3. 打开浏览器，输入`localhost:8097`

* argparse
  
    ```python
        import argparse

        parser = argparse.ArgumentParser(description='姓名')
        parser.add_argument('--family', type=str,help='姓')
        parser.add_argument('--name', type=str,help='名')
        args = parser.parse_args()

        #打印姓名
        print(args.family+args.name)

    ```
    在命令行中输入:

    `python demo.py --family=张 --name=三`

    运行结果

    `张三`

* 远程传输图像
  经测试，通过Xshell+Xming转发的图片的显示 过慢，速度小于直接用pycharm运行显示图像的速度。所以以后首选用Pycharm绘图并保存，如果不行的话再用Xshell+Xming转发服务

* Xshell+Xming绘图
  1. 打开Xshell，连接到远程服务器
  2. 打开Xming， 配置的DISPLAY变量要写Xshell的隧道中的变量的`.`之前的数字。（我配置的时候是10）
  3. 用Xshell键入`python plot.py`进行绘图
  4. 结果会弹出一个Xming的小窗口，然后图像从雪花点慢慢变成绘制的图像。过程及其满，需要十几分钟。