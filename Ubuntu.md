Ubuntu常用
===

模板：


* de
  ``


* 传本地文件到远程（速度：1MB/s）
  `scp C:/Users/haywa/Downloads/vgg16_caffe.pth shaolun@131.123.39.102:/home/shaolun/PYTHON/object-detection/faster-rcnn.pytorch/data/pretrained_model`


* 删除文件夹
  `rm -rf dir_name`


* 退出ssh链接
  `exit`


* 创建虚拟环境
  `mkvirtualenv jwyang-faster-rcnn`

* 进入虚拟环境
  `workon MY_ENV`

* 退出虚拟环境
  `deactivate`

* 删除虚拟环境
  `rmvirtualenv my_env`

* 查看虚拟环境安装了哪些包
  `pip list`

* 创建软链接
  `ln -s VOCdevkit的绝对路径 VOCdevkit2007`

* 根据requirements.txt安装包
  `pip install -r requirements.txtO`

* 查看所有虚拟环境
  进入` /home/shaolun/.virtualenvs/`下查看文件

* 防止pip instal超时
  `pip --default-timeout=1000 install my_lib`

* 检测tf或者torch可不可以使用GPU
  `tf.test.is_gpu_available()`或者
  `torch.cuda.is_available()`

* 检测CUDA
  `nvcc -V`

* pip安装包找不到指定的lib
  `pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html`

* 生成requirements文件
  `pip freeze > requirements.txt`

* 虚拟环境的路径
  `/home/shaolun/.virtualenvs/`

* 有时候pycharm会出现‘Connecting to 131..’ => 'timeout expired'
  有可能是远程服务器正在运行着某个程序导致得连接不上，关闭即可。

* 直接进入~下
  `cd` 即可

* 开启visdom
  `visdom`
  或者
  `python -m visdom.server`

* 查看并中断被占用的端口的进程：
  `lsof -i:8097`=>
  `kill 8097`

* 创建软链接
  `ln -s /home/shaolun//PYTHON/object-detection/faster-rcnn.pytorch/data/VOCdevkit /home/shaolun/PYTHON/GAN/GAN_first/VOCdevkit`
  注意：两个路径都要写绝对路径

* 进入到conda 的base下：
  `source  ~/.bashrc`
      
* 安装vim
  `sudo apt install vim`

* 监控cpu
  `htop`

* 监控gpu
  `watch nvidia-smi`

* ubuntu 查看隐藏文件夹
  `ls -a`

***
* 查看kaggle的安装是否成功
  `kaggle competitions list`

***
* 复制文件
  `cp file ../destination`

* 复制文件夹
  `cp -r dir dest_dir`

* 移动（剪切）文件
  `mv ./file.zip ./dir_name`

---
* 解压文件
  * .tar
    `tar -xvf filename.tar`
  * .gz
    `gunzip filename.gz`
  * .tar.gz
    `tar -zxvf filename.tar.gz`
  * .rar
    `rar x filename.rar`
  * .zip
    `unzip filename.zip`

---
* 只显示前十个图片/文件
  `ls | head -10`

* 生成当前目录的结构树
  `sudo apt install tree`
  `tree`

* 改变tensor的维度，重新排列顺序 
  [W, H, c]
  `img = img.permute(2, 1, 0)`
  [c, H, W]

* 搭建网络Sequential时的基本blcok
  ```python
          def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
  ```

* 退出
  `:q!` 不保存强制退出
  `:wq`         //按【:wq】 保存后离开
  `:wq!`        //按【:wq!】 强制保存后离开
  
 * 通过URL下载
 `wget your_url`