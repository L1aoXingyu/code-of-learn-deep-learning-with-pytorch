## 配置AWS云计算平台

这是一个帮助文档，我们会一步一步讲解如何从0开始在AWS上申请CPU或者GPU机器进行使用。



### 申请账号并登陆

首先我们需要在[aws官网](https://aws.amazon.com/)上面注册账号，这里需要绑定一张信用卡，所以没有master或者VISA卡的同学可以开通一张，实在不熟悉的同学可以搜索一下"如何注册aws账号"。

然后我们进入到控制面板，可以看到下面的图片，点击"EC2"

![](https://ws1.sinaimg.cn/large/006tNc79gy1fo7xn33e4cj31kw0wo11u.jpg)


然后我们就能够进入到下一个界面


![](https://ws1.sinaimg.cn/large/006tNc79gy1fo7xoznbz3j31kw0j7dmu.jpg)


这个界面只需要注意三个地方，一个是右上角的地区，需要选择一个离你比较近的地区，整个亚太地区可以选择韩国，日本，新加坡和孟买，需要注意的是不同的地区实例价格是不同的，如果你有vpn，那么推荐选择俄勒冈，因为这个地区最便宜，比亚太地区便宜了4到5倍。然后是左边的一个方框"限制"，如果你申请CPU的计算实例，那么不用管，如果你要申请GPU计算实例，就需要点击"限制"进行申请，因为GPU实例会产生费用，亚马逊需要和你确认这个事情，一般需要两到三个工作日。

接下面就可以开始启动实例了，点击中间的红框即可开始。


### 申请实例并启动


![](https://ws2.sinaimg.cn/large/006tNc79gy1fo7xpjsxwlj31kw0q8dp2.jpg)


进入上面的界面之后，需要选择操作系统，这里我们一般都选择linux系统，当然还有很多社区AMI，也就是别人配置好的系统，这里先暂时不用管，我们一般就在上面两个红框中选择一个，第一个是一个空的系统，什么都没有，第二个是一个深度学习的系统，装好了CUDA以及很多框架，可以选择这一个，如果选择这个，那么需要的磁盘空间可能更大。



点击选择之后便可以进入下面的界面。

![](https://ws4.sinaimg.cn/large/006tNc79gy1fo7xqq958cj31kw0ki112.jpg)


这里需要选择实例类型，如果新注册的用户可以免费使用一年的t2.mirco实例，这个实例是没有GPU的，如果要使用GPU的实例，那么从上面的实例类型中选择GPU计算，便可以快速跳转到下面这里。


![](https://ws4.sinaimg.cn/large/006tNc79gy1fo7xr45wmkj31kw0nktgt.jpg)

这里有很多个实例，一般我们就选第一个p2.xlarge，这个实例包含一个Nvidia k40GPU，后面有8块GPU和16块GPU的版本，当然费用也更高。除此之外，下面还有 p3.2xlarge，这里面包含更新的 GPU，速度也会快很多，当然价格也会贵一些，有一点需要注意，选择 p2.xlarge 只能安装 cuda8，而选择 p3.2xlarge 则可以安装 cuda9。选择完成之后我们可以进入下一步配置实例信息。


![](https://ws2.sinaimg.cn/large/006tNc79gy1fo7xrl5bi9j31kw08j77v.jpg)

这里我们只需要关注根目录的大小，也就是云端计算平台的硬盘大小，因为我们需要存放数据集，需要安装框架，所以需要大一点，新注册的用户可以免费试用30G的存储，我们可以设置为40G，一般费用比较便宜。然后点击审核和启动实例。



接着进入到下面这个界面，我们可以点击右下角的启动来启动实例了。


![](https://ws1.sinaimg.cn/large/006tNc79gy1fo7xs8wl8hj31kw0sp13n.jpg)




接着会跳出一个对话框如下。


![9.png](http://upload-images.jianshu.io/upload_images/3623720-4a6cd6ff1321e5fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这里需要你创建一个密钥对，因为现在aws不支持密码登录，所以需要密钥登录，你在名称那一栏取一个名字，然后点击**下载密钥对**就可以了。



然后你就可以看到你的实例正在启动，点击下图红框的地方进入具体的实例位置。


![](https://ws3.sinaimg.cn/large/006tNc79gy1fo7xtcjn2fj31kw0c177o.jpg)



然后可以进入到下面的界面，可以看到实例正在启动，右键点击实例这一栏，然后点击连接。

![](https://ws1.sinaimg.cn/large/006tNc79gy1fo7xtys9mej31kw0iu422.jpg)




接着便会出来下面的窗口，按着这个窗口的操作，如果使用windows系统，需要PuTTY连接，因为我的电脑是mac，所以这个部分没有尝试。在mac下打开终端，先进入刚才存放密钥的位置，然后输出`chmod 400 yourkey.pem`，这里我的密钥是'liao.pem'，这个命令只需要第一次连接的时候输入，后面连接就不用管了，然后通过下面的命令连到你的远程linux服务器。


![12.png](http://upload-images.jianshu.io/upload_images/3623720-1c476e3770c0eb63.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




比如，这是我在终端中进行的输入，第一次连接会弹出一个问题，输入yes即可。

![13.png](http://upload-images.jianshu.io/upload_images/3623720-825156b98dba8b84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



然后我们便进入到了系统，可以看到，红框就表示连接的远程服务器。

![14.png](http://upload-images.jianshu.io/upload_images/3623720-8a19f59377d88055.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




### 安装Anaconda

下面简单演示一下如何在远程环境下安装anaconda，这里需要一点bash命令的基础。首先使用自己的电脑进入到[Anaconda的官网](https://www.anaconda.com/download/#linux)，然后右键点击Download，保存链接地址。

![15.png](http://upload-images.jianshu.io/upload_images/3623720-54ba5def9981eb27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




然后在刚刚连接的远程服务器上面输入

```bash
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
```

后面是刚刚复制的地址，然后输入回车，就开始下载了，下载完成之后是一个后缀为.sh的文件，输入`sudo sh 文件名.sh`就可以开始安装了。


![16.png](http://upload-images.jianshu.io/upload_images/3623720-709e1ab46eb204a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




安装完成之后，通过下面的命令配置环境变量。

```bash
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc
```

这样便完成了远程Anaconda的安装。



### 安装CUDA

【注意】只有CPU的实例可以跳过步骤。

我们去Nvidia官网下载CUDA并安装。选择正确的版本并获取下载地址。

【注意】目前官方默认是 cuda9，如果选择的是 p2.xlarge，则需要安装 cuda8，可以使用下面的命令来下载并安装 cuda8

```bash
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo sh cuda_8.0.61_375.26_linux-run
```



![](https://github.com/mli/gluon-tutorials-zh/blob/master/img/cuda.png?raw=true)

然后使用`wget`下载并且安装 cuda9

```bash
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
```

这里需要回答几个问题。

```
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: y
Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]: y
Do you want to run nvidia-xconfig?
(y)es/(n)o/(q)uit [ default is no ]: n
Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y
Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n
```

安装完成后运行

```bash
nvidia-smi
```

就可以看到这个实例的GPU了。最后将CUDA加入到library path方便之后安装的库找到它。

cuda 8

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64" >>.bashrc
```

cuda 9

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-9.1/lib64" >>.bashrc
```



### 运行Jupyter notebook

接下来在远程终端运行Jupyter notebook。

```bash
jupyter notebook
```

如果成功的话会看到类似的输出

![](https://github.com/mli/gluon-tutorials-zh/blob/master/img/jupyter.png?raw=true)

因为我们的实例没有暴露8888端口，所以我们可以通过ssh映射到本地

```bash
ssh -L8888:locallhost:8888 ubuntu@your-ip.amazonaws.com
```

 然后把jupyter log里的URL复制到本地浏览器就行了。

【注意】如果本地运行了Jupyter notebook，那么8888端口就可能被占用了。要么关掉本地jupyter，要么把端口映射改成别的。例如，假设aws使用默认8888端口，我们可以通过ssh映射到本地8889端口：

```bash
ssh -N -f -L localhost:8889:localhost:8888 ubuntu@your-ip.amazonaws.com
```

然后在本地浏览器打开localhost:8889，这时会提示需要token值。接下来，我们将aws上jupyter log里的token值（例如上图里：...localhost:8888/?token=`token值`）复制粘贴即可。



### 后续

因为云服务按时间计费，通常我们不用时需要把样例关掉，到下次要用时再开。



![17.png](http://upload-images.jianshu.io/upload_images/3623720-6e4fb6cb2d39d66f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如果是停掉（Stop)，下次可以直接继续用，但硬盘空间会计费。如果是终结(Termination)，我们一般会先把操作系统做镜像，下次开始时直接使用镜像（AMI）（上面的教程使用了Ubuntu 16.06 AMI）就行了，不需要再把上面流程走一次。


![18.png](http://upload-images.jianshu.io/upload_images/3623720-e4aac81d991e1a28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




**云虽然很方便，但是不便宜，所以在使用完GPU实例之后一定要记得关掉。**



上面就是整个的配置流程，有问题欢迎提出issue。