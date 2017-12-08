## 深度学习入门之PyTorch 

Learn Deep Learning with PyTorch

非常感谢您能够购买此书，这个github repository包含有[深度学习入门之PyTorch](https://item.jd.com/17915495606.html)的实例代码。由于本人水平有限，在写此书的时候参考了一些网上的资料，在这里对他们表示敬意。由于深度学习的技术在飞速的发展，同时PyTorch也在不断更新，且本人在完成此书的时候也有诸多领域没有涉及，所以这个repository会不断更新作为购买次书的一个后续服务，希望我能够在您深度学习的入门道路上提供绵薄之力。

**注意：由于PyTorch版本更迭，书中的代码可能会出现bug，所以一切代码以该github中的为主。**

![image.png](http://upload-images.jianshu.io/upload_images/3623720-7cc3a383f486d157.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 配置环境

书中已经详细给出了如何基于Anaconda配置python环境，以及PyTorch的安装，如果你使用自己的电脑，并且有Nvidia的显卡，那么你可以愉快地进入深度学习的世界了，如果你没有Nvidia的显卡，那么我们需要一个云计算的平台来帮助我们学习深度学习之旅。[如何配置aws计算平台](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/aws.md)


**以下的课程目录和书中目录有出入，因为内容正在不断更新，所有的内容更新完成会更迭到书的第二版中！**
## 课程目录
### part1: 深度学习基础
- Chapter 2: PyTorch基础
    - [Tensor和Variable](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/Tensor-and-Variable.ipynb)
    - [自动求导机制](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/autograd.ipynb)
    - 数据的读取
    - autograd.function的介绍
    - Module和Sequential
    - 自定义参数的初始化
    - 模型保存和读取

- Chapter 3: PyTorch高级
    - tensorboard可视化
    - 优化器
    - 自定义loss和非标准层
    - 数据并行和多GPU
    - PyTorch的分布式应用
    - 使用ONNX转化为Caffe2模型

- Chapter 4: 多层感知器
    - 线性模型
    - Logistic 回归
    - 多层神经网络

- Chapter 5: 卷积神经网络
    - 从0开始手动搭建卷积网络
    - 批标准化
    - 使用重复元素的深度网络，VGG
    - 更加丰富化结构的网络，GoogLeNet
    - 深度残差网络，ResNet
    - 稠密连接的卷积网络，DenseNet

- Chapter 6: 循环神经网络
    - LSTM和GRU
    - 使用RNN进行时间序列分析
    - 使用RNN进行图像分类
    - Word Embedding和N-Gram模型
    - Seq-LSTM做词性预测

### part2: 深度学习的应用
- Chapter 7: 计算机视觉
    - 图像增强的方法
    - Fine-tuning: 通过微调进行迁移学习
    - 语义分割: 通过卷积实现像素级别的分类
    - 使用卷积网络进行目标检测
    - 使用triplet loss进行人脸识别
    - Neural Transfer: 通过卷积网络实现风格迁移
    - Deep Dream: 探索卷积网络眼中的世界

- Chapter 8: 自然语言处理
    - char rnn实现文本生成
    - 联合卷积网络实现图片字幕
    - 使用rnn进行情感分析
    - seq2seq实现机器翻译
    - cnn+rnn+attention实现文本识别
    - Tree-lstm实现语义相关性分析

### part3: 高级内容
- Chapter 9: 生成对抗网络
    - 自动编码器
    - 变分自动编码器
    - 生成对抗网络的介绍
    - 深度卷积对抗网络(DCGANs)
    - Wasserstein-GANs
    - 条件生成对抗网络(Conditional
     GANs)
    - Pix2Pix

- Chapter 10: 深度增强学习
    - 深度增强学习的介绍
    - Policy gradient
    - Actor-critic gradient
    - Deep Q-networks

## 一些别的资源

关于深度学习的一些公开课程以及学习资源，可以参考我的这个[repository](https://github.com/SherlockLiao/Roadmap-of-DL-and-ML)

可以关注我的[知乎专栏](https://zhuanlan.zhihu.com/c_94953554)和[博客](https://sherlockliao.github.io/)，会经常分享一些深度学习的文章

关于PyTorch的资源

我的github repo [pytorch-beginner](https://github.com/SherlockLiao/pytorch-beginner)

[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

[the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)

[practical-pytorch](https://github.com/spro/practical-pytorch)

[PyTorchZeroToAll](https://github.com/hunkim/PyTorchZeroToAll)

[Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list)
