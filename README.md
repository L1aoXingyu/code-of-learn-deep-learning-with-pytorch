# 深度学习入门之PyTorch 

Learn Deep Learning with PyTorch

非常感谢您能够购买此书，这个github repository包含有[深度学习入门之PyTorch](https://item.jd.com/17915495606.html)的实例代码。由于本人水平有限，在写此书的时候参考了一些网上的资料，在这里对他们表示敬意。由于深度学习的技术在飞速的发展，同时PyTorch也在不断更新，且本人在完成此书的时候也有诸多领域没有涉及，所以这个repository会不断更新作为购买次书的一个后续服务，希望我能够在您深度学习的入门道路上提供绵薄之力。

**注意：由于PyTorch版本更迭，书中的代码可能会出现bug，所以一切代码以该github中的为主。**

![image.png](http://upload-images.jianshu.io/upload_images/3623720-7cc3a383f486d157.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 配置环境

书中已经详细给出了如何基于Anaconda配置python环境，以及PyTorch的安装，如果你使用自己的电脑，并且有Nvidia的显卡，那么你可以愉快地进入深度学习的世界了，如果你没有Nvidia的显卡，那么我们需要一个云计算的平台来帮助我们学习深度学习之旅。[如何配置aws计算平台](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/aws.md)


**以下的课程目录和书中目录有出入，因为内容正在更新到第二版，第二版即将上线！！**
## 课程目录
### part1: 深度学习基础
- Chapter 2: PyTorch基础
    - [Tensor和Variable](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/Tensor-and-Variable.ipynb)    
    - [自动求导机制](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/autograd.ipynb)
    - [动态图与静态图](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/dynamic-graph.ipynb)


- Chapter 3: 神经网络
    - [线性模型与梯度下降](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/linear-regression-gradient-descend.ipynb)
    - [Logistic 回归与优化器](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/logistic-regression/logistic-regression.ipynb)
    - [多层神经网络，Sequential 和 Module](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/nn-sequential-module.ipynb)
    - [深层神经网络](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/deep-nn.ipynb)
    - [参数初始化方法](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/param_initialize.ipynb)
    - 优化算法
        - [SGD](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/sgd.ipynb)
        - [动量法](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/momentum.ipynb)
        - [Adagrad](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/adagrad.ipynb)
        - [RMSProp](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/rmsprop.ipynb)
        - [Adadelta](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/adadelta.ipynb)
        - [Adam](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter3_NN/optimizer/adam.ipynb)
- Chapter 4: 卷积神经网络
    - [PyTorch 中的卷积模块](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/basic_conv.ipynb)
    - [批标准化，batch normalization](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/batch-normalization.ipynb)
    - [使用重复元素的深度网络，VGG](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/vgg.ipynb)
    - [更加丰富化结构的网络，GoogLeNet](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/googlenet.ipynb)
    - [深度残差网络，ResNet](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/resnet.ipynb)
    - [稠密连接的卷积网络，DenseNet](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/densenet.ipynb)
    - 更好的训练卷积网络
        - [数据增强](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/data-augumentation.ipynb)
        - [正则化](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/regularization.ipynb)
        - [学习率衰减](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter4_CNN/lr-decay.ipynb)
- Chapter 5: 循环神经网络
    - [循环神经网络模块：LSTM 和 GRU](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/pytorch-rnn.ipynb)
    - [使用 RNN 进行图像分类](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/rnn-for-image.ipynb)
    - [使用 RNN 进行时间序列分析](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/time-series/lstm-time-series.ipynb)
    - 自然语言处理的应用：
        - [Word Embedding](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/nlp/word-embedding.ipynb)
        - [N-Gram 模型](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/nlp/n-gram.ipynb)
        - [Seq-LSTM 做词性预测](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/nlp/seq-lstm.ipynb)
- Chapter 6: 生成对抗网络
    - [自动编码器](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter6_GAN/autoencoder.ipynb)
    - [变分自动编码器](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter6_GAN/vae.ipynb)
    - [生成对抗网络](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter6_GAN/gan.ipynb)
    - 深度卷积对抗网络 (DCGANs) 生成人脸
- Chapter 7: 深度强化学习
    - [Q Learning](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter7_RL/q-learning-intro.ipynb)
    - [Open AI gym](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter7_RL/open_ai_gym.ipynb)
    - [Deep Q-networks](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter7_RL/dqn.ipynb)
- Chapter 8: PyTorch高级
    - [tensorboard 可视化](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter8_PyTorch-Advances/tensorboard.ipynb)
   - [灵活的数据读取介绍](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter8_PyTorch-Advances/data-io.ipynb)
    - autograd.function 的介绍
    - 数据并行和多 GPU
    - 使用 ONNX 转化为 Caffe2 模型
    - 如何部署训练好的神经网络
    - 打造属于自己的 PyTorch 的使用习惯

### part2: 深度学习的应用
- Chapter 9: 计算机视觉
    - [Fine-tuning: 通过微调进行迁移学习](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter9_Computer-Vision/fine_tune/)
    - kaggle初体验:猫狗大战
    - [语义分割: 通过 FCN 实现像素级别的分类](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/tree/master/chapter9_Computer-Vision/segmentation)
    - Pixel to Pixel 生成对抗网络
    - Neural Transfer: 通过卷积网络实现风格迁移
    - Deep Dream: 探索卷积网络眼中的世界

- Chapter 10: 自然语言处理
    - [Char RNN 实现文本生成](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/blob/master/chapter10_Natural-Language-Process/char_rnn/) 
    - Image Caption: 实现图片字幕生成
    - seq2seq 实现机器翻译
    - cnn + rnn + attention 实现文本识别

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



## Acknowledgement

本书的第二版内容其中一些部分参考了 mxnet gluon 的中文教程，[通过MXNet/Gluon来动手学习深度学习](https://zh.gluon.ai/)。

Gluon 是一个和 PyTorch 非常相似的框架，非常简单、易上手，推荐大家去学习一下，也安利一下 gluon 的中文课程，全中文授课，有视频，有代码练习，可以说是最全面的中文深度学习教程。