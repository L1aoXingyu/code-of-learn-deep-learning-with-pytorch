# Char-RNN-PyTorch
使用字符级别的RNN进行文本生成，使用PyTorch框架。[Gluon实现](https://github.com/SherlockLiao/Char-RNN-Gluon)

## Requirements
[PyTorch 0.3](http://pytorch.org/)

[MxTorch](https://github.com/SherlockLiao/mxtorch)

[tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

按照 pytorch 官网安装 pytorch，将 mxtorch 下载下来，放到根目录，安装 tensorboardX 实现 tensorboard 可视化

```bash
\Char-RNN-PyTorch
	\mxtorch
	\data
	\dataset
	\models
	config.py
	main.py
```



### 训练模型

所有的配置文件都放在 config.py 里面，通过下面的代码来训练模型

```bash
python main.py train
```

也可以在终端修改配置

```bash
python main.py train \
	--txt='./dataset/poetry.txt' \ # 训练用的txt文本
	--batch=128  \ # batch_size
	--max_epoch=300 \ 
	--len=30 \ # 输入RNN的序列长度
	--max_vocab=5000 \ # 最大的字符数量
	--embed_dim=512 \ # 词向量的维度
	--hidden_size=512 \ # 网络的输出维度
	--num_layers=2 \ # RNN的层数
	--dropout=0.5
```

如果希望使用训练好的网络进行文本生成，使用下面的代码

```bash
python main.py predict \
	--begin='天青色等烟雨' \ # 生成文本的开始，可以是一个字符，也可以一段话
	--predict_len=100 \ # 希望生成文本的长度
	--load_model='./checkpoints/CharRNN_best_model.pth' # 读取训练模型的位置
```

## Result
如果使用古诗的数据集进行训练，可以得到下面的结果

```bash
天青色等烟雨翩 黄望堪魄弦夜 逐奏文明际天月辉 豪天明月天趣 天外何山重满 遥天明上天  心空游无拂天外空寂室叨
```

如果使用周杰伦的歌词作为训练集，可以得到下面的结果

```bash
这感觉得可能 我这玻童来 城堡药比生对这些年风天　脚剧飘逐在尘里里步的路 麦缘日下一经经 听觉得远回白择
```
