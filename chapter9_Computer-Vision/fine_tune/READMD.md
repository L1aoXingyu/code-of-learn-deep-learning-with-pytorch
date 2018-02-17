## Fine Tune 教程

### Requirements

[PyTorch 0.3](http://pytorch.org/)

[MxTorch](https://github.com/SherlockLiao/mxtorch)

[tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

按照 pytorch 官网安装 pytorch，将 mxtorch 下载下来，放到根目录，安装 tensorboardX 实现 tensorboard 可视化

```bash
\fine_tune
	\mxtorch
	\hymenoptera_data
		\train
		\val
	\checkpoints
	config.py
	main.py
	get_data.sh
```



### 下载数据

打开终端，运行 bash 脚本来获取数据

```bash
bash get_data.sh
```



### 训练模型

所有的配置文件都放在 config.py 里面，通过下面的代码来训练模型

```bash
python main.py train
```

也可以在终端修改配置，比如改变 epochs 和 batch_size

```bash
python main.py train \ 
	--max_epochs=100 \
	--batch_size=16
```

