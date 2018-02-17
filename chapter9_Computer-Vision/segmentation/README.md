## 语意分割教程

### Requirements

[PyTorch 0.3](http://pytorch.org/)

[MxTorch](https://github.com/SherlockLiao/mxtorch)

[tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (optional)

按照 pytorch 官网安装 pytorch，将 mxtorch 下载下来，放到根目录，同时可以安装 tensorboardX 实现 tensorboard 可视化

```bash
\segmentation
	\mxtorch
	\data
	\models
	\dataset
	\checkpoints
	config.py
	main.py
```



### 下载数据

打开终端，运行 bash 脚本来获取数据

```bash
bash get_data.sh
```



### 训练模型

所有的配置文件都放在 config.py 里面，通过下面的代码来训练模型

```python
python main.py train
```

也可以在终端修改配置，比如改变 epochs 和 batch_size

```python
python main.py train \ 
	--max_epochs=100 \
	--batch_size=16
```

#### 可选

如果安装了 tensorboardX，则可以使用 tensorboard 可视化，通过

```bash
python main.py train --vis_dir='./vis'
```



### 训练效果

#### 准确率，iou 和 loss

![](https://ws3.sinaimg.cn/large/006tNc79gy1fojg2ye52uj30td07sgm6.jpg)

#### 分割效果

![](https://ws1.sinaimg.cn/large/006tNc79gy1fojg42xvvaj30us0haq4o.jpg)



![](https://ws3.sinaimg.cn/large/006tNc79gy1fojiid8vpbj30hk0fvq3l.jpg)