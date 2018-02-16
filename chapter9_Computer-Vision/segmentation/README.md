## 语意分割教程

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



### 训练效果

