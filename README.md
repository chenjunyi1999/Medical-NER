# Medical-NER

## 简介

一个使用`Pytorch` 构建的基于 `BERT+BiLSTM+CRF` 的中文医疗信息命名实体识别程序。

## 项目结构 

`data`:存放训练数据<br>
`config.py`: 模型参数，训练超参数，文件路径等配置信息<br>
`dataset.py`: 定义数据集以及与数据处理相关的函数<br>
`main.py`：主函数<br>
`model.py`:模型文件(BERT+BiLSTM+CRF)<br>
`preprocess.py`:处理原始数据，使用BIO标签 <br>
`utils.py`:一些工具函数(模型训练，验证，测试，推理等)<br>


## 如何使用

  1. **安装依赖库**

```
pip install -r requirements.txt
```

2. **处理原始数据集**

```
python preprocess.py
```

3. **训练模型(含测试结果)**

```
python main.py --mode='train'
```

4. **模型推理**

```
python main.py --mode='infer' --ckpt_name="best" --txt="xxxxxxxxxx(中文输入)"
```


## 参考文献

1. [Pytorch-crf doc](https://pytorch-crf.readthedocs.io/en/stable/)

