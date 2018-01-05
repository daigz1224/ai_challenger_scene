# ai_challenger_scene

## 1. 题目简介
- [官网](https://challenger.ai/competition/scene)


- 如何根据图像的视觉内容为图像赋予一个语义类别（例如，教室、街道等）是图像场景分类的目标，也是图像检索、图像内容分析和目标识别等问题的基础。但由于图片的尺度、角度、光照等因素的多样性以及场景定义的复杂性，场景分类一直是计算机视觉中的一个挑战性问题。
- 8 万张图片，分属于 80 个日常场景类别，例如航站楼，足球场等。每个场景类别包含 600-1100 张图片。数据集分为训练 (70%)、验证 (10%)、测试 A (10%)与测试 B (10%)四部分。

## 2. 数据说明

- 训练标注数据包含照片 id 和所属场景类别标签号。训练数据文件与验证数据文件的结构如下所示：

```json
[
    {
        "image_id":"5d11cf5482c2cccea8e955ead0bec7f577a98441.jpg",
        "label_id": 0
    },
    {
        "image_id":"7b6a2330a23849fb2bace54084ae9cc73b3049d3.jpg",
        "label_id": 11
    },
]
```

- label_id 有场景的中英文对照，例如 label_id 为 0，表示航站楼，airport_terminal。

## 3. 任务目标

- 以 json 格式提交预测结果：

```json
[
    {
        "image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg",
        "label_id": [5,18,32]
    },
    {
        "image_id":"b5a9a726c9d752d8ac1c722182512d33e66a6f88.jpg",
        "label_id": [6,33,35]
    },
]
```

- 评价的标准以算法在测试集图片上的预测正确率作为最终评价标准，总体正确率函数S为：

$$S=\cfrac{1}{N}\sum\limits^N_{i=1}p_i$$

其中，N 为测试集图片数目，$p_i$ 为第 i 张图片的准确度。以置信度递减的顺序提供三个分类的标签号，记为 $l_j,j=1,2,3$。对图片i的真实标签值记为 $g_i$ ,如果三个预测标签中包含真实标签值，则预测准确度为 1，否则准确度为 0，即：

$$p_i=\min\limits_jd(l_i,g_i)$$

当 $l_i=g_i$时，$d(l_i,g_i)=1$ ，否则为 0。

## 4. 数据处理

- 程序 data_process.py 将 json 文件生成二进制 scene.pth，包含 train，test，val，id2label 的信息。
- 计算图像均值和方差

```python
r = 0 # r mean
g = 0 # g mean
b = 0 # b mean
r_2 = 0 # r^2
g_2 = 0 # g^2
b_2 = 0 # b^2
total = 0
for img_name in img_list:
    img = mx.image.imread(path + img_name) # ndarray, width x height x 3
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]
 
    r += img[:, :, 0].sum().asscalar()
    g += img[:, :, 1].sum().asscalar()
    b += img[:, :, 2].sum().asscalar()
 
    r_2 += (img[:, :, 0]**2).sum().asscalar()
    g_2 += (img[:, :, 1]**2).sum().asscalar()
    b_2 += (img[:, :, 2]**2).sum().asscalar()
r_mean = r / total
g_mean = g / total
b_mean = b / total
r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2
```

- 定义 ClsDataset 的 `__getitem__` 方法，通过 scene.pth 和 train_dir, val_dir, test_dir 的信息，使用 `input, label, _ = data` 获得训练集和验证集的图片数据和标签，使用 `input, label, image_ids = data` 获得测试集的图片数据，标签和 ID 信息。

```python
class ClsDataSet(gl.data.Dataset):
    def __init__(self, json_file, img_path, transform):
        self._img_path = img_path
        self._transform = transform
        with open(json_file, 'r') as f:
            annotation_list = json.load(f)
        self._img_list = [[i['image_id'], i['label_id']]
                          for i in annotation_list]
    def __getitem__(self, idx):
        img_name = self._img_list[idx][0]
        label = np.float32(self._img_list[idx][1])
        img = mx.image.imread(os.path.join(self._img_path, img_name))
        img = self._transform(img)
        return img, label
    def __len__(self):
        return len(self._img_list)
```

- 验证集/测试集  transforms 包括：


```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_transforms =  transforms.Compose([
            transforms.Scale(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            normalize,
        ])
```

- 训练集 transforms 包括随机裁剪，随机水平翻转，归一化：

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_transforms =  transforms.Compose([
            transforms.RandomSizedCrop(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
```

## 5. 模型训练

- 超参数配置

|    超参数     |     数值     |              说明               |
| :--------: | :--------: | :---------------------------: |
|  img_size  |    256     |           处理后的图像尺寸            |
|    lr1     |     0      |       （如果用了预训练模型）特征层学习率       |
|    lr2     |   0.0005   |            分类层学习率             |
|  lr_decay  |    0.5     |             学习率衰减             |
| batch_size |    128     |              批大小              |
| max_epoch  |    100     |           最大数据集迭代次数           |
|  shuffle   |    True    |       DataLoader时的混洗操作        |
| plot_every |     10     |           每十步看一次结果            |
|  workers   |     4      |          CPU 多线程加载数据          |
|    loss    |  'celoss'  | `torch.nn.CrossEntropyLoss()` |
|   model    | 'resnet34' |            默认加载模型             |
| load_path  |    None    |          默认不加载预训练模型           |

- Adam 优化算法

```python
def get_optimizer(self, lr1, lr2):
        self.optimizer =  torch.optim.Adam(
            [
             {'params': self.features.parameters(), 'lr': lr1},
             {'params': self.classifier.parameters(), 'lr':lr2}
            ] )
        return self.optimizer
```

- 使用 ResNet-34 模型训练

```python
def resnet34(opt):
    model = torchvision.models.resnet34(pretrained=not opt.load_path)
    return ResNet(model, opt, feature_dim=512, name='res34')
```

- 使用 Place365 的预训练模型 ResNet-50

```python
def resnet365(opt):
    model = torch.load('checkpoints/whole_resnet50_places365.pth.tar')
    return ResNet(model, opt, name='res_365')
```

## 6. 结果

- ResNet-34 模型训练的结果：在测试集上得到 93.8 % 的准确率。
- 使用 Place365 的预训练模型 ResNet-50 在测试集上达到 95.7 % 的准确率。
- 最终排名前 30% 左右，考虑到一直在使用的云服务器，后面也就没有继续调参以及模型融合。

## 附录1：运行流程

```shell
ssh -i "aws_oregon.pem" -L8888:localhost:8888 ubuntu@ec2-54-190-13-238.us-west-2.compute.amazonaws.com
ssh -i "aws_oregon.pem" -L8097:localhost:8097 ubuntu@ec2-54-190-13-238.us-west-2.compute.amazonaws.com
```

1. 新建 AWS 竞价实例
2. 更新，安装必要软件
3. 安装 CUDA 8.0, Miniconda
4. 创建 pytorch 虚拟环境
5. Git clone 下载代码
6. 新建 data 文件夹，上传数据集
7. 新建 checkpoints 文件夹保存模型
8. 修改 utils.py 中文件的路径

## 附录2：各个程序文件的作用

- *data.process.py* -- 将 train，val 的 json 文件，标签文件，测试集生成二进制 scene.pth。


- *dataset.py* -- 定义 ClsDataset 类读入 train, test, val 数据集，分别定义 train 和 val/test 数据集的 transforms 方法。
- *basic_module.py* -- 封装 `torch.nn.Module`，定义 load, save, get_optimizer, update_optimizer 方法。
- *loss.py* -- 定义 loss 函数（源自 torch）。
- *resnet.py* -- 定义各个 ResNet 的网络模型 （源自 `torchvision.models`，或预训练模型文件 .pth.tar）。
- *utils.py* -- 封装 visdom 的使用，定义了评价准则的计算方法，设置当前默认超参数以及根据命令行更新参数的 parse 方法。
- *main.py*
  - *train* -- 读入参数，建立模型，定义优化方法和 loss 函数，通过 DataLoader 读入数据，迭代数据集（前向传播，后向传播，优化更新参数，计算损失和准确率，可视化结果），计算验证集分数（如果分数上升，保存模型和分数后继续；如果分数下降，降低学习率，加载回之前的最好模型），多次迭代。
  - *val* -- 计算模型在验证集上的分数。
  - *submit* -- 测试验证集，并生成可提交的 json 文件。
