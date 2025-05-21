**第一部分：环境准备与数据加载**

**(幻灯片/屏幕显示代码：第 1-10 行)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
```

*   **解说：**
    *   首先，我们导入所有必要的库。
    *   `torch`, `torch.nn`, `torch.optim`: PyTorch 核心库，用于构建神经网络和优化。
    *   `torch.nn.functional as F`: 提供了很多激活函数（如 ReLU）和池化操作等，通常以函数形式调用。
    *   `numpy`: 用于数值计算，特别是在数据预处理阶段。
    *   `matplotlib.pyplot`: 用于绘图，如训练曲线和图像可视化。
    *   `os` 和 `struct`: 用于处理文件路径和读取 MNIST 二进制数据文件。
    *   `from torchvision import transforms`: `torchvision` 库包含了许多用于计算机视觉的工具，`transforms` 模块用于图像预处理和数据增强。
    *   `from torch.utils.data import Dataset, DataLoader`: `Dataset` 是表示数据集的抽象类，`DataLoader` 则用于高效地加载和迭代数据。

**(幻灯片/屏幕显示代码：第 12-17 行)**

```python
# ...existing code...
# 设置随机种子以确保结果可复现
torch.manual_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 指定使用CPU
device = torch.device('cpu')
# ...existing code...
```

*   **解说：**
    *   `torch.manual_seed(42)`: 设置 PyTorch 的随机种子为42，确保实验结果的可复现性。
    *   `plt.rcParams[...]`: Matplotlib 配置，用于正确显示中文和负号。
    *   `device = torch.device('cpu')`: 指定模型和数据在 CPU 上运行。如果你的机器有兼容的 GPU，可以改为 `torch.device('cuda')`。

**(幻灯片/屏幕显示代码：第 20-32 行)**

```python
# ...existing code...
# 从本地文件加载MNIST数据集
def load_mnist(path, kind='train'):
    """从本地文件加载MNIST数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images.reshape(len(labels), 28, 28)  # 重塑为 (N, H, W) 格式
    
    return images, labels
# ...existing code...
```

*   **解说：**
    *   `load_mnist` 函数用于从本地二进制文件加载 MNIST 数据。
    *   它首先构建标签和图像文件的完整路径。
    *   使用 `struct.unpack` 解析文件头信息，如图像数量、行数、列数。
    *   `np.fromfile` 读取实际的像素数据和标签数据。
    *   关键的一步是 `images = images.reshape(len(labels), 28, 28)`。这里将每张展平的图像 (784像素) 重新塑形为 (N, H, W) 格式，即 (样本数, 高度, 宽度)，也就是 (N, 28, 28)。这是 PyTorch 中处理图像的常见格式之一，特别是对于单通道图像。

**(幻灯片/屏幕显示代码：第 34-47 行)**

```python
# ...existing code...
# 自定义MNIST数据集类
class MnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
# ...existing code...
```

*   **解说：**
    *   这里我们定义了一个自定义的 `MnistDataset` 类，它继承自 `torch.utils.data.Dataset`。自定义 Dataset 类需要实现三个核心方法：
        *   `__init__(self, images, labels, transform=None)`: 构造函数，接收图像数据、标签数据和一个可选的 `transform` (用于数据预处理)。
        *   `__len__(self)`: 返回数据集中样本的总数。
        *   `__getitem__(self, idx)`: 根据索引 `idx` 获取单个样本。它取出对应的图像和标签。如果定义了 `transform`，则将其应用于图像上。最后返回处理后的图像和标签。

**(幻灯片/屏幕显示代码：第 49-58 行)**

```python
# ...existing code...
# 加载数据集
try:
    current_dir = r"C:\Users\swuis\Desktop\asd\Algorithm Design"
    X_train, y_train = load_mnist(current_dir, kind='train')
    X_test, y_test = load_mnist(current_dir, kind='t10k')
    
    print(f'训练数据集: {X_train.shape}')
    print(f'测试数据集: {X_test.shape}')
except FileNotFoundError:
    print("未找到MNIST数据文件，请确保文件位于指定目录下")
    exit(1)
# ...existing code...
```

*   **解说：**
    *   调用 `load_mnist` 函数加载训练集和测试集。请确保 `current_dir` 指向你存放 MNIST 数据的正确路径。
    *   打印加载后的数据集形状，训练集应为 (60000, 28, 28)，测试集为 (10000, 28, 28)。
    *   使用 `try-except` 块处理文件未找到的异常。

**第二部分：数据预处理与 DataLoader**

**(幻灯片/屏幕显示代码：第 60-63 行)**

```python
# ...existing code...
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])
# ...existing code...
```

*   **解说：**
    *   `transforms.Compose([...])`: 这是一个可以将多个图像转换操作串联起来的工具。
    *   `transforms.ToTensor()`: 这个转换非常重要。
        *   它将输入的 PIL 图像或 NumPy 数组（形状为 HxWxC，对于灰度图是 HxW）转换为 PyTorch 张量。
        *   它会将像素值从 [0, 255] 的范围重新缩放到 [0.0, 1.0] 的范围。
        *   它还会改变图像的维度顺序，从 (H, W, C) 变为 (C, H, W)。对于我们的 (28, 28) 灰度图，会变成 (1, 28, 28) 的张量，其中 1 代表通道数。
    *   `transforms.Normalize((0.1307,), (0.3081,))`: 对张量进行标准化处理。
        *   公式是 `output = (input - mean) / std`。
        *   `(0.1307,)` 是 MNIST 数据集在所有像素上的平均值，`(0.3081,)` 是标准差。这些值是根据 MNIST 数据集统计得出的。注意，它们是单元素元组，因为 MNIST 是单通道（灰度）图像。
        *   标准化有助于模型更快收敛，并可能提高性能。

**(幻灯片/屏幕显示代码：第 65-76 行)**

```python
# ...existing code...
# 创建数据集
train_dataset = MnistDataset(
    images=X_train.astype(np.float32) / 255.0, 
    labels=y_train, 
    transform=transform
)

test_dataset = MnistDataset(
    images=X_test.astype(np.float32) / 255.0, 
    labels=y_test, 
    transform=transform
)
# ...existing code...
```

*   **解说：**
    *   使用我们之前定义的 `MnistDataset` 类来创建训练和测试数据集实例。
    *   `images=X_train.astype(np.float32) / 255.0`: 在将 NumPy 图像数组传递给 `MnistDataset` 之前，我们先将其数据类型转换为 `float32` 并除以 `255.0`，将其值缩放到 [0, 1] 范围。虽然 `transforms.ToTensor()` 也会做这个缩放（如果输入是 `uint8`），但这里显式操作确保了输入给 `ToTensor` 的是浮点型 NumPy 数组，`ToTensor` 会直接将其转换为张量而不再进行缩放。
    *   `labels=y_train`: 传递标签。
    *   `transform=transform`: 将我们定义的 `transform` 对象传递给数据集，这样在 `__getitem__` 中就会应用这些转换。

**(幻灯片/屏幕显示代码：第 78-80 行)**

```python
# ...existing code...
# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# ...existing code...
```

*   **解说：**
    *   `batch_size = 64`: 设置每个批次加载的样本数量。
    *   `DataLoader(train_dataset, batch_size=batch_size, shuffle=True)`: 创建训练数据加载器。
        *   `shuffle=True`: 在每个 epoch 开始时打乱训练数据，这有助于提高模型的泛化能力。
    *   `DataLoader(test_dataset, batch_size=batch_size, shuffle=False)`: 创建测试数据加载器。
        *   `shuffle=False`: 测试时通常不需要打乱数据。

**第三部分：定义 CNN 模型**

**(幻灯片/屏幕显示代码：第 82-107 行)**

```python
# ...existing code...
# 定义CNN模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 第一个卷积层，输入1通道，输出32特征图
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 第二个卷积块
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
# ...existing code...
```

*   **解说：**
    *   我们定义了一个名为 `ConvNet` 的 CNN 模型，继承自 `nn.Module`。
    *   **`__init__` 方法 (构造函数):**
        *   `super(ConvNet, self).__init__()`: 调用父类构造函数。
        *   **第一个卷积块:**
            *   `self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)`:
                *   定义第一个二维卷积层。
                *   `in_channels=1`: 输入图像是单通道（灰度图）。
                *   `out_channels=32`: 该卷积层将产生32个特征图（或称为通道）。
                *   `kernel_size=3`: 卷积核大小为 3x3。
                *   `stride=1`: 卷积核滑动的步长为1。
                *   `padding=1`: 在图像边缘填充1圈0。对于3x3的核，padding=1可以保持输入输出图像的尺寸不变（在不考虑池化的情况下）。
            *   `self.bn1 = nn.BatchNorm2d(32)`:
                *   定义一个批量归一化层，作用于32个通道。批量归一化可以加速训练，提高模型稳定性，并有一定的正则化效果。
            *   `self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)`:
                *   定义一个最大池化层。
                *   `kernel_size=2`: 池化窗口大小为 2x2。
                *   `stride=2`: 池化窗口滑动的步长为2。这将使特征图的尺寸减半。
        *   **第二个卷积块:**
            *   `self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)`:
                *   输入通道数为32（来自 `conv1` 的输出），输出通道数为64。
            *   `self.bn2 = nn.BatchNorm2d(64)`: 批量归一化层，作用于64个通道。
            *   `self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)`: 再次进行最大池化，尺寸再次减半。
        *   **全连接层:**
            *   `self.fc1 = nn.Linear(64 * 7 * 7, 128)`:
                *   定义第一个全连接层。
                *   输入特征数 `64 * 7 * 7` 是如何计算的？
                    *   原始输入图像尺寸: 28x28。
                    *   经过 `conv1` (padding=1, stride=1): 尺寸仍为 28x28。
                    *   经过 `pool1` (kernel=2, stride=2): 尺寸变为 14x14。
                    *   经过 `conv2` (padding=1, stride=1): 尺寸仍为 14x14。
                    *   经过 `pool2` (kernel=2, stride=2): 尺寸变为 7x7。
                    *   此时，我们有64个这样的 7x7 特征图。所以展平后的特征数量是 `64 * 7 * 7 = 3136`。
                *   输出特征数为128。
            *   `self.dropout = nn.Dropout(0.3)`:
                *   定义一个 Dropout 层，丢弃率为0.3 (即30%的神经元输出会被随机置零)。Dropout 是一种正则化技术，用于防止过拟合。它只在训练时生效。
            *   `self.fc2 = nn.Linear(128, 10)`:
                *   第二个全连接层，也是输出层。输入特征数为128，输出特征数为10（对应0-9这10个类别）。
    *   **`forward` 方法 (定义前向传播):**
        *   `x` 是输入数据，形状为 (batch_size, 1, 28, 28)。
        *   `x = self.pool1(F.relu(self.bn1(self.conv1(x))))`:
            *   数据首先通过 `conv1`，然后通过 `bn1` 进行批量归一化，接着通过 `F.relu` 激活函数（ReLU 是一种常用的非线性激活函数），最后通过 `pool1` 进行最大池化。
        *   `x = self.pool2(F.relu(self.bn2(self.conv2(x))))`:
            *   类似地，数据通过第二个卷积块。
        *   `x = x.view(-1, 64 * 7 * 7)`:
            *   在进入全连接层之前，需要将池化后的多维特征图展平（flatten）为一维向量。
            *   `x.view(-1, num_features)`: `-1` 表示该维度的大小由其他维度和总元素数量自动推断。这里是将每个样本的 (64, 7, 7) 特征图展平为长度为 `64*7*7` 的向量。
        *   `x = F.relu(self.fc1(x))`: 通过第一个全连接层，并应用 ReLU 激活。
        *   `x = self.dropout(x)`: 应用 Dropout。
        *   `x = self.fc2(x)`: 通过输出层，得到10个类别的原始分数（logits）。
        *   `return x`: 返回模型的输出。

**第四部分：模型实例化、损失函数与优化器**

**(幻灯片/屏幕显示代码：第 109-110 行)**

```python
# ...existing code...
# 实例化模型
model = ConvNet().to(device)
print(model)
# ...existing code...
```

*   **解说：**
    *   `model = ConvNet().to(device)`: 实例化我们定义的 `ConvNet` 模型，并使用 `.to(device)` 将其所有参数和缓冲区移动到之前指定的计算设备（CPU 或 GPU）上。
    *   `print(model)`: 打印模型结构，可以帮助我们检查网络是否按预期构建。

**(幻灯片/屏幕显示代码：第 112-113 行)**

```python
# ...existing code...
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# ...existing code...
```

*   **解说：**
    *   `criterion = nn.CrossEntropyLoss()`: 定义损失函数。交叉熵损失是分类任务中常用的损失函数，它内部包含了 Softmax 操作，所以我们的模型输出层不需要显式添加 Softmax。
    *   `optimizer = optim.Adam(model.parameters(), lr=0.001)`: 定义优化器。
        *   `optim.Adam`: Adam 是一种高效的自适应学习率优化算法。
        *   `model.parameters()`: 将模型中所有需要训练的参数传递给优化器。
        *   `lr=0.001`: 设置学习率。

**第五部分：模型训练**

**(幻灯片/屏幕显示代码：第 115-121 行)**

```python
# ...existing code...
# 训练模型
num_epochs = 1  # 轮次
train_losses = []
train_losses_step = [] # 每100步的训练损失
test_accuracies_step = [] # 每100步的测试准确率
running_loss_step = 0.0 # 用于计算每100步的训练损失

print("开始训练...")
# ...existing code...
```

*   **解说：**
    *   `num_epochs = 1`: 设置训练的总轮数。
    *   `train_losses = []`: 用于存储每个 epoch 结束后的平均训练损失。
    *   `train_losses_step = []`: 用于存储每100个训练步骤的平均训练损失。
    *   `test_accuracies_step = []`: 用于存储每100个训练步骤后在测试集上的准确率。
    *   `running_loss_step = 0.0`: 一个累加器，用于计算过去100步的训练损失总和。
    *   打印开始训练的提示。

**(幻灯片/屏幕显示代码：第 122-140 行)**

```python
# ...existing code...
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # running_loss_step 在外部初始化，并在每100步后重置
    
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到设备
        images = images.to(device)
        labels = labels.long().to(device) # 更新标签转换
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() # 累加当前批次的损失，用于计算轮次平均损失
        running_loss_step += loss.item() # 累加用于计算每100步的平均损失
# ...existing code...
```

*   **解说：**
    *   **外层循环 `for epoch in range(num_epochs):`**: 遍历每个训练轮次。
        *   `model.train()`: 将模型设置为训练模式。这会启用 Dropout 和 BatchNorm 等层的训练特定行为。
        *   `running_loss = 0.0`: 初始化当前 epoch 的总损失。
    *   **内层循环 `for i, (images, labels) in enumerate(train_loader):`**: 遍历训练数据加载器中的每个批次。
        *   `images = images.to(device)`: 将图像数据移到指定设备。
        *   `labels = labels.long().to(device)`: 将标签数据转换为 `long` 类型（`torch.int64`）并移到设备。`CrossEntropyLoss` 期望目标标签是长整型。
        *   `optimizer.zero_grad()`: 清除先前计算的梯度，防止梯度累积。
        *   `outputs = model(images)`: 执行前向传播，获取模型输出。
        *   `loss = criterion(outputs, labels)`: 计算损失。
        *   `loss.backward()`: 执行反向传播，计算梯度。
        *   `optimizer.step()`: 根据梯度更新模型参数。
        *   `running_loss += loss.item()`: 累加当前批次的损失值（`.item()` 从单元素张量中获取 Python 数字），用于计算整个 epoch 的平均损失。
        *   `running_loss_step += loss.item()`: 累加损失，用于计算每100步的平均损失。

**(幻灯片/屏幕显示代码：第 142-166 行) - 中间验证部分**

```python
# ...existing code...
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}')
            # 记录每100步的训练损失 (平均值)
            train_losses_step.append(running_loss_step / 100)
            running_loss_step = 0.0 # 重置100步的累加损失
            
            # 每100步进行一次中间验证
            model.eval()
            num_test_batches_for_step = 0
            correct_step = 0
            total_step = 0
            with torch.no_grad():
                for test_images, test_labels in test_loader:
                    test_images = test_images.to(device)
                    test_labels = test_labels.long().to(device)
                    
                    test_outputs = model(test_images)
                    num_test_batches_for_step += 1
                    
                    _, predicted_step = torch.max(test_outputs.data, 1)
                    total_step += test_labels.size(0)
                    correct_step += (predicted_step == test_labels).sum().item()
            
            current_step_accuracy = 100 * correct_step / total_step if total_step > 0 else 0
            
            test_accuracies_step.append(current_step_accuracy) # 记录每100步的测试准确率
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Intermediate Test Accuracy: {current_step_accuracy:.2f}%')
            model.train() # 切回训练模式
# ...existing code...
```

*   **解说：**
    *   `if (i+1) % 100 == 0:`: 每训练100个批次，执行以下操作：
        *   打印当前的 Epoch、Step 和该批次的损失。
        *   `train_losses_step.append(running_loss_step / 100)`: 计算这100步的平均训练损失并记录。
        *   `running_loss_step = 0.0`: 重置100步损失累加器。
        *   **中间验证:**
            *   `model.eval()`: 将模型切换到评估模式。这会禁用 Dropout，并使 BatchNorm 使用训练时学习到的均值和方差。
            *   初始化 `correct_step` 和 `total_step` 用于计算当前验证的准确率。
            *   `with torch.no_grad():`: 在评估模式下，不需要计算梯度。
            *   `for test_images, test_labels in test_loader:`: 遍历整个测试数据集。
                *   将测试数据移到设备。
                *   `test_outputs = model(test_images)`: 获取模型在测试数据上的预测。
                *   `_, predicted_step = torch.max(test_outputs.data, 1)`: 获取预测类别。
                *   累加 `total_step` 和 `correct_step`。
            *   `current_step_accuracy = 100 * correct_step / total_step`: 计算当前测试准确率。
            *   `test_accuracies_step.append(current_step_accuracy)`: 记录这个准确率。
            *   打印中间测试准确率。
            *   `model.train()`: **非常重要！** 将模型切换回训练模式，以便继续训练。

**(幻灯片/屏幕显示代码：第 168-172 行)**

```python
# ...existing code...
    epoch_avg_train_loss = running_loss / len(train_loader) # 计算当前轮次的平均训练损失
    train_losses.append(epoch_avg_train_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] complete. Average Training Loss (per batch): {epoch_avg_train_loss:.4f}')

print("训练完成.")
# ...existing code...
```

*   **解说：**
    *   在一个 epoch 的所有批次训练完成后：
    *   `epoch_avg_train_loss = running_loss / len(train_loader)`: 计算该 epoch 的平均训练损失（每个批次的平均损失）。
    *   `train_losses.append(epoch_avg_train_loss)`: 记录这个 epoch 的平均损失。
    *   打印该 epoch 完成的信息和平均训练损失。
    *   所有 epoch 训练完成后，打印 "训练完成."。

**第六部分：模型评估**

**(幻灯片/屏幕显示代码：第 175-193 行)**

```python
# ...existing code...
# 评估模型 - 在所有轮次训练完成后进行
print("开始评估...")
model.eval()
correct = 0
total = 0
num_final_test_batches = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.long().to(device)
        
        outputs = model(images)
        num_final_test_batches += 1
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total if total > 0 else 0

print(f'最终评估结果: Test Accuracy: {final_accuracy:.2f}%')
# ...existing code...
```

*   **解说：**
    *   在所有训练轮次完成后，对模型在整个测试集上进行最终评估。
    *   `print("开始评估...")`
    *   `model.eval()`: 将模型设置为评估模式。
    *   初始化 `correct` 和 `total` 计数器。
    *   `with torch.no_grad():`: 禁用梯度计算。
    *   `for images, labels in test_loader:`: 遍历测试数据加载器。
        *   数据移动到设备。
        *   `outputs = model(images)`: 获取预测。
        *   `_, predicted = torch.max(outputs.data, 1)`: 获取预测类别。
        *   累加 `total` 和 `correct`。
    *   `final_accuracy = 100 * correct / total`: 计算最终的测试准确率。
    *   打印最终测试准确率。

**第七部分：模型保存与绘图**

**(幻灯片/屏幕显示代码：第 195-197 行)**

```python
# ...existing code...
# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存.")
# ...existing code...
```

*   **解说：**
    *   `torch.save(model.state_dict(), 'mnist_cnn_model.pth')`: 保存训练好的模型的状态字典（包含所有学习到的参数）到文件 `mnist_cnn_model.pth`。`.pth` 是 PyTorch 模型常用的扩展名。
    *   打印模型已保存的提示。

**(幻灯片/屏幕显示代码：第 199-220 行)**

```python
# ...existing code...
# 绘制训练曲线
plt.figure(figsize=(12, 5)) # 调整图像大小以容纳两个子图

# 子图1: 每100步的训练损失
plt.subplot(1, 2, 1)
plt.plot(train_losses_step, label='每100步训练损失')
plt.xlabel('步骤 (x100)')
plt.ylabel('损失 (平均每批次)')
plt.title('每100步的训练损失')
plt.legend()
plt.grid(True)

# 子图2: 每100步的测试准确率
plt.subplot(1, 2, 2)
plt.plot(test_accuracies_step, label='每100步测试准确率', color='green')
plt.xlabel('步骤 (x100)')
plt.ylabel('准确率 (%)')
plt.title('每100步的测试准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_step_loss_accuracy.png') # 更新文件名
plt.show()
# ...existing code...
```

*   **解说：**
    *   使用 Matplotlib 绘制训练过程中的指标。
    *   `plt.figure(figsize=(12, 5))`: 创建一个图形窗口。
    *   `plt.subplot(1, 2, 1)`: 创建第一个子图，用于绘制“每100步的训练损失”。
        *   `plt.plot(train_losses_step, ...)`: 使用之前记录的 `train_losses_step` 数据绘图。
        *   设置标题、x轴标签、y轴标签、图例和网格。
    *   `plt.subplot(1, 2, 2)`: 创建第二个子图，用于绘制“每100步的测试准确率”。
        *   `plt.plot(test_accuracies_step, ...)`: 使用记录的 `test_accuracies_step` 数据绘图。
        *   设置相关标签和样式。
    *   `plt.tight_layout()`: 自动调整子图布局，防止重叠。
    *   `plt.savefig('training_step_loss_accuracy.png')`: 将图像保存到文件。
    *   `plt.show()`: 显示图像。

**第八部分：可视化预测结果**

**(幻灯片/屏幕显示代码：第 222-244 行)**

```python
# ...existing code...
# 可视化一些结果
def visualize_predictions():
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 获取预测结果
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(10):
        img = images[i].numpy().reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'predict: {predicted[i].item()}, actually: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

visualize_predictions()
# ...existing code...
```

*   **解说：**
    *   `visualize_predictions` 函数用于可视化模型对测试集中部分样本的预测。
    *   `model.eval()`: 设置为评估模式。
    *   `dataiter = iter(test_loader)` 和 `images, labels = next(dataiter)`: 从测试数据加载器中获取一个批次的图像和标签。
    *   `outputs = model(images)` 和 `_, predicted = torch.max(outputs, 1)`: 对这批图像进行预测。
    *   `fig, axes = plt.subplots(2, 5, ...)`: 创建 2x5 的子图网格，用于显示10张图片。
    *   `axes = axes.flatten()`: 展平子图数组，方便索引。
    *   循环显示前10张图像：
        *   `img = images[i].numpy().reshape(28, 28)`:
            *   `images[i]` 是一个 (1, 28, 28) 的张量。
            *   `.numpy()` 转换为 NumPy 数组。
            *   `.reshape(28, 28)` 去掉通道维度，变为 (28, 28) 以便 `imshow` 显示。如果图像在GPU上，需要先 `.cpu()`。
        *   `axes[i].imshow(img, cmap='gray')`: 显示灰度图像。
        *   `axes[i].set_title(...)`: 设置标题，显示预测值和真实标签。
        *   `axes[i].axis('off')`: 关闭坐标轴。
    *   `plt.tight_layout()`，`plt.savefig(...)`，`plt.show()`: 调整布局、保存并显示图像。
    *   最后，调用 `visualize_predictions()` 执行可视化。

**总结**

这个脚本全面地演示了如何使用 PyTorch 从头开始构建、训练、评估一个 CNN 模型，并对结果进行可视化。我们学习了自定义 Dataset、图像变换、CNN 层（卷积、批归一化、池化、全连接、Dropout）、训练循环中的中间验证、模型保存以及结果分析等重要概念。

希望这个详细的讲解能帮助大家更好地理解 CNN 的工作原理和 PyTorch 的使用。

---

希望这份文稿对您有所帮助！

找到具有 1 个许可证类型的类似代码
