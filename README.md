**(第一部分：导入必要的库)**

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

**解说：**

*   `import torch`: 这是 PyTorch 深度学习框架的核心库，提供了张量操作（类似 NumPy 的多维数组，但支持 GPU 加速）和自动求导等功能。
*   `import torch.nn as nn`: `nn` 模块包含了构建神经网络所需的各种层，比如卷积层、全连接层、激活函数等。
*   `import torch.optim as optim`: `optim` 模块提供了各种优化算法，如 SGD、Adam，用于在训练过程中更新网络的权重。
*   `import torch.nn.functional as F`: `F` 模块包含了一些函数式的操作，比如激活函数（ReLU）、池化操作等。与 `nn` 模块中的层不同，这些通常是无状态的。
*   `import numpy as np`: NumPy 是 Python 中用于科学计算的基础库，主要用于处理数组和矩阵。我们用它来处理从文件中读取的数据。
*   `import matplotlib.pyplot as plt`: Matplotlib 是一个绘图库，我们将用它来可视化训练过程中的损失和准确率，以及展示预测结果。
*   `import os`: `os` 模块提供了与操作系统交互的功能，比如文件路径操作。
*   `import struct`: `struct` 模块用于处理二进制数据，MNIST 数据集是以二进制格式存储的。
*   `from torchvision import transforms`: `torchvision` 是 PyTorch 的一个计算机视觉工具包，`transforms` 模块提供了常用的图像预处理方法。
*   `from torch.utils.data import Dataset, DataLoader`: `Dataset` 和 `DataLoader` 是 PyTorch 中用于加载和处理数据的核心工具。`Dataset` 用于封装数据集，`DataLoader` 用于按批次加载数据。

**(第二部分：初始化设置)**

```python
# ...existing code...
# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 指定使用CPU
device = torch.device('cpu')
print(f"使用设备: {device}")
# ...existing code...
```

**解说：**

*   `torch.manual_seed(42)`: 设置随机种子。在神经网络中，权重的初始化、数据的打乱等操作都带有随机性。设置随机种子可以确保每次运行代码时，这些随机操作的结果都是一样的，从而使得实验结果可以复现。`42` 只是一个常用的种子数值，你可以选择其他整数。
*   `device = torch.device('cpu')`: 这行代码指定了模型训练和数据计算所使用的设备。这里我们明确指定使用 CPU。如果你的电脑有支持 CUDA 的 NVIDIA 显卡，并且安装了相应版本的 PyTorch，你可以将其改为 `torch.device('cuda')` 来使用 GPU 加速训练，通常会快很多。
*   `print(f"使用设备: {device}")`: 打印出当前使用的设备信息。

**(第三部分：从本地文件加载 MNIST 数据集)**

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
        images = images.reshape(len(labels), 1, 28, 28)  # 重塑为CNN输入格式
    
    return images, labels
# ...existing code...
```

**解说：**

*   `def load_mnist(path, kind='train'):`: 定义了一个函数 `load_mnist`，用于从本地加载 MNIST 数据。
    *   `path`: 数据集文件所在的目录路径。
    *   `kind`: `'train'` 表示加载训练集，`'t10k'` 表示加载测试集。
*   `labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')`: 使用 `os.path.join` 构建标签文件的完整路径。MNIST 的标签文件名是固定的格式。
*   `images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')`: 构建图像文件的完整路径。
*   `with open(labels_path, 'rb') as lbpath:`: 以二进制只读模式 (`'rb'`) 打开标签文件。`with` 语句确保文件在使用完毕后会被正确关闭。
    *   `magic, n = struct.unpack('>II', lbpath.read(8))`: MNIST 文件头部包含一些元信息。`lbpath.read(8)` 读取文件的前8个字节。`struct.unpack('>II', ...)` 将这8个字节按照大端序（`>`）解析为两个无符号整数（`I`）。第一个是魔数（magic number），第二个是标签数量（`n`）。
    *   `labels = np.fromfile(lbpath, dtype=np.uint8)`: 从文件中剩下的部分读取标签数据，数据类型为无符号8位整数 (`np.uint8`)。
*   `with open(images_path, 'rb') as imgpath:`: 以二进制只读模式打开图像文件。
    *   `magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))`: 读取图像文件的前16个字节，解析为魔数、图像数量（`num`）、图像行数（`rows`，即28）和图像列数（`cols`，即28）。
    *   `images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)`: 读取图像数据，并将其重塑为一个二维数组。每张图像是 28x28=784 个像素。`len(labels)` 确保图像数量与标签数量一致。
    *   `images = images.reshape(len(labels), 1, 28, 28)`: 将图像数据重塑为 CNN 所需的输入格式：`(数量, 通道数, 高度, 宽度)`。MNIST 是灰度图，所以通道数是 1。
*   `return images, labels`: 返回加载的图像和标签数据。

**(第四部分：自定义 MNIST 数据集类)**

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

**解说：**

*   `class MnistDataset(Dataset):`: 我们定义一个自定义的数据集类 `MnistDataset`，它继承自 PyTorch 的 `torch.utils.data.Dataset`。自定义数据集类需要实现三个核心方法：`__init__`、`__len__` 和 `__getitem__`。
*   `def __init__(self, images, labels, transform=None):`: 构造函数。
    *   `self.images = images`: 存储图像数据。
    *   `self.labels = labels`: 存储标签数据。
    *   `self.transform = transform`: 存储数据预处理的操作。`transform` 是一个可选参数，可以传入一系列图像变换操作。
*   `def __len__(self):`: 这个方法需要返回数据集的总样本数量。这里我们返回标签的数量。
*   `def __getitem__(self, idx):`: 这个方法根据给定的索引 `idx` 返回一个数据样本（图像和对应的标签）。
    *   `image = self.images[idx]`: 获取指定索引的图像。
    *   `label = self.labels[idx]`: 获取指定索引的标签。
    *   `if self.transform:`: 检查是否定义了 `transform`。
        *   `image = self.transform(image)`: 如果定义了，就对图像应用预处理操作。
    *   `return image, label`: 返回处理后的图像和标签。

**(第五部分：加载数据集和数据预处理)**

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

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])
# ...existing code...
```

**解说：**

*   `try...except FileNotFoundError`: 这是一个错误处理块。
    *   `current_dir = r"C:\Users\swuis\Desktop\asd\Algorithm Design"`: 指定 MNIST 数据文件所在的目录。**请注意：** 这个路径是硬编码的，你需要根据你实际存放数据文件的位置修改它。
    *   `X_train, y_train = load_mnist(current_dir, kind='train')`: 调用 `load_mnist` 函数加载训练数据。
    *   `X_test, y_test = load_mnist(current_dir, kind='t10k')`: 加载测试数据。
    *   `print(f'训练数据集: {X_train.shape}')`: 打印训练数据的形状，方便检查。
    *   `print(f'测试数据集: {X_test.shape}')`: 打印测试数据的形状。
    *   `except FileNotFoundError:`: 如果在指定路径下找不到数据文件，会捕获 `FileNotFoundError` 异常。
    *   `print("未找到MNIST数据文件，请确保文件位于指定目录下")`: 打印错误提示。
    *   `exit(1)`: 退出程序。
*   `transform = transforms.Compose([...])`: 定义了一系列数据预处理操作。`transforms.Compose` 可以将多个 `transform` 操作串联起来。
    *   `transforms.ToTensor()`: 这个转换非常重要。它会将 PIL 图像或 NumPy `ndarray`（形状为 H x W x C，取值范围为 [0, 255]）转换为 PyTorch 张量（形状为 C x H x W，取值范围为 [0.0, 1.0]）。对于我们的 NumPy 数组（已经是 C x H x W，但类型是 uint8），它会将其转换为 float32 类型的张量，并进行归一化到 [0.0, 1.0]（如果原始数据是 uint8）。
    *   `transforms.Normalize((0.1307,), (0.3081,))`: 对张量进行标准化处理。公式是 `output = (input - mean) / std`。`(0.1307,)` 是 MNIST 数据集在所有像素上的平均值，`(0.3081,)` 是标准差。注意这里的元组形式，因为我们只有一个通道（灰度图）。标准化有助于模型更快、更稳定地收敛。

**(第六部分：创建数据集实例)**

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

**解说：**

*   `train_dataset = MnistDataset(...)`: 创建训练数据集的实例。
    *   `images=X_train.astype(np.float32) / 255.0`: 传入训练图像。这里我们先将图像数据从 `uint8` 类型转换为 `float32` 类型，然后除以 255.0，将像素值从 [0, 255] 缩放到 [0.0, 1.0] 之间。这一步是手动的归一化，因为 `transforms.ToTensor()` 对于已经是 NumPy 数组的输入，如果不是 `uint8` 类型，可能不会自动缩放到 [0,1]。虽然 `transforms.ToTensor()` 对 `uint8` 的 NumPy 数组会做这个缩放，但这里显式操作更清晰，并且确保了在 `transforms.ToTensor()` 之前数据是浮点型且在 [0,1] 范围内。
    *   `labels=y_train`: 传入训练标签。
    *   `transform=transform`: 传入我们之前定义的预处理操作。
*   `test_dataset = MnistDataset(...)`: 类似地，创建测试数据集的实例，使用测试图像和标签，并应用相同的 `transform`。

**(第七部分：创建数据加载器)**

```python
# ...existing code...
# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# ...existing code...
```

**解说：**

*   `batch_size = 64`: 设置每个批次加载的样本数量。批处理训练是深度学习中的标准做法，可以提高训练效率和稳定性。
*   `train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)`: 创建训练数据加载器。
    *   `train_dataset`: 我们之前创建的训练数据集实例。
    *   `batch_size=batch_size`: 指定批次大小。
    *   `shuffle=True`: 在每个 epoch 开始时，打乱训练数据的顺序。这有助于防止模型过拟合，提高模型的泛化能力。
*   `test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)`: 创建测试数据加载器。
    *   `shuffle=False`: 对于测试集，通常不需要打乱顺序，因为评估结果与数据顺序无关。

**(第八部分：定义 CNN 模型)**

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

**解说：**

*   `class ConvNet(nn.Module):`: 定义一个名为 `ConvNet` 的类，它继承自 `nn.Module`。这是 PyTorch 中定义所有神经网络模型的基类。
*   `def __init__(self):`: 构造函数，在这里定义网络的各个层。
    *   `super(ConvNet, self).__init__()`: 调用父类 `nn.Module` 的构造函数，这是必须的。
    *   **第一个卷积块**:
        *   `self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)`: 定义第一个卷积层。
            *   `1`: 输入通道数 (灰度图为1)。
            *   `32`: 输出通道数 (即卷积核的数量，也代表提取的特征图数量)。
            *   `kernel_size=3`: 卷积核大小为 3x3。
            *   `stride=1`: 卷积核滑动的步长为1。
            *   `padding=1`: 在输入图像周围填充1圈0。当 `kernel_size=3, padding=1` 时，卷积操作后图像尺寸不变 (对于 `stride=1`)。
        *   `self.bn1 = nn.BatchNorm2d(32)`: 定义一个批量归一化层 (Batch Normalization)。它对卷积层的输出进行归一化，可以加速训练，提高模型的稳定性和泛化能力。参数 `32` 是输入的通道数，与 `conv1` 的输出通道数一致。
        *   `self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)`: 定义一个最大池化层。
            *   `kernel_size=2`: 池化窗口大小为 2x2。
            *   `stride=2`: 池化窗口滑动的步长为2。这将使特征图的尺寸减半 (例如，28x28 变为 14x14)。池化层可以减少计算量，提取主要特征，并具有一定的平移不变性。
    *   **第二个卷积块**:
        *   `self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)`: 定义第二个卷积层。输入通道数为 `32` (来自 `conv1` 的输出)，输出通道数为 `64`。其他参数类似。
        *   `self.bn2 = nn.BatchNorm2d(64)`: 对应的批量归一化层，输入通道数为 `64`。
        *   `self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)`: 第二个最大池化层。如果输入是 14x14，输出将是 7x7。
    *   **全连接层**:
        *   `self.fc1 = nn.Linear(64 * 7 * 7, 128)`: 定义第一个全连接层 (也叫线性层)。
            *   `64 * 7 * 7`: 输入特征的数量。经过两个池化层后，特征图的尺寸变为 7x7，并且有 `64` 个通道，所以总特征数是 `64 * 7 * 7 = 3136`。
            *   `128`: 输出特征的数量 (即该层神经元的数量)。
        *   `self.dropout = nn.Dropout(0.3)`: 定义一个 Dropout 层。Dropout 是一种正则化技术，在训练过程中以一定的概率 (`0.3` 即 30%) 随机将一些神经元的输出置为0，可以有效防止过拟合。
        *   `self.fc2 = nn.Linear(128, 10)`: 定义第二个全连接层，也是输出层。
            *   `128`: 输入特征数 (来自 `fc1` 的输出)。
            *   `10`: 输出特征数，对应 MNIST 数据集的10个类别 (数字0到9)。
*   `def forward(self, x):`: 定义模型的前向传播逻辑。输入 `x` 是一个批次的图像数据。
    *   `x = self.pool1(F.relu(self.bn1(self.conv1(x))))`: 数据流经第一个卷积块。
        *   `self.conv1(x)`: 通过第一个卷积层。
        *   `self.bn1(...)`: 通过批量归一化层。
        *   `F.relu(...)`: 应用 ReLU (Rectified Linear Unit) 激活函数。ReLU 函数的形式是 `max(0, z)`，它是一种常用的非线性激活函数，可以引入非线性，并有助于缓解梯度消失问题。
        *   `self.pool1(...)`: 通过第一个最大池化层。
    *   `x = self.pool2(F.relu(self.bn2(self.conv2(x))))`: 数据流经第二个卷积块，操作类似。
    *   `x = x.view(-1, 64 * 7 * 7)`: 展平操作 (Flatten)。将池化层输出的多维特征图转换为一维向量，以便输入到全连接层。
        *   `x.view(...)` 是 PyTorch 中用于改变张量形状的方法。
        *   `-1` 表示该维度的大小由其他维度和总元素数量自动推断。这里是将每个样本的 `64x7x7` 特征图展平成一个长度为 `3136` 的向量。
    *   `x = F.relu(self.fc1(x))`: 数据通过第一个全连接层，并应用 ReLU 激活函数。
    *   `x = self.dropout(x)`: 应用 Dropout。注意 Dropout 通常只在训练时起作用，在评估时会自动关闭。
    *   `x = self.fc2(x)`: 数据通过第二个全连接层 (输出层)。这一层的输出是每个类别的原始得分 (logits)，没有经过 softmax。
    *   `return x`: 返回模型的原始输出。

**(第九部分：实例化模型、定义损失函数和优化器)**

```python
# ...existing code...
# 实例化模型
model = ConvNet().to(device)
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# ...existing code...
```

**解说：**

*   `model = ConvNet().to(device)`: 创建 `ConvNet` 类的一个实例，即我们的 CNN 模型。`.to(device)` 将模型的所有参数和缓冲区移动到之前定义的 `device` (CPU 或 GPU) 上。
*   `print(model)`: 打印模型结构，可以看到每一层的详细信息。
*   `criterion = nn.CrossEntropyLoss()`: 定义损失函数。对于多分类问题，交叉熵损失函数 (`CrossEntropyLoss`) 是非常常用的。PyTorch 的 `nn.CrossEntropyLoss` 内部会自动对模型的原始输出 (logits) 应用 Softmax 函数，并计算交叉熵损失。
*   `optimizer = optim.Adam(model.parameters(), lr=0.001)`: 定义优化器。
    *   `optim.Adam`: 使用 Adam 优化算法。Adam 是一种自适应学习率的优化算法，通常效果较好且收敛较快。
    *   `model.parameters()`: 将模型中所有需要训练的参数传递给优化器。
    *   `lr=0.001`: 设置学习率 (learning rate) 为 0.001。学习率控制了模型参数在每次更新时的步长。

**(第十部分：训练模型)**

```python
# ...existing code...
# 训练模型
num_epochs = 5  # 减少轮次以加快CPU训练速度
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到设备
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
# ...existing code...
```

**解说：**

*   `num_epochs = 5`: 设置训练的总轮次 (epoch)。一个 epoch 表示模型完整地遍历一次所有训练数据。这里设置为5轮，是为了在 CPU 上能较快看到结果。
*   `train_losses = []`, `test_losses = []`, `test_accuracies = []`: 创建空列表，用于存储每个 epoch 的训练损失、测试损失和测试准确率，方便后续可视化。
*   `for epoch in range(num_epochs):`: 外层循环，遍历每个 epoch。
    *   `model.train()`: 将模型设置为训练模式。这很重要，因为像 Dropout 和 BatchNorm 这样的层在训练和评估模式下的行为是不同的。例如，Dropout 只在训练时激活。
    *   `running_loss = 0.0`: 初始化当前 epoch 的累计损失。
    *   `for i, (images, labels) in enumerate(train_loader):`: 内层循环，遍历 `train_loader` 中的每个批次数据。`enumerate` 同时提供索引 `i` 和数据 `(images, labels)`。
        *   `images = images.to(device)`: 将图像数据移动到指定设备。
        *   `labels = torch.tensor(labels, dtype=torch.long).to(device)`: 将标签数据转换为 `torch.long` 类型的张量，并移动到指定设备。`CrossEntropyLoss`期望的标签类型是 `long`。
        *   `optimizer.zero_grad()`: 清除先前计算的梯度。在每次反向传播之前，都需要执行此操作，否则梯度会累积。
        *   `outputs = model(images)`: 前向传播，将图像输入模型，得到预测输出 (logits)。
        *   `loss = criterion(outputs, labels)`: 计算当前批次的损失。
        *   `loss.backward()`: 反向传播，计算损失相对于模型参数的梯度。
        *   `optimizer.step()`: 参数更新。优化器根据计算出的梯度和定义的优化算法 (Adam) 来更新模型的权重。
        *   `running_loss += loss.item()`: 累加当前批次的损失。`loss.item()` 获取损失张量中的标量值。
        *   `if (i+1) % 100 == 0:`: 每处理100个批次，打印一次训练信息，包括当前 epoch、步数和当前批次的损失。
    *   `epoch_loss = running_loss / len(train_loader)`: 计算当前 epoch 的平均训练损失。
    *   `train_losses.append(epoch_loss)`: 将平均训练损失添加到列表中。

**(第十一部分：评估模型)**

```python
# ...existing code...
    # 评估模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Test Loss: {test_loss:.4f}, '
          f'Test Accuracy: {accuracy:.2f}%')
# ...existing code...
```

**解说：**

*   在每个 epoch 训练结束后，我们会对模型在测试集上进行评估。
*   `model.eval()`: 将模型设置为评估模式。这会关闭 Dropout，并让 BatchNorm 使用运行时的统计数据而不是当前批次的统计数据。
*   `test_loss = 0.0`, `correct = 0`, `total = 0`: 初始化测试损失、正确预测的数量和总样本数。
*   `with torch.no_grad():`: 这是一个上下文管理器，在其作用域内，所有计算都不会追踪梯度。这在评估阶段是必要的，因为我们不需要计算梯度，可以节省内存和计算资源。
    *   `for images, labels in test_loader:`: 遍历测试数据加载器中的每个批次。
        *   `images = images.to(device)` 和 `labels = torch.tensor(labels, dtype=torch.long).to(device)`: 将数据移动到设备。
        *   `outputs = model(images)`: 前向传播，获取模型预测。
        *   `loss = criterion(outputs, labels)`: 计算测试损失。
        *   `test_loss += loss.item()`: 累加测试损失。
        *   `_, predicted = torch.max(outputs.data, 1)`: 获取预测结果。`torch.max(outputs.data, 1)` 会返回两个张量：第一个是每行中的最大值 (即最大概率对应的logit)，第二个是最大值对应的索引 (即预测的类别)。我们只需要索引，所以用 `_` 忽略第一个返回值。`dim=1` 表示在第二个维度 (类别维度) 上取最大值。
        *   `total += labels.size(0)`: 累加当前批次的样本数量。
        *   `correct += (predicted == labels).sum().item()`: 比较预测标签 `predicted` 和真实标签 `labels`。`(predicted == labels)` 会生成一个布尔张量，`.sum()` 计算其中 `True` (即预测正确) 的数量，`.item()` 将结果转换为 Python 数字。
*   `test_loss /= len(test_loader)`: 计算平均测试损失。
*   `test_losses.append(test_loss)`: 将平均测试损失添加到列表中。
*   `accuracy = 100 * correct / total`: 计算测试准确率，并乘以100转换为百分比。
*   `test_accuracies.append(accuracy)`: 将测试准确率添加到列表中。
*   `print(...)`: 打印当前 epoch 的训练损失、测试损失和测试准确率。

**(第十二部分：保存模型)**

```python
# ...existing code...
# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存.")
# ...existing code...
```

**解说：**

*   `torch.save(model.state_dict(), 'mnist_cnn_model.pth')`: 保存训练好的模型。
    *   `model.state_dict()`: 返回一个包含模型所有可学习参数 (权重和偏置) 的字典。这是一种推荐的保存模型的方式，因为它只保存参数，不保存模型结构，比较灵活。
    *   `'mnist_cnn_model.pth'`: 保存模型的文件名。`.pth` 是 PyTorch 常用的模型文件扩展名。
*   `print("模型已保存.")`: 打印保存成功的提示。

**(第十三部分：绘制训练曲线)**

```python
# ...existing code...
# 绘制训练曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率 (%)')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
# ...existing code...
```

**解说：**

*   这部分代码使用 `matplotlib` 来可视化训练过程。
*   `plt.figure(figsize=(10, 4))`: 创建一个新的图形窗口，设置其大小为宽10英寸，高4英寸。
*   `plt.subplot(1, 2, 1)`: 创建一个1行2列的子图网格，并激活第一个子图 (左边的图)。
    *   `plt.plot(train_losses, label='训练损失')`: 绘制训练损失曲线。`train_losses` 是 x 轴为轮次 (隐式)，y 轴为损失值。
    *   `plt.plot(test_losses, label='测试损失')`: 在同一子图上绘制测试损失曲线。
    *   `plt.xlabel('轮次')`: 设置 x 轴标签。
    *   `plt.ylabel('损失')`: 设置 y 轴标签。
    *   `plt.legend()`: 显示图例 (即 '训练损失' 和 '测试损失' 的标签)。
*   `plt.subplot(1, 2, 2)`: 激活第二个子图 (右边的图)。
    *   `plt.plot(test_accuracies, label='测试准确率')`: 绘制测试准确率曲线。
    *   `plt.xlabel('轮次')`: 设置 x 轴标签。
    *   `plt.ylabel('准确率 (%)')`: 设置 y 轴标签。
    *   `plt.legend()`: 显示图例。
*   `plt.tight_layout()`: 自动调整子图参数，使其填充整个图像区域，避免标签重叠。
*   `plt.savefig('training_curves.png')`: 将绘制的图形保存为名为 training_curves.png 的图片文件。
*   `plt.show()`: 显示图形窗口。

**(第十四部分：可视化一些预测结果)**

```python
# ...existing code...
# 可视化一些结果
def visualize_predictions():
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 获取预测结果
    outputs = model(images.to(device)) # 确保图像在正确的设备上
    _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(10):
        if i >= len(images): break # 防止索引越界
        img = images[i].cpu().numpy().reshape(28, 28) # 将图像移回CPU并转换为numpy
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'预测: {predicted[i].item()}, 实际: {labels[i].item()}') # .item() 获取标量值
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

visualize_predictions()
```

**解说：**

*   `def visualize_predictions():`: 定义一个函数来可视化模型的预测结果。
*   `model.eval()`: 再次确保模型处于评估模式。
*   `dataiter = iter(test_loader)`: 获取测试数据加载器的迭代器。
*   `images, labels = next(dataiter)`: 从迭代器中获取一个批次的测试数据。
*   `outputs = model(images.to(device))`: 将这批图像输入模型进行预测。注意，如果 `images` 不在 `device` 上，需要先移动过去。
*   `_, predicted = torch.max(outputs, 1)`: 获取预测的类别。
*   `fig, axes = plt.subplots(2, 5, figsize=(12, 5))`: 创建一个 2x5 的子图网格，用于显示10张图片。
*   `axes = axes.flatten()`: 将 2x5 的 `axes` 数组展平成一维数组，方便索引。
*   `for i in range(10):`: 循环显示前10张图片（或当前批次中可用的图片数量）。
    *   `if i >= len(images): break`: 添加一个检查，如果批次大小小于10，防止索引越界。
    *   `img = images[i].cpu().numpy().reshape(28, 28)`:
        *   `images[i]`: 获取第 `i` 张图像张量。
        *   `.cpu()`: 如果张量在 GPU 上，将其移回 CPU，因为 NumPy 和 Matplotlib 通常在 CPU 上操作。
        *   `.numpy()`: 将 PyTorch 张量转换为 NumPy 数组。
        *   `.reshape(28, 28)`: 将图像从 (1, 28, 28) 或 (28, 28) 重塑为 28x28，以便 `imshow` 显示。如果 `transform` 中的 `ToTensor` 已经将图像变成了 C x H x W，那么这里可能需要 `images[i].squeeze().cpu().numpy()` 来移除通道维度（如果通道为1）。不过，由于我们之前在 `MnistDataset` 中加载的 `X_test` 是 `(N, 1, 28, 28)`，并且 `ToTensor` 会处理它，所以 `images[i]` 应该是 `(1, 28, 28)`。`reshape(28,28)` 适用于这种情况，或者 `squeeze()` 后再 `reshape`。为了安全，通常 `images[i][0].cpu().numpy()` 或者 `images[i].squeeze().cpu().numpy()` 更鲁棒。鉴于代码中直接 `reshape(28,28)`，我们假设 `images[i]` 此时可以被这样重塑。
    *   `axes[i].imshow(img, cmap='gray')`: 使用 `imshow` 显示灰度图像。`cmap='gray'` 指定使用灰度颜色映射。
    *   `axes[i].set_title(f'预测: {predicted[i].item()}, 实际: {labels[i].item()}')`: 设置子图的标题，显示模型的预测值和真实标签。`.item()` 用于从单元素张量中获取 Python 数字。
    *   `axes[i].axis('off')`: 关闭坐标轴显示。
*   `plt.tight_layout()`: 调整布局。
*   `plt.savefig('prediction_examples.png')`: 保存预测示例图片。
*   `plt.show()`: 显示图片。
*   `visualize_predictions()`: 调用该函数执行可视化。

**(结尾)**

好了，以上就是对这个 CNN 实现 MNIST 数字识别实验代码的逐行详细讲解。通过这个实验，我们学习了如何加载和预处理数据，如何定义一个卷积神经网络模型，如何训练和评估模型，以及如何可视化结果。希望对大家理解 CNN 的工作原理和 PyTorch 的使用有所帮助！

---

希望这份视频文稿能够满足您的要求！

找到具有 1 个许可证类型的类似代码
