**第一部分：环境准备与数据加载**

**(幻灯片/屏幕显示代码：第 1-7 行)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
```

*   **解说：**
    *   首先，我们导入了所有必需的库。
    *   `torch` 是 PyTorch 深度学习框架的核心库。
    *   `torch.nn` 模块包含了构建神经网络所需的各种层，比如线性层、LSTM 层等。
    *   `torch.optim` 模块提供了各种优化算法，如 Adam、SGD。
    *   `numpy` 是 Python 中用于科学计算的基础包，我们用它来处理数据。
    *   `matplotlib.pyplot` 用于数据可视化，比如绘制训练曲线和显示图像。
    *   `os` 模块提供了与操作系统交互的功能，比如检查文件是否存在。
    *   `struct` 模块用于处理二进制数据，MNIST 数据集就是以二进制格式存储的。

**(幻灯片/屏幕显示代码：第 9-13 行)**

```python
# ...existing code...
# 设置随机种子以确保结果可复现
torch.manual_seed(35)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ...existing code...
```

*   **解说：**
    *   `torch.manual_seed(35)`：这行代码设置了 PyTorch 的随机种子。在深度学习中，很多操作（如权重初始化、数据打乱）都带有随机性。设置固定的随机种子可以确保每次运行代码时，这些随机操作的结果都是一样的，这对于调试和结果复现非常重要。数字 `35` 是一个任意选择的种子值。
    *   `plt.rcParams['font.sans-serif'] = ['SimHei']` 和 `plt.rcParams['axes.unicode_minus'] = False`：这两行是 Matplotlib 的配置，用于确保在绘图时能够正确显示中文字符和负号。

**(幻灯片/屏幕显示代码：第 15 行)**

```python
# ...existing code...
# 设备配置：CPU计算
device = torch.device('cpu')
# ...existing code...
```

*   **解说：**
    *   `device = torch.device('cpu')`：这行代码指定了模型训练和数据计算所使用的设备。在这里，我们明确指定使用 CPU。如果你的计算机有支持 CUDA 的 NVIDIA GPU，并且安装了相应版本的 PyTorch，你可以将其改为 `torch.device('cuda')` 来利用 GPU 加速计算。对于这个实验，CPU 也是完全可以胜任的。

**(幻灯片/屏幕显示代码：第 18-28 行)**

```python
# ...existing code...
# 从本地文件加载MNIST数据集
def load_mnist(path, kind='train'):
    """从本地文件加载MNIST数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte') #构建完整路径
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte') 
    
    with open(labels_path, 'rb') as lbpath: #二进制读取标签文
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels
# ...existing code...
```

*   **解说：**
    *   这里定义了一个函数 `load_mnist`，专门用于从本地文件加载 MNIST 数据集。MNIST 数据集通常包含四个文件：训练图像、训练标签、测试图像和测试标签。这些文件是二进制格式的。
    *   `labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')` 和 `images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')`：这两行代码根据传入的 `path`（数据集所在的文件夹路径）和 `kind`（'train' 或 't10k'，代表测试集）构建标签文件和图像文件的完整路径。
    *   `with open(labels_path, 'rb') as lbpath:`：以二进制只读模式 (`'rb'`) 打开标签文件。`with` 语句确保文件在使用完毕后会被自动关闭。
    *   `magic, n = struct.unpack('>II', lbpath.read(8))`：MNIST 标签文件的前8个字节包含两个整数：一个魔数（magic number）和一个标签数量（n）。`struct.unpack('>II', ...)` 用于解析这两个大端序（'>'）的无符号整数（'I'）。
    *   `labels = np.fromfile(lbpath, dtype=np.uint8)`：从文件的剩余部分读取所有标签数据，数据类型为无符号8位整数 (`np.uint8`)。
    *   对于图像文件，操作类似：
        *   `magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))`：图像文件的前16个字节包含四个整数：魔数、图像数量（num）、每张图像的行数（rows，通常是28）和列数（cols，通常是28）。
        *   `images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)`：读取所有图像数据，并将其重塑为一个二维数组。每行代表一张展平的图像（28x28=784个像素点）。
    *   最后，函数返回加载的图像和标签。

**(幻灯片/屏幕显示代码：第 30-37 行)**

```python
# ...existing code...
# 加载数据集（从当前目录）
try:
    X_train, y_train = load_mnist(r'C:\Users\swuis\Desktop\asd\Algorithm Design', kind='train')
    X_test, y_test = load_mnist(r'C:\Users\swuis\Desktop\asd\Algorithm Design', kind='t10k')
    
    print(f'训练数据集: {X_train.shape}')
    print(f'测试数据集: {X_test.shape}')
except FileNotFoundError:
    print("未找到MNIST数据文件，请确保文件位于程序同一目录下")
    exit(1)
# ...existing code...
```

*   **解说：**
    *   这部分代码调用了我们刚刚定义的 `load_mnist` 函数来加载训练集和测试集。
    *   `r'C:\Users\swuis\Desktop\asd\Algorithm Design'` 是 MNIST 数据文件所在的路径。请确保这个路径与你存放数据文件的实际路径一致。`r''` 表示这是一个原始字符串，可以避免反斜杠的转义问题。
    *   `kind='train'` 加载训练数据，`kind='t10k'` 是 MNIST 测试集的标准文件名的一部分，用于加载测试数据。
    *   加载后，打印训练集和测试集的形状（shape），通常训练集是 (60000, 784)，测试集是 (10000, 784)。
    *   `try-except FileNotFoundError` 块用于捕获文件未找到的错误。如果指定路径下没有数据文件，程序会打印错误信息并退出。

**第二部分：数据预处理**

**(幻灯片/屏幕显示代码：第 39-45 行)**

```python
# ...existing code...
# 数据预处理：归一化并转换为PyTorch张量
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 转换为PyTorch张量
X_train = torch.from_numpy(X_train).reshape(-1, 28, 28)
X_test = torch.from_numpy(X_test).reshape(-1, 28, 28)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
# ...existing code...
```

*   **解说：**
    *   `X_train = X_train.astype(np.float32) / 255.0` 和 `X_test = X_test.astype(np.float32) / 255.0`：
        *   这两行对图像数据进行预处理。首先，`astype(np.float32)` 将图像数据的类型从原来的 `uint8`（0-255的整数）转换为 `float32`（32位浮点数）。
        *   然后，除以 `255.0` 将像素值从 [0, 255] 的范围归一化到 [0, 1] 的范围。数据归一化是神经网络训练中常用的技巧，有助于模型更快、更稳定地收敛。
    *   `X_train = torch.from_numpy(X_train).reshape(-1, 28, 28)`：
        *   `torch.from_numpy()` 将 NumPy 数组转换为 PyTorch 张量（Tensor）。PyTorch 模型直接操作的是张量。
        *   `.reshape(-1, 28, 28)` 将展平的图像数据（784个像素）重新塑形为 (N, 28, 28) 的三维张量，其中 N 是图像数量。对于 RNN/LSTM 模型，我们通常将图像的每一行（28个像素）视为一个时间步的输入，总共有28个时间步。
    *   `y_train = torch.from_numpy(y_train).long()` 和 `y_test = torch.from_numpy(y_test).long()`：
        *   同样将标签数据从 NumPy 数组转换为 PyTorch 张量。
        *   `.long()` 将标签的数据类型转换为长整型（`torch.int64`），这是 PyTorch 中分类任务（特别是使用 `CrossEntropyLoss` 时）通常要求的标签类型。

**(幻灯片/屏幕显示代码：第 47-48 行)**

```python
# ...existing code...
# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
# ...existing code...
```

*   **解说：**
    *   `torch.utils.data.TensorDataset(X_train, y_train)`：这个函数将图像张量和标签张量打包成一个数据集对象。`TensorDataset` 假设第一个维度是样本数量，它会按索引将图像和对应的标签配对。
    *   我们分别为训练数据和测试数据创建了 `TensorDataset` 对象。

**第三部分：超参数与数据加载器**

**(幻灯片/屏幕显示代码：第 50-57 行)**

```python
# ...existing code...
# 超参数
input_size = 28  # 输入大小（每行28个像素点）
hidden_size = 128  # 隐藏层大小
num_layers = 2  # LSTM层数
num_classes = 10  # 类别数（0-9）
batch_size = 64  # 批大小
num_epochs = 1  # 训练轮数
learning_rate = 0.001  # 学习率
# ...existing code...
```

*   **解说：**
    *   这部分定义了模型训练所需的一些超参数：
        *   `input_size = 28`：对于 RNN，我们将图像的每一行视为一个序列中的一个元素。每行有28个像素，所以输入特征的维度是28。
        *   `hidden_size = 128`：LSTM 隐藏层神经元的数量。这个值越大，模型的容量越大，但也可能更容易过拟合。
        *   `num_layers = 2`：LSTM 堆叠的层数。多层 LSTM 可以学习更复杂的模式。
        *   `num_classes = 10`：输出类别的数量，因为 MNIST 是识别0到9这10个数字。
        *   `batch_size = 64`：每次训练迭代中使用的样本数量。批处理可以提高训练效率并稳定梯度。
        *   `num_epochs = 1`：整个训练数据集将被遍历的次数。这里初始设置为1，后面会根据是否加载预训练模型来调整。
        *   `learning_rate = 0.001`：学习率控制模型参数更新的步长。

**(幻灯片/屏幕显示代码：第 59-60 行)**

```python
# ...existing code...
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# ...existing code...
```

*   **解说：**
    *   `torch.utils.data.DataLoader`：这是 PyTorch 中非常重要的数据加载工具。它接收一个 `Dataset` 对象（我们之前创建的 `train_dataset` 和 `test_dataset`），并提供按批次（`batch_size`）迭代数据的功能。
    *   `shuffle=True` (对于 `train_loader`)：表示在每个 epoch 开始时，都会打乱训练数据的顺序。这有助于模型学习，防止模型记住数据的特定顺序。
    *   `shuffle=False` (对于 `test_loader`)：测试时通常不需要打乱数据顺序。

**第四部分：定义 LSTM 模型**

**(幻灯片/屏幕显示代码：第 62-79 行)**

```python
# ...existing code...
# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # 我们只需要最后一个时间步的输出
        out = out[:, -1, :]  # out: (batch_size, hidden_size)
        out = self.fc(out)  # out: (batch_size, num_classes)
        return out
# ...existing code...
```

*   **解说：**
    *   我们定义了一个名为 `LSTM` 的类，它继承自 `nn.Module`，这是 PyTorch 中所有神经网络模块的基类。
    *   **`__init__` 方法 (构造函数):**
        *   `super(LSTM, self).__init__()`：调用父类 `nn.Module` 的构造函数，这是必须的。
        *   `self.hidden_size = hidden_size` 和 `self.num_layers = num_layers`：存储传入的超参数。
        *   `self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`：
            *   创建了一个 LSTM 层。
            *   `input_size`：输入序列中每个元素的特征维度（这里是28，代表图像每行的像素数）。
            *   `hidden_size`：LSTM 隐藏状态的特征维度（128）。
            *   `num_layers`：LSTM 的层数（2）。
            *   `batch_first=True`：这个参数非常重要。它指定输入和输出张量的第一个维度是批次大小 (batch size)。如果为 `False`（默认值），则第一个维度是序列长度。我们的数据是 (batch_size, seq_length, input_size) 即 (64, 28, 28)，所以 `batch_first=True` 是正确的。
        *   `self.fc = nn.Linear(hidden_size, num_classes)`：
            *   创建了一个全连接层（也叫线性层）。
            *   它的输入特征数是 LSTM 的 `hidden_size`，输出特征数是 `num_classes` (10)。这个层用于将 LSTM 的输出映射到最终的类别分数。
    *   **`forward` 方法:**
        *   这个方法定义了数据在网络中的前向传播路径。它接收输入 `x`。
        *   `h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)` 和 `c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)`：
            *   这两行初始化 LSTM 的初始隐藏状态 (`h0`) 和初始细胞状态 (`c0`)。
            *   它们的形状是 `(num_layers, batch_size, hidden_size)`。`x.size(0)` 获取当前批次的样本数量。
            *   `.to(device)` 将这些状态张量移动到之前定义的计算设备（CPU 或 GPU）上。
        *   `out, _ = self.lstm(x, (h0, c0))`：
            *   这是 LSTM 层的前向传播。它接收输入序列 `x` 和初始状态 `(h0, c0)`。
            *   它返回两个输出：`out` 和 `(hn, cn)`（这里用 `_` 忽略了后者）。
            *   `out` 张量包含了 LSTM 在每个时间步的输出。如果 `batch_first=True`，它的形状是 `(batch_size, seq_length, hidden_size)`。在这里，`seq_length` 是28（图像的行数）。
        *   `out = out[:, -1, :]`：
            *   对于序列分类任务，我们通常只关心 LSTM 处理完整个序列后的最终状态。
            *   `out[:, -1, :]` 提取了所有批次中，最后一个时间步 (`-1`) 的 LSTM 输出。此时 `out` 的形状变为 `(batch_size, hidden_size)`。
        *   `out = self.fc(out)`：
            *   将最后一个时间步的 LSTM 输出传递给全连接层，得到每个类别的分数。此时 `out` 的形状是 `(batch_size, num_classes)`。
        *   `return out`：返回最终的类别分数。

**第五部分：模型实例化、加载与训练准备**

**(幻灯片/屏幕显示代码：第 81-82 行)**

```python
# ...existing code...
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
continue_training = True
# ...existing code...
```

*   **解说：**
    *   `model = LSTM(input_size, hidden_size, num_layers, num_classes)`：实例化我们定义的 `LSTM` 模型，传入之前定义的超参数。
    *   `.to(device)`：将整个模型的所有参数和缓冲区移动到指定的计算设备上。
    *   `continue_training = True`：初始化一个标志，用于控制是否进行训练。

**(幻灯片/屏幕显示代码：第 84-99 行)**

```python
# ...existing code...
# 检查是否存在已保存的模型，如果有则加载
model_path = 'mnist_rnn_model.ckpt'
if os.path.exists(model_path):
    print(f"加载已训练的模型: {model_path}")
    model.load_state_dict(torch.load(model_path))
    user_input = input("是否继续训练？(y/n): ")
    if user_input.lower() == 'y':
        continue_training = True
        train_count = int(input("请输入继续训练的轮数: "))
        if train_count > 0:
            num_epochs = train_count
        else:
            print("输入的轮数无效，将跳过训练")
            continue_training = False
    else:
        continue_training = False
else:
    print("未找到已保存的模型，将从头开始训练")
# ...existing code...
```

*   **解说：**
    *   这部分代码实现了加载预训练模型的功能，并允许用户选择是否继续训练。
    *   `model_path = 'mnist_rnn_model.ckpt'`：定义了保存和加载模型参数的文件名。`.ckpt` 通常表示检查点文件。
    *   `if os.path.exists(model_path):`：检查当前目录下是否存在这个模型文件。
    *   如果存在：
        *   打印加载信息。
        *   `model.load_state_dict(torch.load(model_path))`：加载已保存的模型参数。`torch.load()` 读取文件内容，`model.load_state_dict()` 将这些参数加载到当前模型 `model` 中。
        *   询问用户是否继续训练。
        *   如果用户输入 'y'（不区分大小写），则设置 `continue_training = True`，并询问用户要继续训练多少轮 (`train_count`)。如果输入的轮数大于0，则更新 `num_epochs`。否则，提示无效并跳过训练。
        *   如果用户输入其他字符，则设置 `continue_training = False`，表示不进行训练，直接测试。
    *   如果模型文件不存在：
        *   打印信息，表示将从头开始训练。此时 `continue_training` 保持为 `True`（初始值），`num_epochs` 保持为之前设置的默认值（1）。

**(幻灯片/屏幕显示代码：第 102-103 行)**

```python
# ...existing code...
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# ...existing code...
```

*   **解说：**
    *   `criterion = nn.CrossEntropyLoss()`：定义损失函数。
        *   `CrossEntropyLoss` (交叉熵损失) 是用于多分类任务的常用损失函数。它内部结合了 `LogSoftmax` 和 `NLLLoss` (负对数似然损失)，所以我们的模型输出层不需要显式添加 Softmax 激活函数。
    *   `optimizer = optim.Adam(model.parameters(), lr=learning_rate)`：定义优化器。
        *   `optim.Adam` 是一种广泛使用的自适应学习率优化算法，通常效果较好。
        *   `model.parameters()`：将模型中所有需要训练的参数传递给优化器。
        *   `lr=learning_rate`：设置学习率。

**第六部分：模型训练**

**(幻灯片/屏幕显示代码：第 105-136 行)**

```python
# ...existing code...
# 如果需要继续训练
if continue_training:
    print(f"开始训练，预计{num_epochs}轮...")
    total_step = len(train_loader)
    
    # 用于存储每100步的训练过程中的损失和准确率
    train_losses_step = []
    train_accuracies_step = []
    
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        
        for i, (images, labels) in enumerate(train_loader):
            # 将图像传输到设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积当前周期的损失和准确率计算所需数据
            _, predicted_batch = torch.max(outputs.data, 1)
            batch_correct_predictions = (predicted_batch == labels).sum().item()
            
            if (i+1) % 100 == 0:
                current_batch_accuracy = 100 * batch_correct_predictions / labels.size(0)
                print(f'周期次 [{epoch+1}/{num_epochs}], 批次数 [{i+1}/{total_step}], 损失率: {loss.item():.4f}, 当前100批次准确率: {current_batch_accuracy:.2f}%')
                train_losses_step.append(loss.item())
                train_accuracies_step.append(current_batch_accuracy)
# ...existing code...
```

*   **解说：**
    *   `if continue_training:`：只有当 `continue_training` 标志为 `True` 时，才执行训练过程。
    *   `print(f"开始训练，预计{num_epochs}轮...")`：打印训练开始的信息。
    *   `total_step = len(train_loader)`：计算每个 epoch 中有多少个批次（batch）。
    *   `train_losses_step = []` 和 `train_accuracies_step = []`：初始化两个列表，用于存储训练过程中每100步的损失值和准确率，方便后续绘图。
    *   **外层循环 `for epoch in range(num_epochs):`**：遍历指定的训练轮数。
        *   `model.train()`：将模型设置为训练模式。这很重要，因为它会启用 Dropout 和 Batch Normalization 等在训练和评估时行为不同的层。
        *   **内层循环 `for i, (images, labels) in enumerate(train_loader):`**：遍历 `train_loader` 中的所有批次数据。`enumerate` 同时提供索引 `i` 和数据 `(images, labels)`。
            *   `images = images.to(device)` 和 `labels = labels.to(device)`：将当前批次的图像和标签数据移动到计算设备上。
            *   **前向传播 (Forward pass):**
                *   `outputs = model(images)`：将图像数据输入模型，得到模型的输出（类别分数）。
                *   `loss = criterion(outputs, labels)`：计算模型输出和真实标签之间的损失。
            *   **反向传播和优化 (Backward pass and optimize):**
                *   `optimizer.zero_grad()`：在计算梯度之前，清除之前累积的梯度。这是因为 PyTorch 的梯度是累加的。
                *   `loss.backward()`：根据当前的损失计算所有模型参数的梯度（反向传播）。
                *   `optimizer.step()`：优化器根据计算出的梯度更新模型的参数。
            *   **计算并记录指标:**
                *   `_, predicted_batch = torch.max(outputs.data, 1)`：
                    *   `torch.max(outputs.data, 1)` 会返回两个张量：一个是每行（每个样本）的最大值，另一个是最大值对应的索引。我们关心的是索引，因为它代表了模型预测的类别。`1` 表示在第二个维度（类别维度）上取最大值。
                    *   `_` 用来忽略最大值本身，我们只需要 `predicted_batch`（预测的类别）。
                *   `batch_correct_predictions = (predicted_batch == labels).sum().item()`：
                    *   `(predicted_batch == labels)` 比较预测类别和真实标签，得到一个布尔张量。
                    *   `.sum()` 计算其中 `True` 的数量（即预测正确的样本数）。
                    *   `.item()` 将只有一个元素的张量转换为 Python 数字。
                *   `if (i+1) % 100 == 0:`：每处理100个批次，就打印一次训练信息并记录损失和准确率。
                    *   `current_batch_accuracy = 100 * batch_correct_predictions / labels.size(0)`：计算当前这个批次的准确率。注意，这里是基于当前批次（`labels.size(0)` 个样本）的准确率，而不是整个 epoch 的。
                    *   打印周期、批次、损失和当前批次的准确率。
                    *   `train_losses_step.append(loss.item())` 和 `train_accuracies_step.append(current_batch_accuracy)`：将当前损失和准确率添加到列表中。

**(幻灯片/屏幕显示代码：第 138-139 行)**

```python
# ...existing code...
        
    # 保存继续训练后的模型
    torch.save(model.state_dict(), model_path)
    print(f"训练完成，模型已保存到: {model_path}")
# ...existing code...
```

*   **解说：**
    *   在所有训练轮次（epochs）完成后：
    *   `torch.save(model.state_dict(), model_path)`：将训练好的模型的状态字典（`model.state_dict()`，包含了模型的所有参数）保存到之前定义的 `model_path` 文件中。
    *   打印模型已保存的信息。

**第七部分：绘制训练曲线**

**(幻灯片/屏幕显示代码：第 141-162 行)**

```python
# ...existing code...
    # 绘制训练曲线
    steps_ran = range(1, len(train_losses_step) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps_ran, train_losses_step, label='每100步训练损失', marker='o')
    plt.title('每100步训练损失曲线')
    plt.xlabel('步骤 (x100)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps_ran, train_accuracies_step, label='每100步训练准确率', marker='o', color='r')
    plt.title('每100步训练准确率曲线')
    plt.xlabel('步骤 (x100)')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_training_step_metrics.png') # Save the new plot
    plt.show()
# ...existing code...
```

*   **解说：**
    *   如果进行了训练，这部分代码会使用 Matplotlib 绘制训练过程中的损失和准确率曲线。
    *   `steps_ran = range(1, len(train_losses_step) + 1)`：创建一个表示步骤（每100个batch为一个步骤）的序列，用于绘图的 x 轴。
    *   `plt.figure(figsize=(12, 5))`：创建一个新的图形窗口，并设置其大小。
    *   `plt.subplot(1, 2, 1)`：创建一个1行2列的子图布局，并选择第一个子图（左边的图）作为当前绘图区域。
    *   `plt.plot(steps_ran, train_losses_step, label='每100步训练损失', marker='o')`：绘制损失曲线。`steps_ran` 是 x 轴数据，`train_losses_step` 是 y 轴数据。`label` 用于图例，`marker='o'` 表示在每个数据点上画一个圆圈。
    *   `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`：设置图表的标题和坐标轴标签。
    *   `plt.legend()`：显示图例。
    *   `plt.grid(True)`：显示网格线。
    *   `plt.subplot(1, 2, 2)`：选择第二个子图（右边的图）。
    *   `plt.plot(steps_ran, train_accuracies_step, label='每100步训练准确率', marker='o', color='r')`：绘制准确率曲线，颜色设置为红色 (`color='r'`)。
    *   其余的 `title`, `xlabel`, `ylabel`, `legend`, `grid` 设置与损失曲线类似。
    *   `plt.tight_layout()`：自动调整子图参数，使之填充整个图像区域，避免标签重叠。
    *   `plt.savefig('rnn_training_step_metrics.png')`：将绘制的图形保存为 PNG 文件。
    *   `plt.show()`：显示图形。

**(幻灯片/屏幕显示代码：第 164-165 行)**

```python
# ...existing code...
else:
    print("跳过训练过程，直接进行测试...")
# ...existing code...
```

*   **解说：**
    *   这是 `if continue_training:` 的 `else` 分支。如果之前用户选择不继续训练，或者没有找到预训练模型但用户也选择不从头训练（虽然在这个脚本的逻辑中，后者情况较少，因为默认会从头训练），则会执行这里，打印一条跳过训练的消息。

**第八部分：模型测试**

**(幻灯片/屏幕显示代码：第 168-182 行)**

```python
# ...existing code...
# 测试模型
print("开始测试模型...")
model.eval()  # 设置为评估模式
with torch.no_grad():
    correct = 0 
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'测试准确率: {100 * correct / total:.2f}%')
# ...existing code...
```

*   **解说：**
    *   无论是否进行了训练，都会执行模型测试部分。
    *   `print("开始测试模型...")`：打印测试开始的信息。
    *   `model.eval()`：将模型设置为评估模式。这非常重要，因为它会关闭 Dropout 层，并让 Batch Normalization 层使用在训练时学习到的统计量，而不是当前批次的统计量。
    *   `with torch.no_grad():`：这是一个上下文管理器，它会禁用梯度计算。在测试或评估阶段，我们不需要计算梯度，这样做可以减少内存消耗并加速计算。
    *   `correct = 0` 和 `total = 0`：初始化计数器，用于统计预测正确的样本数和总样本数。
    *   **循环 `for images, labels in test_loader:`**：遍历 `test_loader` 中的所有测试数据批次。
        *   `images = images.to(device)` 和 `labels = labels.to(device)`：将数据移动到计算设备。
        *   `outputs = model(images)`：将测试图像输入模型，得到预测输出。
        *   `_, predicted = torch.max(outputs.data, 1)`：获取模型预测的类别，与训练时类似。
        *   `total += labels.size(0)`：累加当前批次的样本数量到总样本数。
        *   `correct += (predicted == labels).sum().item()`：累加当前批次中预测正确的样本数量。
    *   `print(f'测试准确率: {100 * correct / total:.2f}%')`：在所有测试数据上评估完毕后，计算并打印最终的测试准确率。

**第九部分：可视化预测结果**

**(幻灯片/屏幕显示代码：第 184-204 行)**

```python
# ...existing code...
# 可视化一些预测结果
def visualize_prediction():
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # 显示前10个图像
            for j in range(10):
                img = images[j].cpu().numpy().reshape(28, 28)
                axes[j].imshow(img, cmap='gray')
                axes[j].set_title(f'predict: {predicted[j].item()}\n actually: {labels[j].item()}')
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()
            break
# ...existing code...
```

*   **解说：**
    *   定义了一个函数 `visualize_prediction`，用于可视化模型对测试集中部分样本的预测结果。
    *   `fig, axes = plt.subplots(2, 5, figsize=(12, 5))`：创建一个包含 2行5列 共10个子图的图形窗口。`axes` 是一个包含所有子图对象的数组。
    *   `axes = axes.flatten()`：将 `axes` 数组展平为一维，方便按索引访问子图。
    *   `with torch.no_grad():`：同样，在预测时禁用梯度计算。
    *   `for i, (images, labels) in enumerate(test_loader):`：从测试数据加载器中取出一个批次的数据。
        *   数据移动到设备，模型进行预测。
        *   `for j in range(10):`：遍历这个批次中的前10张图像（假设批次大小至少为10）。
            *   `img = images[j].cpu().numpy().reshape(28, 28)`：
                *   `images[j]` 取出第 `j` 张图像。
                *   `.cpu()`：如果图像张量在 GPU 上，需要先将其移回 CPU，因为 NumPy 和 Matplotlib 通常在 CPU 上操作。
                *   `.numpy()`：将 PyTorch 张量转换为 NumPy 数组。
                *   `.reshape(28, 28)`：将图像数据从 (28, 28) 的序列形式（或者如果之前没有特别处理，可能是 (1, 28, 28) 或 (28, 28)）变回 28x28 的二维图像格式，以便 `imshow` 显示。
            *   `axes[j].imshow(img, cmap='gray')`：在第 `j` 个子图上显示图像。`cmap='gray'` 表示以灰度模式显示。
            *   `axes[j].set_title(f'predict: {predicted[j].item()}\n actually: {labels[j].item()}')`：设置子图的标题，显示模型的预测值和真实标签。
            *   `axes[j].axis('off')`：关闭坐标轴显示。
        *   `plt.tight_layout()`：调整子图布局。
        *   `plt.show()`：显示图像。
        *   `break`：只可视化第一个批次的前10个样本，然后退出循环。

**(幻灯片/屏幕显示代码：第 206-207 行)**

```python
# ...existing code...
print("可视化预测结果...")
visualize_prediction()
```

*   **解说：**
    *   打印提示信息，并调用 `visualize_prediction()` 函数来执行预测结果的可视化。

**总结**

这个脚本完整地展示了如何使用 PyTorch 构建、训练和评估一个 LSTM 模型来进行 MNIST 手写数字识别。我们学习了数据加载、预处理、模型定义、训练循环、模型保存与加载、评估以及结果可视化等关键步骤。

希望这个逐行讲解能帮助大家更好地理解代码的每一个细节。大家可以尝试修改超参数、网络结构，或者在其他数据集上应用类似的思路。

---

希望这份文稿对您有所帮助！

找到具有 2 个许可证类型的类似代码

qqwwee7453: 请生成一篇对CNN实验每一行代码进行详细讲解的视频文稿，适用于中国本科生深度学习选修课的实验作业水平

GitHub Copilot: 好的，这是一份针对你提供的 CNN 实验代码的详细视频文稿。

---

**视频文稿：CNN 实现 MNIST 数字识别详解**

**开场**

同学们好！今天我们将一起探索一个使用卷积神经网络（CNN）识别 MNIST 手写数字的 Python 实验。CNN 在图像识别领域取得了巨大成功，通过这个实验，我们将深入了解 CNN 的构建、训练以及如何在 PyTorch 中实现它。

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


找到具有 1 个许可证类型的类似代码
