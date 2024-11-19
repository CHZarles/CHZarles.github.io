# 常见的算子

## matmul

```python

# test matmul
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

result = torch.matmul(matrix1, matrix2)

expected_result = torch.tensor([[19, 22], [43, 50]])

# Test if the result is as expected
assert torch.equal(result, expected_result), f"Test failed: {result} != {expected_result}"



```

## sigmoid

Sigmoid 是一种常用的激活函数，特别是在神经网络中。它的数学表达式为：

\\[ \sigma(x) = \frac{1}{1 + e^{-x}} \\]

其中 \\( e \\) 是自然对数的底数。

```python

import torch.nn.functional as F

# Define a tensor
input_tensor = torch.tensor([0.0, 2.0, -2.0])

# Apply sigmoid function
result = torch.sigmoid(input_tensor)

# Expected result
expected_result = torch.tensor([0.5, 0.8808, 0.1192])

# Test if the result is as expected
assert torch.allclose(result, expected_result, atol=1e-4), f"Test failed: {result} != {expected_result}"

```

## softmax

Softmax 是一种常用的激活函数，特别是在多分类问题中。它将一个向量中的每个元素转换为 0 到 1 之间的概率值，并且这些概率值的总和为 1。Softmax 函数的数学表达式为：

\\[ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} \\]

其中 \\( x_i \\) 是输入向量中的第 \\( i \\) 个元素， \\( e \\) 是自然对数的底数。

```python

import torch.nn.functional as F

# Define a tensor
input_tensor = torch.tensor([1.0, 2.0, 3.0])

# Apply softmax function
result = F.softmax(input_tensor, dim=0)

# Expected result
expected_result = torch.tensor([0.0900, 0.2447, 0.6652])

# Test if the result is as expected
assert torch.allclose(result, expected_result, atol=1e-4), f"Test failed: {result} != {expected_result}"

```

## linear

线性变换（Linear Transformation）是线性代数中的一个基本概念，它可以用来描述向量空间中的变换。在线性代换中，输入向量通过一个矩阵变换为输出向量。这个过程可以用矩阵乘法来表示。

### 方程组表示

假设我们有一个线性方程组：

\\[ y_1 = a_{11}x_1 + a_{12}x_2 + \\cdots + a_{1n}x_n \\]

\\[ y_2 = a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\]

\\[ \vdots \\]

\\[ y_m = a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n \\]

其中，\\( x_1, x_2, \\ldots, x_n \\) 是输入变量，\\( y_1, y_2, \\ldots, y_m \\) 是输出变量，\\( a\_{ij} \\) 是系数。

### 矩阵表示

上述方程组可以用矩阵乘法来表示：

\\[ \mathbf{y} = \mathbf{A} \mathbf{x} \\]

其中：

- \\(\mathbf{y}\\) 是输出向量，形状为 \\( m \times 1 \\)
- \\(\mathbf{A}\\) 是系数矩阵，形状为 \\( m \times n \\)
- \\(\mathbf{x}\\) 是输入向量，形状为 \\( n \times 1 \\)

具体来说：

```
| y1 |     | a11 a12 ... a1n |   | x1 |
| y2 |     | a21 a22 ... a2n |   | x2 |
| .. |  =  | ... ... ... ... | * | .. |
| ym |     | am1 am2 ... amn |   | xn |
```

### 示例

在 PyTorch 中，可以使用 `torch.nn.Linear` 来实现线性变换。以下是一个简单的示例：

\\[ y_1 = 0.1 \cdot x + 0.2 \cdot y + 0.1 \\]

\\[y_2 = 0.3 \cdot x + 0.4 \cdot y + 0.2 \\]

\\[y_3 = 0.5 \cdot x + 0.6 \cdot y + 0.3 \\]

令 \\[x = 1， y = 2 \\]. 其实现代码如下

```python
import torch
import torch.nn as nn

| y1 |     | 0.1  0.2 |   | 1.0 |   | 0.1 |
| y2 |  =  | 0.3  0.4 | * |     | + | 0.2 |
| y3 |     | 0.5  0.6 |   | 2.0 |   | 0.3 |

# 定义输入向量
input_tensor = torch.tensor([1.0, 2.0])

# 定义线性层
linear_layer = nn.Linear(2, 3)  # 输入维度为2，输出维度为3

# 初始化权重和偏置
linear_layer.weight = nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
linear_layer.bias = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

# 进行线性变换
output_tensor = linear_layer(input_tensor)

print(output_tensor)


#out => tensor([0.6000, 1.3000, 2.0000], grad_fn=<ViewBackward0>)
```

## relu

`ReLU` 是 Rectified Linear Unit 的缩写，是一种常用的激活函数，广泛应用于神经网络中。它的定义如下：

\\[
\\text{ReLU}(x) = \\max(0, x)
\\]

在 PyTorch 中，可以通过 `torch.nn.ReLU` 或 `torch.nn.functional.relu` 来使用 ReLU 激活函数。例如：

```python
import torch
import torch.nn as nn

# 使用 nn.ReLU
relu = nn.ReLU()
input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
output_tensor = relu(input_tensor)
print(output_tensor)  # 输出: tensor([0., 0., 1., 2.])

# 使用 torch.nn.functional.relu
import torch.nn.functional as F
output_tensor = F.relu(input_tensor)
print(output_tensor)  # 输出: tensor([0., 0., 1., 2.])
```

## 操作数据集

在 PyTorch 中，`Dataset` 和 `DataLoader` 是处理和加载数据的两个重要工具。`Dataset` 用于存储样本及其对应的标签，而 `DataLoader` 用于将 `Dataset` 对象包装成一个可迭代对象，以便进行批量处理。

### 1. 创建自定义 Dataset

首先，我们需要创建一个自定义的 `Dataset` 类。这个类需要继承 `torch.utils.data.Dataset` 并实现以下三个方法：

- `__init__`: 初始化数据集
- `__len__`: 返回数据集的大小
- `__getitem__`: 根据索引返回一个样本

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
```

### 2. 创建 DataLoader

接下来，我们使用 `DataLoader` 来加载数据集。`DataLoader` 可以自动将数据集分成小批量，并在训练时进行随机打乱。

```python
# 示例数据
data = torch.randn(100, 3)  # 100 个样本，每个样本有 3 个特征
labels = torch.randint(0, 2, (100,))  # 100 个标签，值为 0 或 1

# 创建自定义数据集
dataset = CustomDataset(data, labels)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

### 3. 迭代 DataLoader

最后，我们可以使用 `DataLoader` 来迭代数据集。在训练模型时，我们通常会在每个 epoch 中迭代整个数据集。

```python
for epoch in range(5):  # 训练 5 个 epoch
    for batch_data, batch_labels in dataloader:
        # 在这里进行训练
        print(f"数据: {batch_data}, 标签: {batch_labels}")
```

## 损失函数，优化器的使用

在深度学习中，损失函数用于衡量模型预测值与真实值之间的差距，而优化器则用于调整模型参数以最小化损失函数的值。PyTorch 提供了多种损失函数和优化器，下面我们将介绍如何使用它们。

### 1. 损失函数

PyTorch 在 `torch.nn` 模块中提供了多种常用的损失函数。以下是一些常见的损失函数及其使用示例：

- `nn.MSELoss`: 均方误差损失，常用于回归任务。
- `nn.CrossEntropyLoss`: 交叉熵损失，常用于分类任务。

```python
import torch
import torch.nn as nn

# 示例：均方误差损失
mse_loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
loss = mse_loss(input, target)
print(f"MSE Loss: {loss.item()}")

# 示例：交叉熵损失
cross_entropy_loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
loss = cross_entropy_loss(input, target)
print(f"Cross Entropy Loss: {loss.item()}")
```

## 用gpu手写训练和预测一个模型
