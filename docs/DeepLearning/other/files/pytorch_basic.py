import numpy as np
import torch

# test matmul 
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

result = torch.matmul(matrix1, matrix2)

expected_result = torch.tensor([[19, 22], [43, 50]])

# Test if the result is as expected
assert torch.equal(result, expected_result), f"Test failed: {result} != {expected_result}"

print("Test passed: Matrix multiplication result is correct.")



## test sigmoid
import torch.nn.functional as F

# Define a tensor
input_tensor = torch.tensor([0.0, 2.0, -2.0])

# Apply sigmoid function
result = torch.sigmoid(input_tensor)

# Expected result
expected_result = torch.tensor([0.5, 0.8808, 0.1192])

# Test if the result is as expected
assert torch.allclose(result, expected_result, atol=1e-4), f"Test failed: {result} != {expected_result}"

print("Test passed: Sigmoid function result is correct.")

import torch
import torch.nn.functional as F

# Define a tensor
input_tensor = torch.tensor([1.0, 2.0, 3.0])

# Apply softmax function
result = F.softmax(input_tensor, dim=0)

# Expected result
expected_result = torch.tensor([0.0900, 0.2447, 0.6652])

# Test if the result is as expected
assert torch.allclose(result, expected_result, atol=1e-4), f"Test failed: {result} != {expected_result}"
print("Test passed: Softmax function result is correct.")



import torch
import torch.nn as nn

#
# | y1 |     | 0.1  0.2 |   | 1.0 |   | 0.1 |
# | y2 |  =  | 0.3  0.4 | * |     | + | 0.2 |
# | y3 |     | 0.5  0.6 |   | 2.0 |   | 0.3 |

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




import torch
from torch.utils.data import DataLoader, Dataset


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



# 示例数据
data = torch.randn(100, 3)  # 100 个样本，每个样本有 3 个特征
labels = torch.randint(0, 2, (100,))  # 100 个标签，值为 0 或 1

# 创建自定义数据集
dataset = CustomDataset(data, labels)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for epoch in range(5):  # 训练 5 个 epoch
    for batch_data, batch_labels in dataloader:
        # 在这里进行训练
        print(f"数据: {batch_data}, 标签: {batch_labels}")



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
