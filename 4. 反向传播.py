# 使用pytorch实现反向传播
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度

def forward(x):
    return x * w  # w是一个Tensor

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    l = 0  # 为了在for循环之前定义l,以便之后的输出不报错，无实际意义
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 建立计算图
        l.backward()  # 反向传播，计算出所有梯度，存到变量里面（w），释放计算图
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor
        w.grad.data.zero_()  # 梯度清零
    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, forward(4).item())