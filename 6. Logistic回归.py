import  torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0], [4.0]])
y_data = torch.Tensor([[0],[0],[1], [1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):#构造函数
        super().__init__()
        self.linear = torch.nn.Linear(1,1)#构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return  y_pred

model = LogisticRegressionModel()#实例化模型

# 构造损失函数
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss.backward()         # 调用loss.backward()时，它会计算损失相对于模型参数的梯度，并将这些梯度存储在每个参数的.grad属性中。
                            # 这些梯度在optimizer.step()中被用来更新模型的参数。
    optimizer.step()        # 进行更新
    optimizer.zero_grad()   # 梯度归零
    print(epoch,loss.item())

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))        # view() 改变tensor形状
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')        # 显示红色直线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()                                  # 显示网格先
plt.show()
