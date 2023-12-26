import  torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):#构造函数
        super().__init__()
        self.linear = torch.nn.Linear(1,1)#构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b
    def forward(self, x):
        y_pred = self.linear(x)#可调用对象，计算y=wx+b
        return  y_pred

model = LinearModel()#实例化模型

# 构造损失函数
criterion = torch.nn.MSELoss(size_average=False)
# 构造优化器，优化器不是Module，不会构建计算图
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
# model.parameters()保存的是Weights和Bais参数的值。
# model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上



for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss.backward()         # 调用loss.backward()时，它会计算损失相对于模型参数的梯度，并将这些梯度存储在每个参数的.grad属性中。
                            # 这些梯度在optimizer.step()中被用来更新模型的参数。
    optimizer.step()        # 进行更新
    optimizer.zero_grad()   # 梯度归零
    print(epoch,loss.item())

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
