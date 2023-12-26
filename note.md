## **zip()**

函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

```python
a = [1,2,3]
b = [4,5,6]

zipped = zip(a,b)
print(zipped)
print(list(zipped))

a1, a2 = zip(*zip(a,b))
print(*zip(a,b))
print(list(a1))
print(list(a2))

# <zip object at 0x000001C4C56455C0>
# [(1, 4), (2, 5), (3, 6)]
# (1, 4) (2, 5) (3, 6)
# [1, 2, 3]
# [4, 5, 6]
```

```python
# 列表元素依次相连：
l = ['a', 'b', 'c', 'd', 'e','f']
print zip(l[:-1],l[1:])

# [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f')]
```

```python
# 可以用于对二维列表（矩阵）取列:
matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(list(zip(*matrix)))

# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

```python
# 值得注意：
matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(*matrix)

# [1, 2, 3] [4, 5, 6] [7, 8, 9]
# *放在列表外面应该是将列表最外层的中括号去掉
```

## **meshgrid()**

用于生成网格采样点矩阵。

```python
import numpy as np
import matplotlib.pyplot as plt

m, n = 5, 4
x = np.linspace(0, m-1, m)
y = np.linspace(0, n-1, n)

# x: array([0., 1., 2., 3., 4.])
# y: array([0., 1., 2., 3.])

X, Y = np.meshgrid(x, y)
print(X)
print(Y)

plt.plot(X, Y, 'o--')
plt.grid(True)
plt.show()

# 输出
# X
array([[0., 1., 2., 3., 4.],
       [0., 1., 2., 3., 4.],
       [0., 1., 2., 3., 4.],
       [0., 1., 2., 3., 4.]])
# Y
array([[0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1.],
       [2., 2., 2., 2., 2.],
       [3., 3., 3., 3., 3.]])
# 假设 x, y 分别为 m, n 维向量，则矩阵（数组）X, Y 的 dimension 都是： n * m。其中矩阵 X 中的行都为向量 x，矩阵 Y 的列都为向量 y
# 生成的X和Y可以直接放在plt.plot()里面生成网格点。
```

## 画3D图

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

w = np.arange(0.0, 4.1, 0.1)
b = np.arange(0.0, 4.1, 0.1)
[W, B] = np.meshgrid(w, b)

fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)	
ax.plot_surface(W, B, l_sum/3)

# ax.plot_surface应与np.meshgrid()配合使用, 第三个参数应该也是二维数组
```

## np.ones()

```python
a = np.ones((4, 4))
print(a)
b = np.ones(4)
print(b)

# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
# [1. 1. 1. 1.]
```

## 梯度下降

一些理解：梯度下降中，横轴是w，纵轴是cost，所以梯度为：
$$
\frac{∂cost(w)}{∂w}
$$


### cost function 求MSE

<img src="C:\Users\Phimi\Desktop\DLStart\assets\image-20231222151744427.png" alt="image-20231222151744427" style="zoom: 67%;" />

### Gradient

<img src="C:\Users\Phimi\Desktop\DLStart\assets\image-20231222151958659.png" alt="image-20231222151958659" style="zoom:67%;" />

### 梯度求导过程（很重要）

<img src="assets\image-20231222132553619.png" alt="image-20231222132553619" style="zoom:50%;" />

### 梯度下降与随机梯度下降 

<img src="C:\Users\Phimi\Desktop\DLStart\assets\image-20231222153543537.png" alt="image-20231222153543537" style="zoom: 50%;" />



## Tensor和tensor

torch.Tensor()是python类，更明确地说，是默认张量类型torch.FloatTensor()的别名，torch.Tensor([1,2])会调用Tensor类的构造函数__init__，生成单精度浮点类型的张量。

```python
>>> a=torch.Tensor([1,2])
>>> a.type()
'torch.FloatTensor'
```

而torch.tensor()仅仅是python函数，函数原型是：

```python
torch.tensor(data, dtype=None, device=None, requires_grad=False)
# torch.tensor会从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor、torch.FloatTensor和torch.DoubleTensor。

>>> a=torch.tensor([1,2])
>>> a.type()
'torch.LongTensor'

>>> a=torch.tensor([1.,2.])
>>> a.type()
'torch.FloatTensor'

>>> a=np.zeros(2,dtype=np.float64)
>>> a=torch.tensor(a)
>>> a.type()
'torch.DoubleTensor'
```

一般来说`torch.Tensor([1])`会生成一个单精度浮点张量，而`a=torch.tensor([1])`会生成一个整形张量。

每次进行loss()计算时都会动态地构建一个计算图。
y_pred ＝ x*w时，w是需要计算梯度的Tensor，则y_pred也需要计算梯度。
一个Tensor里面的grad也是一个Tensor，所以更新权重时，需要用w.grad.data。如果直接使用w.grad乘以学习率来更新权重，那就是在建立计算图，但计算图在之前调用backward()的时候已经释放了，需要到下一次计算loss的时候才能再建立计算图，所以不能这样弄。
使用张量的data来进行计算是不会建立计算图的。
w.grad.item()是直接把梯度里面的数值直接拿出来变成一个python里的标量。也是为了防止产生计算图。

## 随机梯度下降

损失函数：计算的是一个样本的误差
代价函数：是整个训练集上所有样本误差的平均

##  使用Pytorch实现线性回归

<img src="assets\image-20231223164512976.png" alt="image-20231223164512976" style="zoom: 67%;" />

loss函数包括一次正向的前馈过程，用来算出损失，backward函数是反馈过程，算出梯度。

线性回归就是一个最简单的只有一个神经元的神经网络。

<img src="C:\Users\Phimi\Desktop\DLStart\assets\image-20231223170002786.png" alt="image-20231223170002786" style="zoom:50%;" />

最终计算出的loss必须是一个标量而不是矩阵，可以是loss矩阵内所有值的求和，因为算出来的loss是个向量的话，是用不了backward()的。

### 仿射模型

<img src="C:\Users\Phimi\Desktop\DLStart\assets\image-20231223165645120.png" alt="image-20231223165645120" style="zoom:50%;" />	“ 仿射变换 ”就是：“线性变换”+“平移

### pytorch中的\_\_call\_\_， \__init__，forward

```python
class A():
    def __call__(self):
        print('i can be called like a function')

a = A()
a()

# i can be called like a function
```

```python
class A():
    def __call__(self, param):
        print('i can called like a function')
        print('掺入参数的类型是：', type(param))

a = A()
a('i')

# i can called like a function
# 掺入参数的类型是： <class ‘str’>
```

当然也可以在`__call__`里调用其他的函数啊，
在`__call__`函数中调用`forward`函数，并且返回调用的结果。

```python
class A():
    def __call__(self, param):
        print('i can called like a function')
        print('传入参数的类型是：{}   值为： {}'.format(type(param), param))
        res = self.forward(param)
        return res
 
    def forward(self, input_):
        print('forward 函数被调用了')
        print('in  forward, 传入参数类型是：{}  值为: {}'.format( type(input_), input_))
        return input_
 
a = A() 
input_param = a('i')
print("对象a传入的参数是：", input_param)

# i can called like a function
# 传入参数的类型是：<class ‘str’> 值为： i
# forward 函数被调用了
# in forward, 传入参数类型是：<class ‘str’> 值为: i
# 对象a传入的参数是： i
```

现在我们将初始化函数`__init__`也加上，来看一下：
在对象初始化时确定初始年龄，通过调用`a(2)`为对象年龄增加2岁。

```python
class A():
    def __init__(self, init_age):
        print('我年龄是:',init_age)
        self.age = init_age

    def __call__(self, added_age):
        res = self.forward(added_age)
        return res

    def forward(self, input_):
        print('forward 函数被调用了')
        return input_ + self.age
    
print('对象初始化。。。。')
a = A(10)
input_param = a(2)
print("我现在的年龄是：", input_param)

# 对象初始化。。。。
# 我年龄是: 10
# forward 函数被调用了
# 我现在的年龄是： 12
```

model.parameters()保存的是Weights和Bais参数的值。

## 处理多维度特征的输入

```python
import numpy as np
import torch

xy = np.loadtxt('data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
print(x_data)
print(y_data)

# x_data和y_data都是矩阵，其中y_data后面的那个-1外面需要加中括号，否则y_data就只是一个向量了
```

使用`self.sigmoid = torch.nn.Sigmoid()`作为激活函数时，它作为一个网络层，参与构建计算图。

如果想查看某些层的参数，以神经网络的第一层参数为例，可按照以下方法进行。

```python
# 参数说明
# 第一层的参数：
layer1_weight = model.linear1.weight.data
layer1_bias = model.linear1.bias.data
print("layer1_weight", layer1_weight)
print("layer1_weight.shape", layer1_weight.shape)
print("layer1_bias", layer1_bias)
print("layer1_bias.shape", layer1_bias.shape)
```

## epoch

所有的训练样本完成一次前向传播和反向传播参与一次训练)的过程。

## 加载数据集

Dataset类是抽象类不能被实例化，只能被其他子类继承。

```python

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] 
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):	# magic function，用于给dataset[index]调用（该类实例化的对象支持下标操作）
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):				# magic function，用于给len(dataset)调用，其中dataset是该类实例化的对象
        return self.len
```



## if epoch % 2000 == 1999

使用 `if epoch % 2000 == 1999:` 和 `if epoch % 2000 == 0:` 的区别主要在于它们所检查的 `epoch` 值是不同的，并且因此触发的时机也不同。

`if epoch % 2000 == 1999:`

这个条件会在 `epoch` 为 1999, 3999, 5999, ... 时为真。也就是说，它会在每个2000个周期的最后一个周期时触发。

2. `if epoch % 2000 == 0:`

这个条件会在 `epoch` 为 0, 2000, 4000, ... 时为真。也就是说，它会在每个2000个周期的第一个周期时触发。
因此，主要的区别在于触发的时机：一个是在每个2000周期的最后一个周期，而另一个是在每个2000周期的第一个周期。
如果你的意图是每隔2000个周期执行一次操作，那么使用 `if epoch % 2000 == 1999:` 将确保在2000周期的末尾执行操作，而使用 `if epoch % 2000 == 0:` 则会在每个新的2000周期开始时执行操作。

## with torch.no_grad():

强制之后的内容不进行计算图构建。
在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。

## 模型性能测试函数

```python
def test():
    with torch.no_grad():
        y_pred = model(Xtest)
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, Ytest).sum().item() / Ytest.size(0)
        print("test acc:", acc)
        
# 使用torch.where函数对y_pred中的每个元素进行条件判断：如果预测值大于或等于0.5，则赋值为1.0，否则赋值为0.0。这样，y_pred_label中的每个元素都被转换为相应的标签值（0或1）。
# 使用torch.eq函数比较y_pred_label和真实标签Ytest，返回一个布尔Tensor。
# 使用.sum()对所有为True的元素进行求和，这给出了预测正确的样本数量。
# .item()将这个数字提取为一个Python标量。
# 最后，将这个数字除以Ytest的第一个维度的大小（即样本数量），得到模型的准确率。
```

## 独热编码

自然状态码和独热编码是两种常见的编码方式，用于将分类变量转换为机器学习算法可以处理的格式。

1. **自然状态码**：在这种编码方式中，每个状态（或类别）用一个二进制数字表示。在你的例子中，你有6个状态，因此需要6位二进制数字来表示。例如，状态0被编码为000000，状态1被编码为000001，依此类推。
2. **独热编码**（也称为one-hot encoding）：在这种编码方式中，每个状态被表示为一个唯一的二进制向量，其中只有一个位置为1，其余位置为0。例如，状态0被编码为000001，状态1被编码为000010，依此类推。注意，不同的状态编码之间是互斥的（即没有两个状态的编码会有相同的二进制向量）。

```
自然状态码为：
000,001,010,011,100,101
对应独热编码为：
000001,000010,000100,001000,010000,100000
```

