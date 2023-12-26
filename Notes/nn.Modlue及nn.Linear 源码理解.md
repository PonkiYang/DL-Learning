```python
import torch
from torch import nn

m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)

print(output.size())

# torch.Size([128, 30])
```

首先创建类对象m，然后通过`m(input)`实际上调用`__call__(input)`，然后`__call__(input)`调用`forward()`函数，最后返回计算结果为：
[128,20]×[20,30]=[128,30]

所以自己创建多层神经网络模块时，只需要再实现`__init__`和`forward`即可。
一个简单的三层神经网络的例子：

```python
# define three layers
class simpleNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

```

