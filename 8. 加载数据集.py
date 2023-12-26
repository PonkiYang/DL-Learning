import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):  # 构造函数
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 8维到6维
        self.linear2 = torch.nn.Linear(6, 4)  # 6维到4维
        self.linear3 = torch.nn.Linear(4, 1)  # 4维到1维
        self.sigmoid = torch.nn.Sigmoid()  # 因为他里边也没有权重需要更新，所以要一个就行了，单纯的算个数

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()  # 实例化模型
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__=='__main__':
    for epoch in range(1000):
        loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(epoch, loss.item())

