"""
    图像分类CNN的综合案例,图像分类
    流程:
        1.准备数据集,使用torchvision自带的CIFAR10数据集,5w训练集,1w测试集
        2.搭建卷积神经网络
        3.模型训练
        4.模型测试
"""
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

BATCH_SIZE = 8

# 构建数据集
def create_dataset():
    # 获取训练集
    train_dataset = CIFAR10(root='../data',train=True,transform=ToTensor(),download=True)
    # 获取测试集
    test_dataset = CIFAR10(root='../data',train=False,transform=ToTensor(),download=True)
    return train_dataset,test_dataset

# 构建卷积神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 6, 3,1,0)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2, 2,0)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(6, 16, 3,1,0)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2,0)

        # 全连接层
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 优化卷积神经网络,加入正则化，卷积核数量增多，
class ImageModel1(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, 3,1,0)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2, 2,0)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 128, 3,1,0)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2,0)

        # 全连接层
        self.linear1 = nn.Linear(128*6*6, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        return self.out(x)


# 训练模型
def train(train_dataset):
    # 1.创建数据加载器
    dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    # 2.创建模型
    model = ImageModel1()
    # 3.创建损失函数
    criterion = nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    # 5.循环遍历epoch,开始每轮的训练工作
    epochs = 10
    for epoch in range(epochs):
        # 定义变量，记录总损失，总样本数据量，预测正确样本个数,训练开始时间
        total_loss,total_samples,total_correct,start = 0.0,0,0,time.time()
        for x,y in dataloader:
            model.train()
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred,y)
            # 梯度清零+反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计预测正确的样本个数
            total_correct += (y_pred.argmax(dim=1) == y).sum()
            # 统计当前批次的总损失
            total_loss += loss.item() * len(y)
            # 统计当前批次的总样本个数
            total_samples += len(y)

        print(f'epoch:{epoch+1},loss:{total_loss/total_samples:.5f},acc:{total_correct/total_samples:.2f},time:{time.time()-start:.2f}s')

    torch.save(model.state_dict(),'./modal/image_model.pth')




# 模型测试
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    model = ImageModel()
    model.load_state_dict(torch.load('./modal/image_model.pth'))
    total_correct, total_samples = 0,0
    for x,y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)

    print(f'Acc:{total_correct/total_samples:.2f}')



if __name__ == '__main__':
    train_dataset,test_dataset = create_dataset()
    # # 图像展示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()
    # model = ImageModel()
    # summary(model, input_size=(3, 32, 32),batch_size=BATCH_SIZE)
    # train(train_dataset)
    evaluate(test_dataset)