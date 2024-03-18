import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, sample_length):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 5 * int(sample_length/10/2), 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 30)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        
        # 定义全连接层
        self.fc1 = nn.Linear(30, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

def main(sample_length):
    # 使用np.loadtxt函数读取保存的文本数组数据
    sample_length = sample_length
    array = np.load("train_set/train_set_with_rtt_{}.npy".format(sample_length))

    # 打印读取的数组数据
    print(array.shape)

    np.random.shuffle(array)
    features = array[:, :sample_length*9]
    features[:,:sample_length] -= 20070
    features[:,sample_length*3:sample_length*4] -= 20070
    features[:,sample_length*6:sample_length*7] -= 20070

    Y_labels = array[:,sample_length*9]

    reshaped_features = np.hstack((features[:,:200], features[:,600:800], features[:,1200:1400]))
    reshaped_features = reshaped_features.reshape(440,3,sample_length)
    reshaped_features = reshaped_features.reshape(440,3,10,int(sample_length/10))

    # 创建CNN模型实例
    model = CNN(sample_length)

    # 将模型设置为评估模式
    model.eval()

    tensor_data = torch.tensor(reshaped_features, dtype=torch.float32)
    #tensor_data = tensor_data.unsqueeze(1)
    features = model(tensor_data)

    NN_model = RegressionNet()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(NN_model.parameters(), lr=0.01)


    features = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(Y_labels.reshape(440,1), dtype=torch.float32)
    for epoch in range(10000):
        # 前向传播
        outputs = NN_model(features)
        
        # 计算损失
        loss = criterion(outputs, Y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练过程中的损失
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    test_data = np.load('test_set/test_set_with_rtt.npy')

    test_data[:,:sample_length] -= 20070
    test_data[:,sample_length*3:sample_length*4] -= 20070
    test_data[:,sample_length*6:sample_length*7] -= 20070

    X = np.hstack((test_data[:,:sample_length], test_data[:,sample_length*3:sample_length*4], test_data[:,sample_length*6:sample_length*7]))
    Y = test_data[:,sample_length*9]
    X = X.reshape(110, 3, sample_length)
    X = X.reshape(110, 3, 10, int(sample_length/10))
    test_data_tensor = torch.tensor(X, dtype=torch.float32)
    #test_data_tensor = test_data_tensor.unsqueeze(1)
    features = model(test_data_tensor)

    with torch.no_grad():
        predictions = NN_model(features)

    predictions = predictions.numpy()

    predictions_reshaped = predictions.reshape(-1, 10)
    labels_reshaped = Y.reshape(-1, 10)

    error_array = predictions_reshaped - labels_reshaped
    '''
    print(error_array.shape)

    fig, ax = plt.subplots()

    # 绘制小提琴图
    ax.violinplot(error_array.T,showmeans=True)

    # 添加标题和标签
    ax.set_title('prediction error')
    labels = (['{} meters'.format(i) for i in range(11)])
    plt.xticks([i for i in range(1,12)], labels)
    plt.grid()
    ax.set_ylabel('prediction error')

    # 显示图形
    plt.show()
    '''
    return error_array


#error_array = main()
#print(error_array.shape)
#print(np.mean(np.abs(error_array)))