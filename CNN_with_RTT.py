import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# 使用np.loadtxt函数读取保存的文本数组数据
array = np.load("train_set/train_set_with_rtt.npy")

# 打印读取的数组数据
print(array.shape)

np.random.shuffle(array)
features = array[:, :1800]
features[:,:200] -= 20070
features[:,600:800] -= 20070
features[:,1200:1400] -= 20070
print(features)
Y_labels = array[:,1800]
# 将特征重塑为（6 * 200）
reshaped_features = features.reshape(440, 9, 200)

# 打印重塑后的特征形状
print(reshaped_features[1])
print(Y_labels[1])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 定义全连接层
        self.fc1 = nn.Linear(16 * 3 * 99, 128)  
        self.fc2 = nn.Linear(128, 10)  # 最终输出10个特征
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # 展开为一维向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# 创建CNN模型实例
model = CNN()

# 将模型设置为评估模式
model.eval()

tensor_data = torch.tensor(reshaped_features, dtype=torch.float32)
tensor_data = tensor_data.unsqueeze(1)
features = model(tensor_data)

print(features)

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        
        # 定义全连接层
        self.fc1 = nn.Linear(10, 128)
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

NN_model = RegressionNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(NN_model.parameters(), lr=0.01)


features = torch.tensor(features, dtype=torch.float32)
Y = torch.tensor(Y_labels.reshape(440,1), dtype=torch.float32)
for epoch in range(2000):
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

X = test_data[:, :1800]
X[:,:200] -= 20070
X[:,600:800] -= 20070
X[:,1200:1400] -= 20070
Y = test_data[:,1800]
X = X.reshape(110, 9, 200)
test_data_tensor = torch.tensor(X, dtype=torch.float32)
test_data_tensor = test_data_tensor.unsqueeze(1)
features = model(test_data_tensor)

with torch.no_grad():
    predictions = NN_model(features)

predictions = predictions.numpy()

predictions_reshaped = predictions.reshape(-1, 10)
labels_reshaped = Y.reshape(-1, 10)

error_array = predictions_reshaped - labels_reshaped
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