import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

length = 150
array = np.load("train_set/train_set_without_rtt_{}.npy".format(length))

np.random.shuffle(array)
train_X = array[:, :length*6]
train_Y = array[:,length*6]

train_X = train_X.reshape(440, 6, length)
train_X = train_X.reshape(440, 6, 10, int(length/10))

anchor1_response = train_X[:, 0, :, :]
anchor1_request = train_X[:, 1, :, :]
anchor2_response = train_X[:, 2, :, :]
anchor2_request = train_X[:, 3, :, :]
anchor3_response = train_X[:, 4, :, :]
anchor3_request = train_X[:, 5, :, :]

anchor1_RSSI = train_X[:, 0:2, :, :]
anchor2_RSSI = train_X[:, 2:4, :, :]
anchor3_RSSI = train_X[:, 4:6, :, :]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 5 * int(length/10/2), 256)
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

model = CNN()
model.eval()

def generate_CNN_features(model, data):
    tensor_data = torch.tensor(data, dtype=torch.float32)
    features = model(tensor_data)
    return features

anchor1_RSSI_features = generate_CNN_features(model, anchor1_RSSI)
anchor2_RSSI_features = generate_CNN_features(model, anchor2_RSSI)
anchor3_RSSI_features = generate_CNN_features(model, anchor3_RSSI)

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
    
anchor1_NN_model = RegressionNet()
anchor1_criterion = nn.MSELoss()
anchor1_optimizer = optim.SGD(anchor1_NN_model.parameters(), lr=0.01)

anchor2_NN_model = RegressionNet()
anchor2_criterion = nn.MSELoss()
anchor2_optimizer = optim.SGD(anchor2_NN_model.parameters(), lr=0.01)

anchor3_NN_model = RegressionNet()
anchor3_criterion = nn.MSELoss()
anchor3_optimizer = optim.SGD(anchor3_NN_model.parameters(), lr=0.01)

anchor1_RSSI_features = torch.tensor(anchor1_RSSI_features, dtype=torch.float32)
anchor1_Y = torch.tensor(train_Y.reshape(440,1), dtype=torch.float32)
anchor2_RSSI_features = torch.tensor(anchor2_RSSI_features, dtype=torch.float32)
anchor2_Y = torch.tensor(train_Y.reshape(440,1), dtype=torch.float32)
anchor3_RSSI_features = torch.tensor(anchor3_RSSI_features, dtype=torch.float32)
anchor3_Y = torch.tensor(train_Y.reshape(440,1), dtype=torch.float32)


def label_to_distance(labels, anchor_position):
    shape = labels.shape
    distance = np.abs(labels[:,0]-anchor_position).reshape(shape)
    return distance

anchor1_Y = label_to_distance(anchor1_Y, 3)
anchor2_Y = label_to_distance(anchor2_Y, 6)
anchor3_Y = label_to_distance(anchor3_Y, 9)
times = 3000

def training_single_anchor_NN(model, criterion, optimizer, features, labels, times):
    NN_model = model
    NN_criterion = criterion
    NN_optimizer = optimizer
    NN_features = features
    NN_labels = labels

    for epoch in range(times):
        outputs = NN_model(NN_features)
        loss = NN_criterion(outputs, NN_labels)

        NN_optimizer.zero_grad()
        loss.backward()
        NN_optimizer.step()

    return model

#def process_labels(labels, anchor1_position, anchor2_position, anchor3_position):

anchor1_model = training_single_anchor_NN(anchor1_NN_model, anchor1_criterion, anchor1_optimizer, anchor1_RSSI_features, anchor1_Y, times)
anchor2_model = training_single_anchor_NN(anchor2_NN_model, anchor2_criterion, anchor2_optimizer, anchor2_RSSI_features, anchor2_Y, times)
anchor3_model = training_single_anchor_NN(anchor3_NN_model, anchor3_criterion, anchor3_optimizer, anchor3_RSSI_features, anchor3_Y, times)

test_data = np.load('test_set/test_set_without_rtt_{}.npy'.format(length))

test_X = test_data[:, :length*6]
test_Y = test_data[:,length*6]
test_X = test_X.reshape(110, 6, length)
test_X = test_X.reshape(110, 6, 10, int(length/10))

test_anchor1_RSSI = test_X[:, 0:2, :, :]
test_anchor2_RSSI = test_X[:, 2:4, :, :]
test_anchor3_RSSI = test_X[:, 4:6, :, :]

test_anchor1_RSSI_features = generate_CNN_features(model, test_anchor1_RSSI)
test_anchor2_RSSI_features = generate_CNN_features(model, test_anchor2_RSSI)
test_anchor3_RSSI_features = generate_CNN_features(model, test_anchor3_RSSI)

with torch.no_grad():
    anchor1_predictions = anchor1_model(test_anchor1_RSSI_features)
    anchor2_predictions = anchor2_model(test_anchor2_RSSI_features)
    anchor3_predictions = anchor3_model(test_anchor3_RSSI_features)

print(anchor1_predictions)
print(anchor2_predictions)
print(anchor3_predictions)
def triangulation_localization(anchor1_predictions, anchor2_predictions, anchor3_predictions, anchor1_position, anchor2_position, anchor3_position):
    triangulation_predictions_list = []
    for i in range(len(anchor1_predictions)):
        anchor1_low = anchor1_position - anchor1_predictions[i][0]
        anchor1_high = anchor1_position + anchor1_predictions[i][0]
        anchor2_low = anchor2_position - anchor2_predictions[i][0]
        anchor2_high = anchor2_position + anchor2_predictions[i][0]
        anchor3_low = anchor3_position - anchor3_predictions[i][0]
        anchor3_high = anchor3_position + anchor3_predictions[i][0]
        interval_low = max(anchor1_low, anchor2_low, anchor3_low)
        interval_high = min(anchor1_high, anchor2_high, anchor3_high)
        triangulation_predictions_list.append((interval_low + interval_high)/2)
    return triangulation_predictions_list

triangulation_predicitions = triangulation_localization(anchor1_predictions, anchor2_predictions, anchor3_predictions, 3, 6, 9)
print(triangulation_predicitions)
print(test_Y)
        
