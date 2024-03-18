import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import CNN_only_RTT
import CNN_with_RTT
import CNN_without_RTT

length_50 = []
length_100 = []
length_150 = []
length_200 = []
length_300 = []
for i in range(10):
    error_array = CNN_with_RTT.main(50)
    length_50.append(np.mean(np.abs(error_array)))
    error_array = CNN_with_RTT.main(100)
    length_100.append(np.mean(np.abs(error_array)))
    error_array = CNN_with_RTT.main(150)
    length_150.append(np.mean(np.abs(error_array)))
    error_array = CNN_with_RTT.main(200)
    length_200.append(np.mean(np.abs(error_array)))
    error_array = CNN_with_RTT.main(300)
    length_300.append(np.mean(np.abs(error_array)))


plt.figure()
plt.boxplot([length_50, length_100, length_150, length_200, length_300])
plt.title('predict error of different sample length')
plt.xticks([1, 2, 3, 4, 5], ['length = 50', 'length = 100', 'length = 150', 'length = 200', 'length = 300'])
plt.xlabel('different sample length')
plt.ylabel('predict error')
plt.show()