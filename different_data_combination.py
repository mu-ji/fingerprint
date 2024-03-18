import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import CNN_only_RTT
import CNN_with_RTT
import CNN_without_RTT

only_RTT_error_list = []
with_RTT_error_list = []
without_RTT_error_list = []
sample_length = 200
for i in range(10):
    error_array = CNN_only_RTT.main(sample_length)
    only_RTT_error_list.append(np.mean(np.abs(error_array)))
    error_array = CNN_with_RTT.main(sample_length)
    with_RTT_error_list.append(np.mean(np.abs(error_array)))
    error_array = CNN_without_RTT.main(sample_length)
    without_RTT_error_list.append(np.mean(np.abs(error_array)))

plt.figure()
plt.boxplot([only_RTT_error_list, with_RTT_error_list, without_RTT_error_list])
plt.title('predict error of different model')
plt.xticks([1, 2, 3], ['only RTT', 'both RTT and RSSI', 'only RSSI'])
plt.xlabel('model')
plt.ylabel('predict error')
plt.show()

