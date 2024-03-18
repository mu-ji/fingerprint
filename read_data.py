import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft

data = np.load('fingerprint_experiment1/distance7.npz')
anchor1_rtt_list = data['anchor1_rtt']
for i in range(len(anchor1_rtt_list)):
    if anchor1_rtt_list[i] > 21000:
        anchor1_rtt_list[i] = anchor1_rtt_list[i-1]

anchor1_response_list = data['anchor1_response']
anchor1_request_list = data['anchor1_request']
anchor2_rtt_list = data['anchor2_rtt']
for i in range(len(anchor2_rtt_list)):
    if anchor2_rtt_list[i] > 21000:
        anchor2_rtt_list[i] = anchor2_rtt_list[i-1]
anchor2_response_list = data['anchor2_response']
anchor2_request_list = data['anchor2_request']
anchor3_rtt_list = data['anchor3_rtt']
for i in range(len(anchor3_rtt_list)):
    if anchor3_rtt_list[i] > 21000:
        anchor3_rtt_list[i] = anchor3_rtt_list[i-1]
anchor3_response_list = data['anchor3_response']
anchor3_request_list = data['anchor3_request']

plt.figure()
ax1 = plt.subplot(221)
ax1.hist(anchor1_rtt_list, label = 'anchor1 rtt')
ax1.hist(anchor2_rtt_list, label = 'anchor2 rtt')
ax1.hist(anchor3_rtt_list, label = 'anchor3 rtt')
ax1.set_title('RTT in different anchor')
ax1.legend()
ax1.grid()

ax2 = plt.subplot(222)
ax2.plot([i for i in range(len(anchor1_response_list))],anchor1_response_list,label = 'anchor1 response RSSI')
ax2.plot([i for i in range(len(anchor2_response_list))],anchor2_response_list,label = 'anchor2 response RSSI')
ax2.plot([i for i in range(len(anchor3_response_list))],anchor3_response_list,label = 'anchor3 response RSSI')
ax2.set_title('response RSSI in different anchor')
ax2.legend()
ax2.grid()

ax3 = plt.subplot(223)
ax3.plot([i for i in range(len(anchor1_request_list))],anchor1_request_list,label = 'anchor1 request RSSI')
ax3.plot([i for i in range(len(anchor2_request_list))],anchor2_request_list,label = 'anchor2 request RSSI')
ax3.plot([i for i in range(len(anchor3_request_list))],anchor3_request_list,label = 'anchor3 request RSSI')
ax3.set_title('response RSSI in different anchor')
ax3.legend()
ax3.grid()

plt.show()

signal = anchor1_response_list
train_signal = signal[:10000]
test_signal = signal[10000:]
# 对原始信号进行傅里叶变换
train_spectrum = np.fft.fft(train_signal)
test_spectrum = np.fft.fft(test_signal)
# 计算频谱的能量
train_energy = np.abs(train_spectrum) ** 2
test_energy = np.abs(test_spectrum) ** 2

# 计算整体能量的5%阈值
coeff = 0.05
train_threshold = coeff * np.sum(train_energy)
test_threshold = coeff * np.sum(test_energy)

# 复制频谱，并将低于阈值的频率成分设为零
train_truncated_spectrum = train_spectrum.copy()
train_truncated_spectrum[train_energy < train_threshold] = 0
train_truncated_signal = np.real(np.fft.ifft(train_truncated_spectrum))

test_truncated_spectrum = test_spectrum.copy()
test_truncated_spectrum[test_energy < test_threshold] = 0
test_truncated_signal = np.real(np.fft.ifft(test_truncated_spectrum))
# 绘制滤波前后的时域信号
plt.figure(figsize=(12, 4))
plt.subplot(2, 2, 1)
plt.plot(train_signal)
plt.title('origial training signal')
plt.subplot(2, 2, 2)
plt.plot(train_truncated_signal)
plt.title('recover training signal')
plt.subplot(2, 2, 3)
plt.plot(test_signal)
plt.title('origial test signal')
plt.subplot(2, 2, 4)
plt.plot(test_truncated_signal)
plt.title('recover test signal')
plt.gca().ticklabel_format(style='plain', useOffset=False)
plt.show()

