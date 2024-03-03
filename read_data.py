import numpy as np
import matplotlib.pyplot as plt

data = np.load('fingerprint_experiment1/distance0.npz')
anchor1_rtt_list = data['anchor1_rtt']
anchor1_response_list = data['anchor1_response']
anchor1_request_list = data['anchor1_request']
anchor2_rtt_list = data['anchor2_rtt']
anchor2_response_list = data['anchor2_response']
anchor2_request_list = data['anchor2_request']
anchor3_rtt_list = data['anchor3_rtt']
anchor3_response_list = data['anchor3_response']
anchor3_request_list = data['anchor3_request']

plt.figure()
ax1 = plt.subplot(221)
ax1.plot([i for i in range(len(anchor1_rtt_list))],anchor1_rtt_list,label = 'anchor1 rtt')
ax1.plot([i for i in range(len(anchor2_rtt_list))],anchor2_rtt_list,label = 'anchor2 rtt')
ax1.plot([i for i in range(len(anchor3_rtt_list))],anchor3_rtt_list,label = 'anchor3 rtt')
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