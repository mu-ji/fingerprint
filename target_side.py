import serial
import numpy as np
import matplotlib.pylab as plt

ser = serial.Serial('COM4', 115200)

times = 0
itration = 3000
rawFrame = []

anchor1_rtt_list = []
anchor1_response_list = []
anchor1_request_list  = []

anchor2_rtt_list = []
anchor2_response_list = []
anchor2_request_list  = []

anchor3_rtt_list = []
anchor3_response_list = []
anchor3_request_list  = []

while times < itration:
#while True:
    #while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 15:
                receiver_id = rawFrame[0]
                print('receiver_id:',receiver_id)
                decimal_data = int.from_bytes(rawFrame[1:5],byteorder='big')
                print('RTT time:',decimal_data)
                response_rssi = bytes(rawFrame[5:9])
                response_rssi = int(response_rssi.decode('utf-8'))
                print('response rssi:',response_rssi)
                request_rssi = bytes(rawFrame[9:13])
                request_rssi = int(request_rssi.decode('utf-8'))
                print('request rssi:',request_rssi)
                print('-------------------------------')
                times = times + 1
                if receiver_id == 1:
                    anchor1_rtt_list.append(decimal_data)
                    anchor1_response_list.append(response_rssi)
                    anchor1_request_list.append(request_rssi)
                elif receiver_id == 2:
                    anchor2_rtt_list.append(decimal_data)
                    anchor2_response_list.append(response_rssi)
                    anchor2_request_list.append(request_rssi)
                elif receiver_id == 3:
                    anchor3_rtt_list.append(decimal_data)
                    anchor3_response_list.append(response_rssi)
                    anchor3_request_list.append(request_rssi)

                     
            rawFrame = []

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