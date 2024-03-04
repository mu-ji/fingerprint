import numpy as np
import random

distance_list = [i for i in range(11)]

def distance_sample(distance,sample_number,sample_length):
    data = np.load('fingerprint_experiment1/distance{}.npz'.format(distance))
    anchor1_rtt_array = data['anchor1_rtt']
    anchor1_response_array = data['anchor1_response']
    anchor1_request_array = data['anchor1_request']
    anchor2_rtt_array = data['anchor2_rtt']
    anchor2_response_array = data['anchor2_response']
    anchor2_request_array = data['anchor2_request']
    anchor3_rtt_array = data['anchor3_rtt']
    anchor3_response_array = data['anchor3_response']
    anchor3_request_array = data['anchor3_request']

    distance_sample_without_rtt_array = np.array([0]*(sample_length*6+1))
    distance_sample_with_rtt_array = np.array([0]*(sample_length*9+1))
    for i in range(sample_number):
        k = random.randint(0,10000)
        anchor1_rtt_sample = anchor1_rtt_array[k:k+sample_length]
        anchor1_response_sample = anchor1_response_array[k:k+sample_length]
        anchor1_request_sample = anchor1_request_array[k:k+sample_length]
        anchor2_rtt_sample = anchor2_rtt_array[k:k+sample_length]
        anchor2_response_sample = anchor2_response_array[k:k+sample_length]
        anchor2_request_sample = anchor2_request_array[k:k+sample_length]
        anchor3_rtt_sample = anchor3_rtt_array[k:k+sample_length]
        anchor3_response_sample = anchor3_response_array[k:k+sample_length]
        anchor3_request_sample = anchor3_request_array[k:k+sample_length]

        sample_without_rtt = np.hstack((anchor1_response_sample,anchor1_request_sample,
                                        anchor2_response_sample,anchor2_request_sample,
                                        anchor3_response_sample,anchor3_request_sample,distance))
        sample_with_rtt = np.hstack((anchor1_rtt_sample,anchor1_response_sample,anchor1_request_sample,
                                    anchor2_rtt_sample,anchor2_response_sample,anchor2_request_sample,
                                    anchor3_rtt_sample,anchor3_response_sample,anchor3_request_sample,distance))
        
        #print(sample_with_rtt)
        distance_sample_with_rtt_array = np.vstack((distance_sample_with_rtt_array,sample_with_rtt))
        distance_sample_without_rtt_array = np.vstack((distance_sample_without_rtt_array,sample_without_rtt))
    
    return distance_sample_without_rtt_array[1:,:],distance_sample_with_rtt_array[1:,:]

a,b = distance_sample(1,10,200)
print(a)
def generate_training_set(sample_number,sample_length):
    train_set_without_rtt = np.array([0]*(sample_length*6+1))
    train_set_with_rtt = np.array([0]*(sample_length*9+1))
    for distance in distance_list:
        distance_sample_without_rtt_array,diatance_sample_with_rtt_array = distance_sample(distance,sample_number,sample_length)
        train_set_without_rtt = np.vstack((train_set_without_rtt,distance_sample_without_rtt_array))
        train_set_with_rtt = np.vstack((train_set_with_rtt,diatance_sample_with_rtt_array))
    
    train_set_without_rtt = train_set_without_rtt[1:,:]
    train_set_with_rtt = train_set_with_rtt[1:,:]
    return train_set_without_rtt,train_set_with_rtt

train_set_without_rtt,train_set_with_rtt = generate_training_set(20,200)
np.save('train_set/train_set_without_rtt.npy',train_set_without_rtt)
np.save('train_set/train_set_with_rtt.npy',train_set_with_rtt)