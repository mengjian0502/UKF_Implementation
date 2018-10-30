'''
UKF practice: Sine wave estimation

Author: Jian Meng
Date: 10/28/18

Dependencies: pykalman
'''

# Required libraries: 
import numpy as np
import matplotlib.pyplot as plt
from pykalman import UnscentedKalmanFilter


# Construct the transition function
def transition_function(state, noise):
    '''
    State transation function
    -------
    Inputs: 
    State: The state vector, in this case is the (x,y) coordinates in 2 - d space
    Noise: The transistion noise
    '''
    x = state[0] + 0.1
    y = np.sin(state[0]) + noise[0]  
    
    return np.array([x,y])

# # TEST OF transition_function
# sample = 100
# state = [0,0]
# noise_generator = np.random.RandomState(0)
# for ii in range(sample):
#     plt.scatter(state[0],state[1],marker='x',c='k')
#     state = transition_function(state,noise_generator.randn(2,1) * 0.1)
# print(state)
# plt.show()

def observation_function(state,noise):
    '''
    Observation function
    '''
    xm = state[0] + noise[1]
    ym = np.sin(state[0]) + noise[1]
    return np.array([xm,ym])

'''----INITIALIZE THE PARAMETERS----'''
transition_covariance = np.eye(2)
noise_generator = np.random.RandomState(0)
observation_covariance = np.eye(2) + noise_generator.randn(2,2) * 0.1
Initial_state = [0,0]
intial_covariance = [[1,0.1], [-0.1,1]]


# UKF
kf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    Initial_state,intial_covariance,
    random_state=noise_generator
)

sample = 200
states, observations = kf.sample(sample,Initial_state)
# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

# True line
t = np.linspace(0,sample*0.1,sample)
y = np.sin(t)

plt.plot(filtered_state_estimates[:,0],filtered_state_estimates[:,1], color='r', ls='-',label='UKF')
plt.plot(smoothed_state_estimates[:,0], smoothed_state_estimates[:,1],color='g', ls='-.',label='UKF Smother')
state_lines = plt.plot(states[:,0], states[:,1], color='b')
state_project = plt.scatter(states[:,0],states[:,1],marker='.',label='States')
plt.plot(t,y,c='k',label='True')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel("Time Step")
plt.savefig("UKF_Sinewave.png",dpi=300)
plt.show()
