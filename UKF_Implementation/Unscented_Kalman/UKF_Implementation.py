'''
UKF example
Reference: Examples from PyKalman Documents

Jian Meng

Dependencies: PyKalman
'''

# Required Libraries
import numpy as np 
import matplotlib.pyplot as plt
from pykalman import UnscentedKalmanFilter


def transition_funciton(state, noise):
    '''
    --------
    The function that evolve the state from k-1 to k.
    --------
    Inputs: 
    state: State vector of the system
    noise: The noise of the dynamic process
    '''
    a = np.sin(state[0]) + state[1] + noise[0]
    b = state[1] + noise[1]

    return np.array([a,b])

def observation_function(state, noise):
    '''
    The function is about the relationship between the state vector 
    and the external measurement
    '''
    C = np.array([
        [-1, 0.5],
        [0.2, 0.1]
    ])
    return np.dot(C, state) + noise


'''----INITIALIZE THE PARAMETERS----'''
transition_covariance = np.eye(2)
noise_generator = np.random.RandomState(0)
observation_covariance = np.eye(2) + noise_generator.randn(2,2) * 0.01
Initial_state = [0,0]
intial_covariance = [[1,0.1], [-0.1,1]]

# UKF
kf = UnscentedKalmanFilter(
    transition_funciton, observation_function,
    transition_covariance, observation_covariance,
    Initial_state,intial_covariance,
    random_state=noise_generator
)

states, observations = kf.sample(500,Initial_state)
# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

plt.figure(figsize=(8,8))
plt.plot(states, c='b')
plt.plot(range(500),np.sin(range(500)))
plt.plot(filtered_state_estimates, color='r', ls='-')
plt.plot(smoothed_state_estimates, color='g', ls='-.')
plt.show()