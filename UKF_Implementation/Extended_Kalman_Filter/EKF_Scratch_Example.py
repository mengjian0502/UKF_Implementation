'''
Extended Kalman Filter Implementation(Start with Scratch)
Author: Jian Meng

Description: 
The algorithm implementation start with the example from github to get an basic
intuition about EKF algorithm and thry to use EKF for more complex circumstances

Dependencies: 
Sympy
'''

# Required Libraries: 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos

'''---- Initialization ------'''
num_states = 5 # States

# Different sensor readings: 
dt = 1.0 / 50.0 # Measurement rate = 50Hz
dtGPS = 1.0 / 10.0 # Sample rate of GPS is 10Hz

# Construct the state vector: 

vs, psis, dpsis, dts, xs, ys, lats, lons = symbols('v psi dpsi T x y lat lon')

# State Vector
state = Matrix([xs,ys,psis,vs,dpsis])
# Dynamic Process
gs = Matrix([
    [xs + (vs / dpsis)*(sin(psis + dpsis * dts) - sin(psis))],
    [ys + (vs / dpsis)*(-cos(psis + dpsis * dts) + cos(psis))],
    [psis + dts * dpsis],
    [vs],
    [dpsis]
])


# Initialize the Covariance Matrix :
'''
The diagonal component corresponding to the expected variance in the
Corresponding State, which is the expected derivation.
'''
P = np.diag([100.0, 100.0, 100.0, 100.0, 100.0])

# Initialize the process noise covariance matrix: 
'''
The Dynamic covariance matrix corresponding to the uncertainty of the 
state equation, including the uncertainties of each state during each
iteration. 
'''

x_noise = 0.5 * dt * 2.0  # Uncertianty in x direction is a 2.0m/s^2 acceleration
y_noise = 0.5 * dt * 2.0  # Uncertainty in y direction is a 2.0m/s^2 acceleration
v_noise = dt * 2.0 # Total velocity noise
psis_noise = dt * 1.0 # Dynamic process of the direction include 1.0 degree error
dpsis_noise = dt * 1.0 # Changes in rotation rate has 1.0 degree error

Q = np.diag([x_noise ** 2,y_noise ** 2,psis_noise ** 2,v_noise**2,dpsis_noise**2])

'''---- Read the measurement -----'''
# The measurement of the position and the angle comes from the data generator

hs = Matrix([
    [xs],
    [ys],
    [vs],
    [dpsis]
])
