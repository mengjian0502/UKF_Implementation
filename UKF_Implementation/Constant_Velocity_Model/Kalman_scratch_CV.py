'''
Basic Kalman Filter implementation(Start with scratch)
Author: Jian Meng
Date: 10/24/18

Discription: 
The basic Kalman Filter Implementation focus on the constant
velocity model and using Kalman Filter technique to estimate 
the state of the moving obeject at each time step

'''

# Required libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.linalg as lin
from scipy.stats import norm

# Current path
my_path = os.path.abspath(__file__)
'''------- Initialize the initialize state and covariance matrix -----'''
# States matrix:
'''
The state matrix contains the position coordinates of the object in x and y
direction (x,y) and also the velocity of the object in x and y direction
'''
x = np.matrix([0.0,0.0,0.0,0.0]).T

# Initialize the covariance matrix
P = np.diag([100.0, 100.0, 100.0, 100.0])

# fig = plt.figure(figsize=(6,6))
# im = plt.imshow(P,cmap=plt.get_cmap('binary'))
# plt.title("Initial_Covariance")
# # Labels of x and y axis
# x_labels = ['$x$','$y$','$\dot x$','$\dot y$']
# y_labels = ['$x$','$y$','$\dot x$','$\dot y$']

# # Set the labels of the graph: 
# plt.xticks(np.arange(4),x_labels,fontsize=22)
# plt.yticks(np.arange(4),y_labels,fontsize=22)
# plt.savefig('Initial_Covariance.png',dpi=300)

# Time interval:
dt = 0.1

# Construct the transition matrix: Shape = (4,4)
A = np.matrix(
    [[1.0,0.0,dt,0.0],
    [0.0,1.0,0.0,dt],
    [0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,1.0]]
)
# Construct the measurement matrix: Shape = (2,4)
H = np.matrix(
    [[0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,1.0]]
)

# Covariance of the Measurement noise, which describe how 'noisy' the measurement is
# Assume the measurement noise is Gaussian noise with mean = 0 and standard diviation is 1
std = 1.0 ** 2
R = np.matrix(
    [[std,0],
    [0,std]]
)

# # Visualize the measurement noise: 
# x_noise = np.arange(-10,10,0.001)
# plt.subplot(121)
# plt.plot(x_noise, norm.pdf(x_noise, 0, std))
# plt.title('$\dot x$')

# plt.subplot(122)
# plt.plot(x_noise, norm.pdf(x_noise, 0, std))
# plt.title('$\dot y$')
# plt.savefig("Noise_of_V.png",dpi=300)

# Process error(Covariance), the error(disturbance) that occured during the state transition
# Assume the disturbance(acceleration) is 2m/s^2

mean_noise = 2
U = np.matrix([
    [0.5*dt**2],
    [0.5*dt**2],
    [dt],
    [dt],
])

Q = U * U.T * mean_noise

'''------ Measurement ----'''

samples = 200
vx = 20 # Velocity in x direction
vy = 10 # Velocity in y direction

# Add noise
mx = np.array(vx + np.random.randn(samples))
my = np.array(vy + np.random.randn(samples))


# Total measurements
T_m = np.vstack((mx,my))

# # Visualize the measurements:
# fig = plt.figure(figsize=(16,5))

# plt.step(range(samples),mx, label='$\dot x$')
# plt.step(range(samples),my, label='$\dot y$')
# plt.title("Measurements")
# plt.legend(loc='best')
# plt.savefig("Measurements.png",dpi=300)
# plt.show()


'''---- Kalman Filter Process ----'''
I = np.eye(4)
xt = []
yt = []
vx_e = []
vy_e = []
for ii in range(samples):
# Project the current state to next time step
    x = A * x 
# Project the current covariance to the next time step
    P = A * P * A.T + Q 
# Denominator of the Kalman Gain Equation
    D = H * P * H.T + R
# Compute the Kalman Gain:
    K = (P * H.T) * lin.pinv(D)
# Update the estimate
    Z = T_m[:,ii].reshape(2,1)
    y = Z - (H * x)   # Residual
    x = x + (K * y)
# Update Error Covariance 
    P = (I - (K * H))*P
# Save the state:
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    vx_e.append(float(x[2]))
    vy_e.append(float(x[3]))

'''---- Visualization of the Final Results ----'''

# Velocity Estimation:
fig = plt.figure(figsize=(16,9))
plt.plot(range(samples),vx_e, label='$\dot x$')
plt.plot(range(samples),vy_e, label='$\dot y$')

plt.axhline(vx,color='#999999',label='$\dot x_{True}$')
plt.axhline(vy,color='#999999',label='$\dot y_{True}$')

plt.xlabel("Filter Step")
plt.title("Estimation of Velocity")
plt.legend(loc='best')
plt.savefig("Velocity_Estimatoin.png",dpi=300)


# Position Estimation:
fig = plt.figure(figsize=(8,8))
plt.scatter(xt,yt,s=10,marker='x',label='Position',c='k')
plt.scatter(xt[0],yt[0],s=100,label='Start',c='r')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Position")
plt.legend(loc='best')
plt.savefig("Position_Estimation.png",dpi=300)
plt.show()