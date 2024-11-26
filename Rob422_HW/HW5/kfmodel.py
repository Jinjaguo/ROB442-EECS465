import numpy as np

###YOUR CODE HERE###
#motion model
A = np.eye(2) #replace with your A
B = np.array([[1.5, 0.1], [0.2, -0.5]])

#sensor model
C = np.array([[1.05, 0.01], [0.01, 0.9]])
    
#motion noise covariance
R = np.array([[2.50696845e-03, 1.79957758e-05], [1.79957758e-05, 2.51063277e-03]])

#sensor noise covariance
Q = np.array([[0.04869528, -0.0058636], [-0.0058636, 1.01216104]])
###YOUR CODE HERE###
