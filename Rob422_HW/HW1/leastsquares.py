import numpy as np
import matplotlib.pyplot as plt

# for Problem a)
# Load data from file
data = np.loadtxt('calibration.txt')
# Load data from file
Commanded_data = data[:, 0]
Measured_data = data[:, 1]
# Pseudo-inverse
A = np.vstack([Commanded_data, np.ones(Commanded_data.shape[0])]).T
ATA = np.dot(A.T, A)
# A_pseudo_inv = np.linalg.inv(ATA) @ A.T
A_pseudo_inv = np.linalg.pinv(A)

# Least-squares fit
parameters = np.dot(A_pseudo_inv, Measured_data)

slope = parameters[0]  # slope of the line
intercept = parameters[1]  # intercept of the line

# calculate fitted values
fitted_values = slope * Commanded_data + intercept

# Sum of Squared Errors
squared_errors = (Measured_data - fitted_values) ** 2
sum_of_squared_errors = np.sum(squared_errors)

plt.scatter(Commanded_data, Measured_data, color='blue', label='data', marker='x')
plt.plot(Commanded_data, fitted_values, color='red', label='fit_line')
plt.xlabel('Commanded Data')
plt.ylabel('Measured Data')
plt.legend()
plt.title('Least Squares Fit')
plt.grid()
plt.show()

print('The Slope is:', slope)
print('The Intercept is:', intercept)
print('The Sum of Squared Errors is:', sum_of_squared_errors)
