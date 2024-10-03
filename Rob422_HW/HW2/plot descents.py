import numpy as np
import matplotlib.pyplot as plt

from newtonsmethod import newton_method
from gradientdescent import gradient_descent

input_x = np.linspace(-10, 10, 100)


def f(x):
    return np.exp(0.5 * x + 1) + np.exp(-0.5 * x - 0.5) + 5 * x


def grad_f(x):
    return 0.5 * np.exp(0.5 * x + 1) - 0.5 * np.exp(-0.5 * x - 0.5) + 5


def hessian_f(x):
    return 0.25 * np.exp(0.5 * x + 1) + 0.25 * np.exp(-0.5 * x - 0.5)


output = []
for x in input_x:
    output.append(f(x))
plt.plot(input_x, output, label='$f(x) = e^{(0.5x+1)} + e^{(-0.5x-0.5)} + 5x$')

# Gradient Descent
x_init = 5
x_opt, points = gradient_descent(f, grad_f, x_init, alpha=0.1, beta=0.6, epsilon=0.0001, max_iter=1000)
f_opt = f(x_opt)
points_y = [f(x) for x in points]

# Newton's Method
x_opt_newton, points_newton = newton_method(f, grad_f, hessian_f, x_init, alpha=0.1, beta=0.6, epsilon=0.0001,
                                            max_iter=1000)
f_opt_newton = f(x_opt_newton)
points_newton_y = [f(x) for x in points_newton]

plt.plot(points, points_y, marker='o', color='red', linestyle='-', label='Descent Path')
plt.plot(points_newton, points_newton_y, marker='o', color='magenta', linestyle='-', label='Newton\'s Method')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent and Newton\'s Method')
plt.show()

plt.plot(range(len(points_y)), points_y, marker='o', color='red', linestyle='-', label='Gradient Descent')
plt.plot(range(len(points_newton_y)), points_newton_y, marker='o', color='magenta', linestyle='-', label='Newton\'s Method')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Gradient Descent and Newton\'s Method')
plt.show()