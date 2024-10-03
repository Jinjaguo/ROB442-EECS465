import numpy as np


# a. Implement backtracking line search for functions of the form f(x) : R → R. Set
# α = 0.1 and β = 0.6. Submit your code as backtracking.py in your zip file.

def backtracking_line_search(f, grad_f, x, delta_x, alpha=0.1, beta=0.6):
    t = 1  # initialize step size

    f_x = f(x)
    grad_f_x = grad_f(x)

    while f(x + t * delta_x) > f_x + alpha * t * np.dot(grad_f_x.T, delta_x):
        t *= beta  # update step size
    return t
