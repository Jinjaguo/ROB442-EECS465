import numpy as np
from backtracking import backtracking_line_search


def gradient_descent(f, grad_f, x_init, alpha=0.1, beta=0.6, epsilon=0.0001, max_iter=1000):
    x = x_init
    points = [x]
    for _ in range(max_iter):
        grad = grad_f(x)
        delta_x = -grad  # negative gradient direction

        # using backtracking line search to compute step length
        t = backtracking_line_search(f, grad_f, x, delta_x, alpha, beta)

        # update x
        x = x + t * delta_x
        points.append(x)

        # check stopping condition
        if np.linalg.norm(grad) < epsilon:
            break

    return x, points
