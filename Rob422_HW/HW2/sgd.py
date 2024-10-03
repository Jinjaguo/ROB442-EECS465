import numpy as np
import matplotlib.pyplot as plt
import time

from SGDtest import fi, fsum, fiprime, fsumprime, fiprimeprime, fsumprimeprime

maxi = 10000


def sgd(x_init, fiprime, t=1, maxi=10000, batch_size=16, iterations=1000):
    x = x_init
    points = [x]
    for i in range(iterations):
        # choose one random ∇fi per iteration
        indices = np.random.randint(1, maxi, size=batch_size)
        grads = [fiprime(x, j) for j in indices]
        grad = np.mean(grads, axis=0)
        delta_x = -t * grad  # negative gradient direction

        # update x
        x = x + delta_x
        points.append(x)

        # check stopping condition
        if np.linalg.norm(grads) < 0.0001:
            break

    return x, points


start = time.time()
batch_size = 1
x, points = sgd(-5, fiprime, t=1, maxi=maxi, batch_size=batch_size, iterations=1000)
end = time.time()
print("Time: ", end - start, 'for batch size', batch_size)

# show the plot of fsum(x) vs. x
xvals = np.arange(-10, 10, 0.01)  # Grid of 0.01 spacing from -10 to 10
yvals = fsum(xvals)  # Evaluate function on xvals
plt.figure()
plt.plot(xvals, yvals)  # Create line plot with yvals against xvals
plt.xlabel('x')
plt.ylabel('fsum(x)')
plt.title('fsum(x) vs. x')
plt.show()
