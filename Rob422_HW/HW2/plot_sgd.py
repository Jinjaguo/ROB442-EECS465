import numpy as np
import matplotlib.pyplot as plt

from SGDtest import fi, fsum, fiprime, fsumprime, fiprimeprime, fsumprimeprime
from sgd import sgd

maxi = 10000
batch_size = 1
x, points = sgd(-5, fiprime, t=1, maxi=maxi, batch_size=batch_size, iterations=1000)

# plot fsum(x) vs. i
points_y = [fsum(x) for x in points]
print("the optimum is at:", x, "with fsum(x) =", fsum(x))
plt.plot(range(len(points_y)), points_y, marker='o', color='blue', linestyle='-', label='$fsum(x^{(i)})$', markevery=25)

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('$fsum(x^{(i)})$')
plt.title('Stochastic Gradient Descent')
plt.show()
