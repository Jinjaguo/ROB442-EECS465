#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt

def fit_plane(p):
    p_mean = numpy.mean(p, axis=0)
    p_centered = p - p_mean
    _, _, Vt = numpy.linalg.svd(p_centered)
    normal = Vt[-1]

    # Q = numpy.cov(p_centered, rowvar=False)
    # eigen_values, eigen_vectors = numpy.linalg.eig(Q)
    # idx = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[idx]
    # eigen_vectors = eigen_vectors[:, idx]

    # normal = eigen_vectors[:, numpy.argmin(eigen_values)]

    return normal, p_mean


def calculate_error(p, normal, p_mean, keep_elem=False):
    # normal /= numpy.linalg.norm(normal)
    # offset = -numpy.dot(normal, p_mean)
    # error = numpy.abs(numpy.dot(normal, p) + offset)
    error = numpy.abs((p - p_mean) @ normal) / numpy.linalg.norm(normal)
    if keep_elem:
        return error ** 2
    return numpy.sum(error ** 2)


def RANSAC(pcs, iter, threshold, N):
    best_model = None
    best_inliers = None
    best_normal = None
    best_mean = None
    best_error = float('inf')

    for i in range(iter):
        idx = numpy.random.choice(pcs.shape[0], 3, replace=False)
        R = pcs[idx]
        normal, p_mean = fit_plane(R)

        pc = numpy.delete(pcs, idx, axis=0)
        C = pc[calculate_error(pc, normal, p_mean, keep_elem=True) < threshold]

        if C.shape[0] > N:
            points = numpy.vstack((R, C))
            normal, p_mean = fit_plane(points)

            error_new = calculate_error(points, normal, p_mean)
            if error_new < best_error:
                best_error = error_new
                best_normal = normal
                best_mean = p_mean
                best_model = (best_normal, best_mean)

    best_inliers = pcs[calculate_error(pcs, best_normal, best_mean, keep_elem=True) < threshold]
    best_error = calculate_error(best_inliers, best_normal, best_mean)

    return best_model, best_inliers, best_error


def plane_equation(normal, p_mean):
    normal = normal / numpy.linalg.norm(normal)
    A = normal[0]
    B = normal[1]
    C = normal[2]
    D = -numpy.dot(normal, p_mean)

    return A, B, C, D
def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])

    #Fit a plane to the data using ransac
    N = 150
    iter = 1500
    threshold = 1e-2
    pc = numpy.array(pc).squeeze()

    (normal, p_mean), best_inliers, _ = RANSAC(pc, iter, threshold, N)
    outliers = numpy.array([point for point in pc if point not in best_inliers])
    A, B, C, D = plane_equation(normal, p_mean)
    print("The equation of the plane is: {:.2f}x + {:.2f}y + {:.2f}z = {:.2f}".format(A, B, C, D))

    #Show the resulting point cloud
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_inliers[:, 0], best_inliers[:, 1], best_inliers[:, 2], color='r', s=30)
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='b', s=30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #Draw the fitted plane
    fig = utils.draw_plane(fig, numpy.asmatrix(normal.reshape(3, 1)), numpy.asmatrix(p_mean.reshape(3, 1)),
                           (0.1, 0.7, 0.1, 0.5), [-0.5, 1], [-0.5, 1])


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
