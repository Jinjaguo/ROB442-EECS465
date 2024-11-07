#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    pc = numpy.array(pc).squeeze()
    pc_mean = numpy.mean(pc, axis=0)
    pc_centered = pc - pc_mean

    Q = numpy.cov(pc_centered, rowvar=False)
    U, S, Vt = numpy.linalg.svd(Q)
    normal = Vt[-1]
    pc_rotated = pc_centered @ Vt

    numpy.set_printoptions(precision=5, suppress=True)
    print("Transformation matrix VT: ")
    print(Vt)

    #Show the resulting point cloud
    fig_rotated = plt.figure()
    ax_rotated = fig_rotated.add_subplot(111, projection='3d')
    ax_rotated.scatter(pc_rotated[:, 0], pc_rotated[:, 1], pc_rotated[:, 2])
    # ax_rotated.set_xlim([-1, 1])
    # ax_rotated.set_ylim([-1, 1])
    # ax_rotated.set_zlim([-1, 1])
    ax_rotated.set_xlabel('X')
    ax_rotated.set_ylabel('Y')
    ax_rotated.set_zlabel('Z')


    #Rotate the points to align with the XY plane AND eliminate the noise
    threshold = 1e-4
    s = S ** 2
    Vt_reduced = Vt.copy()
    Vt_reduced[s < threshold] = 0
    pc_reduced = pc_centered @ Vt_reduced.T

    numpy.set_printoptions(precision=5, suppress=True)
    print("Transformation matrix VT after elimination: ")
    print(Vt_reduced)

    # Show the resulting point cloud
    fig_reduced = plt.figure()
    ax_reduced = fig_reduced.add_subplot(111, projection='3d')
    ax_reduced.scatter(pc_reduced[:, 0], pc_reduced[:, 1], pc_reduced[:, 2])
    ax_reduced.set_xlabel('X')
    ax_reduced.set_ylabel('Y')
    ax_reduced.set_zlabel('Z')

    fig = utils.draw_plane(fig, numpy.asmatrix(normal.reshape(3, 1)), numpy.asmatrix(pc_mean.reshape(3, 1)),
                           (0.1, 0.7, 0.1, 0.5), length=[-1, 1], width=[-0.5, 1])

    # Show the resulting point cloud
    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
