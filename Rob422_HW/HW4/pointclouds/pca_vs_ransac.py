#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from ransac_template import RANSAC, calculate_error
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    pca_error = []
    ransac_error = []
    number_outliers = []
    pca_time = []
    ransac_time = []
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        fig = utils.view_pc([pc])

        ###YOUR CODE HERE###
        pc = numpy.array(pc).squeeze()
        number_outliers.append((i + 1) * 10)

        # PCA
        pca_start = time.time()
        pca_mean = numpy.mean(pc, axis=0)
        pca_centered = pc - pca_mean
        Q = numpy.cov(pca_centered, rowvar=False)
        _, _, Vt = numpy.linalg.svd(Q)
        normal_pca = Vt[-1]

        threshold = 1e-2
        pca_inliers = pc[calculate_error(pc, normal_pca, pca_mean, keep_elem=True) < threshold]
        pca_error.append(calculate_error(pca_inliers, normal_pca, pca_mean))
        pca_outliers = numpy.array([point for point in pc if point not in pca_inliers])
        pca_end = time.time()
        pca_time.append(pca_end - pca_start)

        # RANSAC
        ransac_start = time.time()
        (normal_ransac, ransac_mean), ransac_inliers, error = RANSAC(pc, iter=1500, threshold=1e-2, N=150)
        ransac_end = time.time()
        ransac_error.append(error)
        ransac_outliers = numpy.array([point for point in pc if point not in ransac_inliers])
        ransac_time.append(ransac_end - ransac_start)

        pc = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc]
        if i == (num_tests - 1):
            # Show the resulting point cloud for PCA
            fig_pca = plt.figure()
            pca = fig_pca.add_subplot(111, projection='3d')
            pca.scatter(pca_inliers[:, 0], pca_inliers[:, 1], pca_inliers[:, 2], color='r', s=30)
            pca.scatter(pca_outliers[:, 0], pca_outliers[:, 1], pca_outliers[:, 2], color='b', s=30)
            pca.set_title("PCA")
            pca.set_xlabel('X')
            pca.set_ylabel('Y')
            pca.set_zlabel('Z')
            fig_pca = utils.draw_plane(fig_pca, numpy.asmatrix(normal_pca.reshape(3, 1)),
                                       numpy.asmatrix(pca_mean.reshape(3, 1)), (0.1, 0.7, 0.1, 0.5), width=[-0.5, 1])

            for angle in range(60, 120):
                pca.view_init(20, angle)
                plt.draw()
                plt.pause(.001)
                plt.savefig("{}.png".format(angle))

            # Show the resulting point cloud for RANSAC
            fig_ransac = plt.figure()
            ransac = fig_ransac.add_subplot(111, projection='3d')
            ransac.scatter(ransac_inliers[:, 0], ransac_inliers[:, 1], ransac_inliers[:, 2], color='r', s=30)
            ransac.scatter(ransac_outliers[:, 0], ransac_outliers[:, 1], ransac_outliers[:, 2], color='b', s=30)
            ransac.set_title("RANSAC")
            ransac.set_xlabel('X')
            ransac.set_ylabel('Y')
            ransac.set_zlabel('Z')
            fig_ransac = utils.draw_plane(fig_ransac, numpy.asmatrix(normal_ransac.reshape(3, 1)),
                                          numpy.asmatrix(ransac_mean.reshape(3, 1)), (0.1, 0.7, 0.1, 0.5),
                                          width=[-0.5, 1])

            # Show the error vs outliers
            fig_error = plt.figure()
            error = fig_error.add_subplot(111)
            error.plot(number_outliers, pca_error, label="PCA error", linewidth=2)
            error.plot(number_outliers, ransac_error, label="RANSAC error", linewidth=2)
            error.set_xticks(number_outliers)
            error.set_xlabel('Number of Outliers')
            error.set_ylabel('Errors')
            error.legend()
            fig_error.show()

            numpy.set_printoptions(precision=5, suppress=True)
            print("Computation times of PCA: ", pca_time)
            print("Average computation times of PCA: ", numpy.mean(pca_time))
            print("Computation times of RANSAC: ", ransac_time)
            print("Average computation times of RANSAC: ", numpy.mean(ransac_time))

            input("Press enter to close plots.")

        # this code is just for viewing, you can remove or change it
        # input("Press enter for next test:")
        plt.close('all')
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
