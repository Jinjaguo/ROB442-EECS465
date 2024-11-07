#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
###YOUR IMPORTS HERE###
from ransac_template import RANSAC, calculate_error
import numpy as np

###YOUR IMPORTS HERE###

def add_some_outliers(pc, num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc


def main():
    # Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    pca_error = []
    ransac_error = []
    number_outliers = []
    pca_time = []
    ransac_time = []
    for i in range(0, num_tests):
        pc = add_some_outliers(pc, 10)  # adding 10 new outliers for each test

        ###YOUR CODE HERE###
        pc = numpy.array(pc).squeeze()
        number_outliers.append((i + 1) * 10)

        # PCA
        pca_start = time.time()
        pca_mean = numpy.mean(pc, axis=0)
        pca_centered = pc - pca_mean
        Q = numpy.cov(pca_centered, rowvar=False)
        _, _, Vt = numpy.linalg.svd(Q)
        normal_pca = Vt[-1]  # 取出法向量

        threshold = 1e-2
        pca_inliers = pc[calculate_error(pc, normal_pca, pca_mean, keep_elem=True) < threshold]
        pca_error.append(calculate_error(pca_inliers, normal_pca, pca_mean))
        pca_outliers = numpy.array([point for point in pc if point not in pca_inliers])
        pca_end = time.time()
        pca_time.append(pca_end - pca_start)

        # RANSAC
        ransac_start = time.time()
        (normal_ransac, ransac_mean), ransac_inliers, error = RANSAC(pc, iter=1500, threshold=1e-2, min_inliers=150)
        ransac_end = time.time()
        ransac_error.append(error)
        ransac_outliers = numpy.array([point for point in pc if point not in ransac_inliers])
        ransac_time.append(ransac_end - ransac_start)

        # Convert the point cloud to matrices for plotting
        ### IMPROTANT ###
        pc = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc]

        if i == (num_tests - 1):
            # Show the resulting point cloud for PCA
            fig_pca = plt.figure()
            plt.title("PCA Plane Fitting")
            ax_pca = fig_pca.add_subplot(111, projection='3d')
            ax_pca.scatter(pca_inliers[:, 0], pca_inliers[:, 1], pca_inliers[:, 2], color='r', s=30, label='Inliers')
            ax_pca.scatter(pca_outliers[:, 0], pca_outliers[:, 1], pca_outliers[:, 2], color='b', s=30,
                           label='Outliers')
            utils.draw_plane(fig_pca, np.asmatrix(normal_pca.reshape(3, 1)), np.asmatrix(pca_mean.reshape(3, 1)),
                             color=(0.1, 0.7, 0.1, 0.5), width=[-0.5, 1])

            ax_pca.set_xlabel('X')
            ax_pca.set_ylabel('Y')
            ax_pca.set_zlabel('Z')
            ax_pca.legend()

            fig_pca.savefig('pca_plane_fitting.png')
            plt.show()



            # RANSAC 结果图
            fig_ransac = plt.figure()
            plt.title("RANSAC Plane Fitting")
            ax_ransac = fig_ransac.add_subplot(111, projection='3d')
            ax_ransac.scatter(ransac_inliers[:, 0], ransac_inliers[:, 1], ransac_inliers[:, 2], color='r', s=30,
                              label='Inliers')
            ax_ransac.scatter(ransac_outliers[:, 0], ransac_outliers[:, 1], ransac_outliers[:, 2], color='b', s=30,
                              label='Outliers')
            utils.draw_plane(fig_ransac, np.asmatrix(normal_ransac.reshape(3, 1)),
                             np.asmatrix(ransac_mean.reshape(3, 1)),
                             color=(0.1, 0.7, 0.1, 0.5), width=[-0.5, 1])
            ax_ransac.set_xlabel('X')
            ax_ransac.set_ylabel('Y')
            ax_ransac.set_zlabel('Z')
            ax_ransac.legend()
            fig_ransac.savefig('ransac_plane_fitting.png')
            plt.show()


            # 误差对比图
            fig_error = plt.figure()
            plt.plot(number_outliers, pca_error, label="PCA Error", linewidth=2, marker='o')
            plt.plot(number_outliers, ransac_error, label="RANSAC Error", linewidth=2, marker='o')
            plt.xlabel("Number of Outliers")
            plt.ylabel("Error")
            plt.title("Error vs. Number of Outliers")
            plt.legend()
            plt.grid()

            fig_error.savefig('pca_vs_ransac.png')
            plt.show()


            # 显示计算时间
            np.set_printoptions(precision=5, suppress=True)
            print("PCA Computation Times: ", pca_time)
            print("Average PCA Computation Time: ", np.mean(pca_time))
            print("RANSAC Computation Times: ", ransac_time)
            print("Average RANSAC Computation Time: ", np.mean(ransac_time))



        # this code is just for viewing, you can remove or change it
        # input("Press enter for next test:")
        plt.close('all')
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
