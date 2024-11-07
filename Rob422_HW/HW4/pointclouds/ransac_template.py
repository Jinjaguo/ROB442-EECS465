#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt

import numpy as np


def fit_plane(points):
    p_mean = np.mean(points, axis=0)
    p_centered = points - p_mean
    _, _, Vt = np.linalg.svd(p_centered)
    normal = Vt[-1]  # 最小主成分方向作为法向量
    return normal, p_mean


def calculate_error(points, normal, p_mean, keep_elem=False):
    error = np.abs((points - p_mean) @ normal) / np.linalg.norm(normal)
    if keep_elem:
        return error ** 2  # 返回逐点误差的平方
    return np.sum(error ** 2)  # 返回总误差的平方和


def RANSAC(points, iter, threshold, min_inliers):
    best_model, best_inliers, best_error = None, None, float('inf')

    for i in range(iter):
        # 随机选取三个点来定义一个平面
        sample_idx = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[sample_idx]
        normal, p_mean = fit_plane(sample_points)

        # 计算其他点相对于该平面的误差
        remaining_points = np.delete(points, sample_idx, axis=0)
        inliers = remaining_points[calculate_error(remaining_points, normal, p_mean, keep_elem=True) < threshold]

        # 如果内点数量足够多，重新拟合平面
        if inliers.shape[0] >= min_inliers:
            all_inliers = np.vstack((sample_points, inliers))
            refined_normal, refined_mean = fit_plane(all_inliers)
            error = calculate_error(all_inliers, refined_normal, refined_mean)

            # 更新最佳平面模型
            if error < best_error:
                best_model = (refined_normal, refined_mean)
                best_inliers = all_inliers
                best_error = error

    return best_model, best_inliers, best_error


def plane_equation(normal, p_mean):
    normal = normal / np.linalg.norm(normal)
    A, B, C = normal
    D = -np.dot(normal, p_mean)
    return A, B, C, D


def main():
    # Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    # fig = utils.view_pc([pc])

    # Fit a plane to the data using ransac
    min_inliers = 150
    max_iter = 1500
    threshold = 1e-2
    pc = np.array(pc).squeeze()

    # 使用RANSAC拟合平面
    (normal, p_mean), best_inliers, _ = RANSAC(pc, max_iter, threshold, min_inliers)
    outliers = numpy.array([point for point in pc if point not in best_inliers])
    A, B, C, D = plane_equation(normal, p_mean)
    print("The equation of the plane is: {:.2f}x + {:.2f}y + {:.2f}z = {:.2f}".format(A, B, C, D))

    # Show the resulting point cloud
    # 图像1: 显示点云的内点和外点
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_inliers[:, 0], best_inliers[:, 1], best_inliers[:, 2], color='r', s=30, label='Inliers')
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='b', s=30, label='Outliers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("RANSAC Point Cloud Fitting")
    plt.savefig('RANSAC_Point_Cloud_Fitting.png')  # 保存图像
    plt.show()

    # 图像2: 显示拟合的平面和点云
    fig2 = plt.figure()
    plt.title("RANSAC Plane Fitting")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='b', s=10, label='Original Points')
    ax2.scatter(best_inliers[:, 0], best_inliers[:, 1], best_inliers[:, 2], color='r', s=30, label='Inliers')
    fig2 = utils.draw_plane(fig2, np.asmatrix(normal.reshape(3, 1)), np.asmatrix(p_mean.reshape(3, 1)),
                            (0.1, 0.7, 0.1, 0.5), length=[-0.5, 1], width=[-0.5, 1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    fig2.savefig('RANSAC_Plane_Fitting.png')  # 保存图像
    plt.show()


    ###YOUR CODE HERE###
    # input("Press enter to end:")


if __name__ == '__main__':
    main()
