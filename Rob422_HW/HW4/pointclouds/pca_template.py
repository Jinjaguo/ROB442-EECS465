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
    # fig = utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    pc = numpy.array(pc).squeeze()
    pc_mean = numpy.mean(pc, axis=0)
    pc_centered = pc - pc_mean

    # 计算点云的协方差矩阵
    Q = numpy.cov(pc_centered, rowvar=False)
    # 旋转点云，使其与XY平面对齐
    U, S, Vt = numpy.linalg.svd(Q)
    normal = Vt[-1]
    # 旋转点云，使其与XY平面对齐
    pc_rotated = pc_centered @ Vt

    numpy.set_printoptions(precision=5, suppress=True)
    print("Transformation matrix V^T: ")
    print(Vt)

    # 绘制旋转后的点云
    fig_rotated = plt.figure()
    ax_rotated = fig_rotated.add_subplot(111, projection='3d')
    ax_rotated.scatter(pc_rotated[:, 0], pc_rotated[:, 1], pc_rotated[:, 2])
    ax_rotated.set_xlim([-1, 1])
    ax_rotated.set_ylim([-1, 1])
    ax_rotated.set_zlim([-1, 1])
    ax_rotated.set_xlabel('X')
    ax_rotated.set_ylabel('Y')
    ax_rotated.set_zlabel('Z')
    plt.title("Point Cloud Aligned with XY Plane")
    plt.savefig('PCA_Rotated_Point_Cloud_Aligned_with_XY_Plane.png')


    #Rotate the points to align with the XY plane AND eliminate the noise
    threshold = 1e-4
    s = S ** 2 # 奇异值的平方对应数据方差
    Vt_reduced = Vt.copy()
    Vt_reduced[s < threshold] = 0 # 过滤掉方差小于阈值的分量
    pc_reduced = pc_centered @ Vt_reduced.T # 只保留主要成分

    numpy.set_printoptions(precision=5, suppress=True)
    # 打印经过消噪处理后的转换矩阵Vt
    print("Transformation matrix V^T after elimination: ")
    print(Vt_reduced)

    # 绘制降维后的点云
    fig_reduced = plt.figure()
    ax_reduced = fig_reduced.add_subplot(111, projection='3d')
    ax_reduced.scatter(pc_reduced[:, 0], pc_reduced[:, 1], pc_reduced[:, 2])
    ax_reduced.set_xlim([-1, 1])
    ax_reduced.set_ylim([-1, 1])
    ax_reduced.set_zlim([-1, 1])
    ax_reduced.set_xlabel('X')
    ax_reduced.set_ylabel('Y')
    ax_reduced.set_zlabel('Z')
    plt.title("Noise-Reduced Point Cloud Aligned with XY Plane")
    # 保存图像
    plt.savefig('PCA_Rotated_and_Reduced_Point_Cloud_Aligned_with_XY_Plane.png')

    # 使用法向量和点云的均值定义平面
    normal = Vt[-1]  # 平面的法向量是Vt的最后一列
    plane_point = pc_mean  # 平面上的一个点是均值点

    # 绘制拟合的平面
    fig_plane = plt.figure()
    ax_plane = fig_plane.add_subplot(111, projection='3d')
    ax_plane.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='b', s=1)
    xx, yy = numpy.meshgrid(numpy.linspace(-1, 1, 10), numpy.linspace(-1, 1, 10))
    # 计算平面上的z值
    zz = (-normal[0] * (xx - plane_point[0]) - normal[1] * (yy - plane_point[1])) / normal[2] + plane_point[2]
    ax_plane.plot_surface(xx, yy, zz, color='g', alpha=0.5)

    ax_plane.set_xlabel('X')
    ax_plane.set_ylabel('Y')
    ax_plane.set_zlabel('Z')
    plt.title("Fitted Plane in Point Cloud")
    plt.savefig('PCA_Fitted_Plane_in_Point_Cloud.png')

    # Show the resulting point cloud
    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
