#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt


def get_correspondence(pc_source, pc_target):
    m, n = pc_source.shape
    pc_source_expand = pc_source.reshape(m, 1, n)
    distance = numpy.linalg.norm(pc_source_expand - pc_target, axis=2)
    idx = numpy.argmin(distance, axis=1)
    correspondence = pc_target[idx]

    return correspondence


def get_transform(pc_source, pc_target):
    mean_source = numpy.mean(pc_source, axis=0)
    mean_target = numpy.mean(pc_target, axis=0)

    center_source = pc_source - mean_source
    center_target = pc_target - mean_target

    cov = center_source.T @ center_target
    U, _, Vt = numpy.linalg.svd(cov)
    S = numpy.eye(3)
    S[2, 2] = numpy.linalg.det(Vt.T @ U.T)
    R = Vt.T @ S @ U.T
    t = mean_target - R @ mean_source

    return R, t


def compute_error(R, t, pc_source, pc_target):
    error = numpy.sum(((R @ pc_source.T).T + t - pc_target) ** 2)
    return error


def icp(pc_source, pc_target, max_iterations=100, epsilon=1e-5):
    errors = []
    iterations = []
    pc_source_old = pc_source
    for iter in range(max_iterations):
        iterations.append(iter)

        correspondence = get_correspondence(pc_source, pc_target)
        R, t = get_transform(pc_source, correspondence)
        errors.append(compute_error(R, t, pc_source, correspondence))

        if errors[-1] < epsilon:
            break

        pc_source = (R @ pc_source.T).T + t

    return pc_source, errors, iterations


def main():
    # Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    target_files = 'cloud_icp_target3.csv'
    pc_target = utils.load_pc(target_files)  # Change this to load in a different target
    epsilon = 1e-5
    max_iterations = 100
    pc_source_old = pc_source
    pc_source = numpy.array(pc_source).squeeze()
    pc_target = numpy.array(pc_target).squeeze()

    icp_result = icp(pc_source, pc_target, max_iterations, epsilon)
    pc_source = icp_result[0]
    errors = icp_result[1]
    iterations = icp_result[2]

    pc_source = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc_source]
    pc_target = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc_target]

    fig1 = plt.figure()
    plt.title(f'ICP Error vs. Iterations')
    plt.plot(iterations, errors, linewidth=2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error')
    plt.grid()
    fig1.savefig(f'error_vs_iterations_{target_files}.png')
    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_title('ICP Alignment Result')
    ax.scatter([p[0] for p in pc_target], [p[1] for p in pc_target], [p[2] for p in pc_target], c='r', marker='^', label='Target')
    ax.scatter([p[0] for p in pc_source_old], [p[1] for p in pc_source_old], [p[2] for p in pc_source_old], c='g', marker='s', label='Initial')
    ax.scatter([p[0] for p in pc_source], [p[1] for p in pc_source], [p[2] for p in pc_source], c='y', marker='o', label='Final')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 使用pc_target文件名作为图片名
    fig2.savefig(f'icp_alignment_result_{target_files}.png')

    # print(pc_source)
    ###YOUR CODE HERE###

    plt.show()
    # raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
