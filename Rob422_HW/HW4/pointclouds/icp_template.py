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


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target
    errors = []
    epsilon = 1e-5
    iterations = []
    max_iterations = 100
    pc_source_old = pc_source
    pc_source = numpy.array(pc_source).squeeze()
    pc_target = numpy.array(pc_target).squeeze()

    for iter in range(max_iterations):
        iterations.append(iter)

        correspondence = get_correspondence(pc_source, pc_target)
        R, t = get_transform(pc_source, correspondence)
        errors.append(compute_error(R, t, pc_source, correspondence))

        if errors[-1] < epsilon:
            break

        pc_source = (R @ pc_source.T).T + t

    pc_source = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc_source]
    pc_target = [numpy.asmatrix(pcs.reshape(3, 1)) for pcs in pc_target]

    fig_error = plt.figure()
    error = fig_error.add_subplot(111)
    error.plot(iterations, errors, linewidth=2)
    error.set_xlabel('Number of Iteration')
    error.set_ylabel('Error')

    utils.view_pc([pc_source_old, pc_source, pc_target], None, ['g', 'b', 'r'], ['o', 'o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])


    print(pc_source)
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
