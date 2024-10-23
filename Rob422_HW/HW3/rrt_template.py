import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, \
    set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
import time
import pybullet
from utils import draw_sphere_marker
import imageio

random.seed(1)
STEP_SIZE = 0.05
GOAL_BIAS = 0.1
MAX_ITER = 150
DIST_THRESHOLD = 1e-4


class Node:
    def __init__(self, config, parent=None):
        self.config = np.array(config)
        self.parent = parent


class RRT:
    def __init__(self, start_node, goal_node, collision_fn, joint_limits, step_size=STEP_SIZE, goal_bias=GOAL_BIAS,
                 max_iter=MAX_ITER):
        self.start_node = start_node
        self.goal_node = goal_node
        self.collision = collision_fn
        self.joint_limits = joint_limits
        self.goal_bias = goal_bias
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start_node]

    def random_sample(self):
        if np.random.random() < self.goal_bias:
            return self.goal_node
        else:
            sample_node = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
            return Node(sample_node)

    def nearest(self, node_list, node):
        dist = []
        # weights = np.array([0.035, 0.6, 0.15, 0.15, 0.03, 0.035])
        weights = np.array([1, 1, 1, 1, 1, 1])
        for nodes in node_list:
            diff = nodes.config - node.config
            # for d in range(diff.shape[0]):
            #     diff[d] = min(abs(diff[d]), 2*np.pi-abs(diff[d]))
            diff[4] = min(abs(diff[4]), 2 * np.pi - abs(diff[4]))
            dist.append(np.linalg.norm(diff @ weights))
            # dist.append(np.linalg.norm(diff))
        return node_list[np.argmin(dist)]

    def extend(self, nearest_node, sample_node):
        if np.linalg.norm(nearest_node.config - sample_node.config) < self.step_size:
            return sample_node

        direction = sample_node.config - nearest_node.config
        direction /= np.linalg.norm(direction)

        new_position = nearest_node.config + direction * self.step_size
        clip_position = np.clip(new_position, self.joint_limits[:, 0], self.joint_limits[:, 1])
        new_node = Node(clip_position)

        return new_node

    def connect(self):
        sample_node = self.random_sample()
        nearest_node = self.nearest(self.node_list, sample_node)
        prev_node = nearest_node

        while True:
            new_node = self.extend(prev_node, sample_node)

            if self.collision(new_node.config):
                break

            new_node.parent = prev_node
            self.node_list.append(new_node)
            if np.linalg.norm(new_node.config - sample_node.config) < DIST_THRESHOLD:
                break

            prev_node = new_node

    def shortcut_smoothing(self, path):
        for i in range(self.max_iter):
            points = np.sort(np.random.choice(len(path), 2, replace=False))
            node_one = path[points[0]]
            node_two = path[points[1]]
            prev_node = node_one
            access = True
            new_node_list = []

            while True:
                new_node = self.extend(prev_node, node_two)

                if self.collision(new_node.config):
                    access = False
                    break

                new_node.parent = prev_node
                new_node_list.append(new_node)
                if np.linalg.norm(new_node.config - node_two.config) < 1e-4:
                    break

                prev_node = new_node

            if not access:
                continue

            smooth_path = []
            smooth_path = path[:points[0] + 1]
            smooth_path += new_node_list
            if points[1] + 1 < len(path):
                path[points[1] + 1].parent = smooth_path[-1]
                smooth_path += path[points[1] + 1:]
            path = smooth_path

        return smooth_path

    def reconstruct_path_config(self, smooth_path):
        path = []
        current_node = smooth_path[-1]
        while current_node.parent is not None:
            path.append(current_node.config)
            current_node = current_node.parent
        path.append(smooth_path[0].config)

        return path[::-1]

    def reconstruct_path_node(self):
        path = []
        current_node = self.node_list[-1]
        while current_node.parent is not None:
            path.append(current_node)
            current_node = current_node.parent
        path.append(self.start_node)

        return path[::-1]


joint_names = (
    'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_upper_arm_roll_joint',
    'l_forearm_roll_joint',
    'l_wrist_flex_joint')


def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names = ('l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_upper_arm_roll_joint',
                   'l_forearm_roll_joint', 'l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i]: (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit,
                                     get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in
                    range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    start_node = Node(start_config)
    goal_node = Node(goal_config)
    joint_limit = np.array([[value[0], value[1]] for value in joint_limits.values()])
    joint_limit[4] = [-np.pi, np.pi]

    start_time = time.time()
    rrt = RRT(start_node, goal_node, collision_fn, joint_limit)
    while True:
        rrt.connect()
        if np.linalg.norm(rrt.node_list[-1].config - goal_node.config) < 1e-4:
            path = rrt.reconstruct_path_node()
            print("Planner run time: ", time.time() - start_time)
            break
        if (time.time() - start_time) > 600:
            print("No Solution Found.")
            break

    if path:
        PR2 = robots['pr2']
        for p in path:
            set_joint_positions(PR2, joint_idx, p.config)
            ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
            draw_sphere_marker((ee_pose[0][0], ee_pose[0][1], ee_pose[0][2]), 0.02, (1, 0, 0, 1))

        smooth_path = rrt.shortcut_smoothing(path)
        for sp in smooth_path:
            set_joint_positions(PR2, joint_idx, sp.config)
            ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
            draw_sphere_marker((ee_pose[0][0], ee_pose[0][1], ee_pose[0][2]), 0.02, (0, 0, 1, 1))

    ######################
    # Execute planned path
    reconstruct_path = rrt.reconstruct_path_config(path)
    execute_trajectory(robots['pr2'], joint_idx, reconstruct_path, sleep=0.1)

    reconstruct_smooth_path = rrt.reconstruct_path_config(smooth_path)
    execute_trajectory(robots['pr2'], joint_idx, reconstruct_smooth_path, sleep=0.1)

    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
