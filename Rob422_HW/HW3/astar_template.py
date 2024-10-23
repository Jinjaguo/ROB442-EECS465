import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, \
    joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue
import pybullet as p


#########################
def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def draw_spheres_in_batch(spheres_data):
    for data in spheres_data:
        position, radius, color = data
        vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)


def get_neighbors(node, mode=4):
    dx, dy, dtheta = 0.1, 0.1, np.pi / 2
    if mode == 4:
        moves = [(dx, 0, 0), (0, dy, 0), (-dx, 0, 0), (0, -dy, 0), (0, 0, dtheta), (0, 0, -dtheta)]
    if mode == 8:
        moves = [(dx, 0, 0), (dx, 0, dtheta), (dx, 0, -dtheta), (0, dy, 0), (0, dy, dtheta), (0, dy, -dtheta),
                 (-dx, 0, 0), (-dx, 0, dtheta), (-dx, 0, -dtheta), (0, -dy, 0), (0, -dy, dtheta), (0, -dy, -dtheta),
                 (0, 0, dtheta), (0, 0, -dtheta), (dx, dy, 0), (dx, dy, dtheta), (dx, dy, -dtheta),
                 (-dx, dy, 0), (-dx, dy, dtheta), (-dx, dy, -dtheta), (dx, -dy, 0), (dx, -dy, dtheta),
                 (dx, -dy, -dtheta),
                 (-dx, -dy, 0), (-dx, -dy, dtheta), (-dx, -dy, -dtheta)]  # diagonal actions
    return [(node[0] + move[0], node[1] + move[1], wrap_to_pi(node[2] + move[2])) for move in moves]


def cost(node, goal):
    dtheta = abs(wrap_to_pi(node[2] - goal[2]))
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2 + min(dtheta, 2 * np.pi - dtheta) ** 2)


def reconstruct_path(close_list, current_node):
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = close_list.get(current_node)
    return path[::-1]


def a_star(start, goal, collision_fn):
    open_list = PriorityQueue()
    open_list.put((0, start))
    gcosts = {start: 0}
    fcosts = {start: cost(start, goal)}
    search_list = {}
    close_list = set()
    collision_list = set()
    collision_free_list = set()

    while not open_list.empty():
        _, current_node = open_list.get()

        if current_node in close_list:
            continue

        if collision_fn(current_node):
            collision_list.add(current_node)
            continue

        close_list.add(current_node)
        collision_free_list.add(current_node)

        if np.allclose(current_node[:2], goal[:2], atol=1e-4) and np.allclose(wrap_to_pi(current_node[2]),
                                                                              wrap_to_pi(goal[2]), atol=1e-4):
            return gcosts[current_node], collision_list, collision_free_list, reconstruct_path(search_list,
                                                                                               current_node)

        for neighbor in get_neighbors(current_node):
            proposed_gcost = gcosts[current_node] + cost(current_node, neighbor)

            if neighbor in close_list or (neighbor in gcosts and proposed_gcost >= gcosts[neighbor]):
                continue

            search_list[neighbor] = current_node
            gcosts[neighbor] = proposed_gcost
            fcosts[neighbor] = proposed_gcost + cost(neighbor, goal)
            open_list.put((fcosts[neighbor], neighbor))

    return gcosts[current_node], collision_list, collision_free_list, None


def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi / 2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    draw_graph = True
    g_cost, collision_list, collision_free_list, path = a_star(start_config, goal_config, collision_fn)

    if not path:
        print("No Solution Found.")
    if path:
        print("Path cost: ", g_cost)

    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)

    if draw_graph:
        spheres_to_draw = []

        #for collision in collision_list:
            #spheres_to_draw.append(((collision[0], collision[1], 1.0), 0.05, (1, 0, 0, 1)))

        #for collision_free in collision_free_list:
            #spheres_to_draw.append(((collision_free[0], collision_free[1], 1.0), 0.05, (0, 0, 1, 1)))

        for p in path:
            spheres_to_draw.append(((p[0], p[1], 1.1), 0.05, (0, 0, 0, 1)))

            # Step 5: Batch draw all spheres at once
        draw_spheres_in_batch(spheres_to_draw)

    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
