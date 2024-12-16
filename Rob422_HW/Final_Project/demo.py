import numpy as np
import time
import matplotlib.pyplot as plt
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, \
    joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from queue import PriorityQueue
import pybullet as p

#########################
# Utility Functions
#########################

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def draw_spheres_in_batch(spheres_data):
    """Draw multiple spheres in the simulation environment."""
    for data in spheres_data:
        position, radius, color = data
        vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

def get_neighbors(node, mode=8):
    """Generate neighboring nodes based on movement mode."""
    dx, dy, dtheta = 0.1, 0.1, np.pi / 2
    moves = []
    if mode == 4:
        moves = [
            (dx, 0, 0), (0, dy, 0), (-dx, 0, 0), (0, -dy, 0),
            (0, 0, dtheta), (0, 0, -dtheta)
        ]
    elif mode == 8:
        moves = [
            (dx, 0, 0), (dx, 0, dtheta), (dx, 0, -dtheta),
            (0, dy, 0), (0, dy, dtheta), (0, dy, -dtheta),
            (-dx, 0, 0), (-dx, 0, dtheta), (-dx, 0, -dtheta),
            (0, -dy, 0), (0, -dy, dtheta), (0, -dy, -dtheta),
            (0, 0, dtheta), (0, 0, -dtheta),
            (dx, dy, 0), (dx, dy, dtheta), (dx, dy, -dtheta),
            (-dx, dy, 0), (-dx, dy, dtheta), (-dx, dy, -dtheta),
            (dx, -dy, 0), (dx, -dy, dtheta), (dx, -dy, -dtheta),
            (-dx, -dy, 0), (-dx, -dy, dtheta), (-dx, -dy, -dtheta)
        ]
    return [
        (node[0] + move[0], node[1] + move[1], wrap_to_pi(node[2] + move[2]))
        for move in moves
    ]

#########################
# Control and State Estimation Functions (From Version 2)
#########################

def compute_control(current_pose, target_pose):
    """
    Computes control inputs (linear velocity v, angular velocity w) 
    to move from the current pose to the target pose.
    """
    x, y, theta = current_pose
    x_g, y_g, _ = target_pose  # Unpack x, y, and ignore orientation theta

    # Compute control inputs based on position differences
    position_diff = np.array([x_g - x, y_g - y])
    distance = np.linalg.norm(position_diff)  # Linear distance
    angle_to_target = np.arctan2(y_g - y, x_g - x)  # Angle to the target

    # Compute angular velocity (heading difference)
    heading_error = wrap_to_pi(angle_to_target - theta)

    # Return control inputs: linear velocity v, angular velocity w
    v = 5 * distance  # Adjust scaling factor as needed
    w = 2 * heading_error  # Adjust scaling factor as needed

    return v, w

def Odometry(control_input, previous_pose):
    """
    Simulates odometry by updating the pose based on control inputs.
    """
    v, w = control_input  # Linear and angular velocities
    delta_t = 0.1  # Time step
    x, y, theta = previous_pose
    delta_x = v * delta_t * np.cos(theta)
    delta_y = v * delta_t * np.sin(theta)
    delta_theta = w * delta_t
    return np.array([x + delta_x, y + delta_y, wrap_to_pi(theta + delta_theta)])

def SensorModel(control_input, robots, base_joints, previous_pose):
    """
    Simulates sensor measurements with noise.
    """
    v, w = control_input
    delta_t = 0.1  # Time step
    delta_x = v * delta_t * np.cos(previous_pose[2])
    delta_y = v * delta_t * np.sin(previous_pose[2])
    delta_theta = w * delta_t
    ground_truth = np.array([previous_pose[0] + delta_x, previous_pose[1] + delta_y, wrap_to_pi(previous_pose[2] + delta_theta)])
    k = 0.05  # Sensor noise covariance
    sensor_cov = np.array([[k, 0, 0], [0, k, 0], [0, 0, k]])
    noisy_reading = np.random.multivariate_normal(ground_truth, sensor_cov)
    return noisy_reading, ground_truth

def heuristic_particle_filter(node, goal, particles=100, robots=None, base_joints=None, odometry_fn=None, sensor_fn=None):
    """
    Heuristic based on a particle filter estimation.
    """
    if odometry_fn is None or sensor_fn is None:
        raise ValueError("Odometry and SensorModel functions must be provided.")

    particles_states = np.tile(np.array(node), (particles, 1))

    controls = compute_control(node, goal)  # Compute control inputs based on current node and goal
    noisy_particles = []
    for particle in particles_states:
        new_pose = odometry_fn(controls, particle)
        noisy_pose, _ = sensor_fn(controls, robots, base_joints, new_pose)
        noisy_particles.append(noisy_pose)

    estimated_costs = [
        np.linalg.norm([goal[0] - noisy_pose[0], goal[1] - noisy_pose[1]]) +
        abs(wrap_to_pi(goal[2] - noisy_pose[2]))
        for noisy_pose in noisy_particles
    ]

    return np.mean(estimated_costs)

#########################
# Heuristic Functions
#########################

def heuristic_eu(node, goal):
    """Euclidean distance heuristic."""
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

def heuristic_eu_modified(node, goal):
    """Modified Euclidean distance heuristic that includes orientation."""
    dtheta = abs(wrap_to_pi(node[2] - goal[2]))
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2 + min(dtheta, 2 * np.pi - dtheta) ** 2)

#########################
# Path Reconstruction
#########################

def reconstruct_path(parent_map, current_node):
    """Reconstructs the path from start to goal using the parent map."""
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = parent_map.get(current_node)
    return path[::-1]

#########################
# A* Implementation
#########################

def a_star(start, goal, collision_fn, heuristic_fn=heuristic_eu, time_limit=200.0):
    """
    A* search algorithm.
    """
    start_time = time.time()
    open_list = PriorityQueue()
    open_list.put((0, start))
    gcosts = {start: 0}
    fcosts = {start: heuristic_fn(start, goal)}
    search_list = {}
    close_list = set()
    collision_list = set()
    collision_free_list = set()

    best_cost_progress = []
    all_solutions_found = []  # Store all discovered solutions (time, cost, path)

    solution_cost_over_time = []  # Track solution cost over time
    last_recorded_time = 0  # To control recording frequency

    while not open_list.empty() and (time.time() - start_time) < time_limit:
        _, current_node = open_list.get()

        if current_node in close_list:
            continue

        if collision_fn(current_node):
            collision_list.add(current_node)
            continue

        close_list.add(current_node)
        collision_free_list.add(current_node)

        # Record cost at significant intervals
        current_time = time.time() - start_time
        if current_time - last_recorded_time >= 0.1:  # Record every 0.1 seconds
            solution_cost_over_time.append((current_time, gcosts[current_node]))
            last_recorded_time = current_time

        # Check goal
        if (np.allclose(current_node[:2], goal[:2], atol=1e-4) and
                np.allclose(wrap_to_pi(current_node[2]), wrap_to_pi(goal[2]), atol=1e-4)):
            path = reconstruct_path(search_list, current_node)
            path_cost = gcosts[current_node]
            current_time = time.time() - start_time
            best_cost_progress.append((current_time, path_cost))
            all_solutions_found.append((current_time, path_cost, path))
            solution_cost_over_time.append((current_time, path_cost))
            return path_cost, collision_list, collision_free_list, path, best_cost_progress, all_solutions_found, solution_cost_over_time

        # Explore neighbors
        for neighbor in get_neighbors(current_node):
            step_cost = heuristic_fn(current_node, neighbor)
            proposed_gcost = gcosts[current_node] + step_cost

            if neighbor in close_list or (neighbor in gcosts and proposed_gcost >= gcosts[neighbor]):
                continue

            search_list[neighbor] = current_node
            gcosts[neighbor] = proposed_gcost
            fcosts[neighbor] = proposed_gcost + heuristic_fn(neighbor, goal)
            open_list.put((fcosts[neighbor], neighbor))

    # No solution found
    return None, collision_list, collision_free_list, None, best_cost_progress, all_solutions_found, solution_cost_over_time

#########################
# ANA* Implementation
#########################

def ana_star(start, goal, collision_fn, heuristic_fn=heuristic_eu, time_limit=200.0):
    """
    Anytime Nonparametric A* (ANA*) search algorithm.
    """
    start_time = time.time()
    open_list = PriorityQueue()
    open_list.put((0, start))
    gcosts = {start: 0}
    fcosts = {start: heuristic_fn(start, goal)}
    search_list = {}
    close_list = set()
    collision_list = set()
    collision_free_list = set()

    best_solution = None
    best_cost = float('inf')
    best_cost_progress = []  # Improvements
    all_solutions_found = []  # Store all discovered solutions (time, cost, path)

    solution_cost_over_time = []  # Track solution cost over time
    last_recorded_time = 0  # To control recording frequency

    first_solution_found = False  # Flag to check if first solution is found

    while not open_list.empty() and (time.time() - start_time) < time_limit:
        _, current_node = open_list.get()

        if current_node in close_list:
            continue

        if collision_fn(current_node):
            collision_list.add(current_node)
            continue

        close_list.add(current_node)
        collision_free_list.add(current_node)

        # Record cost at significant intervals
        current_time = time.time() - start_time
        current_cost = gcosts[current_node]
        if current_time - last_recorded_time >= 0.1 or abs(current_cost - best_cost) > 0.1:
            solution_cost_over_time.append((current_time, current_cost))
            last_recorded_time = current_time

        # Check goal
        if (np.allclose(current_node[:2], goal[:2], atol=1e-4) and
                np.allclose(wrap_to_pi(current_node[2]), wrap_to_pi(goal[2]), atol=1e-4)):

            current_cost = gcosts[current_node]
            current_time = time.time() - start_time
            current_path = reconstruct_path(search_list, current_node)

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_path
                best_cost_progress.append((current_time, best_cost))
                all_solutions_found.append((current_time, best_cost, best_solution))
                solution_cost_over_time.append((current_time, best_cost))

                # If this is the first solution found, print the time and indicate optimization
                if not first_solution_found:
                    print(f"ANA* first found a solution at {current_time:.2f} seconds. Continuing to optimize...")
                    first_solution_found = True

        # Explore neighbors
        for neighbor in get_neighbors(current_node):
            step_cost = heuristic_fn(current_node, neighbor)
            proposed_gcost = gcosts[current_node] + step_cost

            if neighbor in close_list or (neighbor in gcosts and proposed_gcost >= gcosts[neighbor]):
                continue

            search_list[neighbor] = current_node
            gcosts[neighbor] = proposed_gcost
            fcosts[neighbor] = proposed_gcost + heuristic_fn(neighbor, goal)
            open_list.put((fcosts[neighbor], neighbor))

    # No solution found
    return best_cost if best_solution is not None else None, collision_list, collision_free_list, best_solution, best_cost_progress, all_solutions_found, solution_cost_over_time

#########################
# Visualization Functions
#########################

def moving_average(data, window_size=5):
    """Compute a moving average for smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(results, title, rows, cols):
    """
    Plots the solution cost over time for a set of results.
    Each subplot corresponds to a different method.
    """
    num_methods = len(results)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
    axs = axs.flatten()  # Flatten the axes array for easy indexing
    fig.suptitle(title)

    for idx, (name, path_cost, comp_time, path, cost_progress, all_solutions_found, solution_cost_over_time) in enumerate(results):
        ax = axs[idx]

        if solution_cost_over_time:
            times, costs = zip(*solution_cost_over_time)

            # Smooth data using moving average
            smoothed_costs = moving_average(costs, window_size=5)
            smoothed_times = times[:len(smoothed_costs)]

            # Plot raw and smoothed data
            ax.plot(times, costs, alpha=0.3, label="Raw Cost", color="gray")
            ax.plot(smoothed_times, smoothed_costs, label="Smoothed Cost", color="tab:blue")
            ax.grid(True, linestyle="--", alpha=0.5)

        # Mark only the best solution point
        if all_solutions_found:
            # The best solution is the one with the lowest cost
            best_solution = min(all_solutions_found, key=lambda x: x[1])
            ax.scatter(best_solution[0], best_solution[1], marker='X', color='green', s=100, label='Best Solution')

        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cost")
        ax.legend()

    # Hide any unused subplots
    for ax in axs[len(results):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#########################
# Path Visualization Function
#########################

def visualize_paths(results, colors, base_z=1.0, height_increment=0.05):
    """
    Visualizes the paths from different planning methods in PyBullet.
    Each method's path is shown in a different color and height.
    """
    spheres_to_draw = []

    for idx, (name, path_cost, comp_time, path, cost_progress, all_solutions_found, solution_cost_over_time) in enumerate(results):
        if path:
            color = colors[idx % len(colors)]
            z_offset = base_z + idx * height_increment
            for node in path:
                x, y, theta = node
                z = z_offset  # Assign different height for each method
                spheres_to_draw.append(((x, y, z), 0.02, color))  # Smaller radius for better visibility

    if spheres_to_draw:
        draw_spheres_in_batch(spheres_to_draw)

#########################
# Main Function
#########################

def main():
    # Connect to PyBullet with GUI
    connect(use_gui=True)
    
    # Load the environment
    robots, obstacles = load_env('env1.json')  # Change to 'env2.json' if needed
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    # Define start and goal configurations
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, 1, -np.pi / 2)

    # Set a time limit for the planning algorithms
    time_limit = 450.0  # Approximately 7.5 minutes, adjust to reach ~8.5 minutes if needed

    # Define planning scenarios
    scenarios = [
        ("A*_EU", a_star, heuristic_eu),
        ("ANA*_EU", ana_star, heuristic_eu),
        ("A*_EU_MOD", a_star, heuristic_eu_modified),
        ("ANA*_EU_MOD", ana_star, heuristic_eu_modified),
        ("A*_PF", a_star, lambda node, goal: heuristic_particle_filter(
            node, goal, particles=10, robots=robots, base_joints=base_joints,
            odometry_fn=Odometry, sensor_fn=SensorModel)),
        ("ANA*_PF", ana_star, lambda node, goal: heuristic_particle_filter(
            node, goal, particles=10, robots=robots, base_joints=base_joints,
            odometry_fn=Odometry, sensor_fn=SensorModel))
    ]
    wait_if_gui()
    # Define colors for each method in RGBA
    colors = [
        (1, 0, 0, 1),      # A*_EU: Red
        (0, 1, 0, 1),      # ANA*_EU: Green
        (0, 0, 1, 1),      # A*_EU_MOD: Blue
        (1, 0.5, 0, 1),    # ANA*_EU_MOD: Orange
        (0.5, 0, 0.5, 1),  # A*_PF: Purple
        (0, 1, 1, 1)       # ANA*_PF: Cyan
    ]

    # Informational Statement
    print("\n========================================")
    print("        Simulation Overview")
    print("========================================\n")
    print("Simulation Duration: Approximately 8.5 minutes\n")
    print("Planning Algorithms and Their Configurations:")
    for idx, (scenario, _, _) in enumerate(scenarios):
        color = colors[idx % len(colors)]
        color_name = ""
        # Assign color names based on RGBA values
        if color == (1, 0, 0, 1):
            color_name = "Red"
        elif color == (0, 1, 0, 1):
            color_name = "Green"
        elif color == (0, 0, 1, 1):
            color_name = "Blue"
        elif color == (1, 0.5, 0, 1):
            color_name = "Orange"
        elif color == (0.5, 0, 0.5, 1):
            color_name = "Purple"
        elif color == (0, 1, 1, 1):
            color_name = "Cyan"
        else:
            color_name = "Unknown"

        # Determine heuristic function
        heuristic_description = ""
        if "EU" in scenario:
            heuristic_description = "Euclidean"
        elif "PF" in scenario:
            heuristic_description = "Particle Filter"
        else:
            heuristic_description = "Unknown"

        # Determine algorithm type
        algo_type = "A*" if "A*" in scenario and "ANA*" not in scenario else "ANA*"

        print(f"{idx + 1}. {scenario}:")
        print(f"   - Algorithm Type: {algo_type}")
        print(f"   - Heuristic Function: {heuristic_description}")
        print(f"   - Path Color: {color_name}\n")

    print("Each algorithm's planned path will be visualized in the simulation environment with the specified color and height offset.")
    print("Press the window close button or terminate the GUI to start the simulation.\n")
    print("========================================\n")

    # Wait for user to read the statement before proceeding
    wait_if_gui()

    results = []

    for name, planner_fn, heuristic_fn in scenarios:
        print(f"Running {name} with time limit {time_limit}s...")
        start_time_planner = time.time()

        if "ANA*" in name:
            path_cost, collision_list, collision_free_list, best_solution, solutions_over_time, all_solutions_found, solution_cost_over_time = planner_fn(
                start_config, goal_config, collision_fn, heuristic_fn=heuristic_fn, time_limit=time_limit
            )
            comp_time = time.time() - start_time_planner

            if best_solution:
                print(f"{name} found a solution with cost: {path_cost:.4f}, Total Planning Time: {comp_time:.4f} s\n")
            else:
                print(f"{name} found no solution within {time_limit}s.\n")
            results.append((name, path_cost, comp_time, best_solution, solutions_over_time, all_solutions_found, solution_cost_over_time))
        else:
            path_cost, collision_list, collision_free_list, path, cost_progress, all_solutions_found, solution_cost_over_time = planner_fn(
                start_config, goal_config, collision_fn, heuristic_fn=heuristic_fn, time_limit=time_limit)
            comp_time = time.time() - start_time_planner

            if path:
                print(f"{name} found a solution with cost: {path_cost:.4f}, Time: {comp_time:.4f} s\n")
            else:
                print(f"{name} found no solution within {time_limit}s.\n")
            results.append((name, path_cost, comp_time, path, cost_progress, all_solutions_found, solution_cost_over_time))

    # Visualize all planned paths in the simulation
    visualize_paths(results, colors, base_z=1.0, height_increment=0.05)

    # Separate results into A* and ANA*
    a_star_results = [res for res in results if "A*" in res[0] and "ANA*" not in res[0]]
    ana_star_results = [res for res in results if "ANA*" in res[0]]

    # Plot A* results, NO NEED FOR DEMO
    #plot_results(a_star_results, "Solution Cost Over Time for A* Methods", rows=2, cols=3)

    # Plot ANA* results, NO NEED FOR DEMO
    #plot_results(ana_star_results, "Solution Cost Over Time for ANA* Methods", rows=2, cols=3)

    # Wait for user to close the GUI
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()

