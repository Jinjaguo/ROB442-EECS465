import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model, \
    joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###
import pybullet as p
from PIL import Image
#########################

from utils import draw_sphere_marker


def save_gif(frames, filename="output.gif", duration=100):
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res


def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn)  # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn)  # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world


def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn)  # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn)  # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi


def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)


def get_translation_jacobian(robot, joint_indices):
    J = np.zeros((3, len(joint_indices)))
    end_effector_position = get_ee_transform(robot, joint_indices)[:3, 3]
    for i, joint_idx in enumerate(joint_indices):
        j_info = p.getJointInfo(robot, joint_idx)  # 获取单个关节的信息
        joint_type = j_info[2]  # 获取关节类型

        if joint_type == p.JOINT_REVOLUTE:
            joint_position = get_joint_position(robot, joint_idx)  # 当前关节位置
            joint_axis = get_joint_axis(robot, joint_idx)  # 当前关节轴方向
            # 旋转关节的雅可比列向量
            J[:, i] = np.cross(joint_axis, end_effector_position - joint_position)

        elif joint_type == p.JOINT_PRISMATIC:
            joint_axis = get_joint_axis(robot, joint_idx)
            # 平移关节的雅可比列向量
            J[:, i] = joint_axis

    return J


def get_rotation_jacobian(robot, joint_indices):
    J_rotation = np.zeros((3, len(joint_indices)))
    for i, joint_idx in enumerate(joint_indices):
        j_info = p.getJointInfo(robot, joint_idx)  # 获取关节信息
        joint_type = j_info[2]  # 获取关节类型

        if joint_type == p.JOINT_REVOLUTE:
            joint_axis = get_joint_axis(robot, joint_idx)  # 旋转关节的轴方向
            J_rotation[:, i] = joint_axis  # 旋转关节的雅可比列向量等于关节轴方向

        elif joint_type == p.JOINT_PRISMATIC:
            # 对于平移关节，旋转雅可比矩阵的列向量为零
            J_rotation[:, i] = np.zeros(3)

    return J_rotation


def get_full_jacobian(robot, joint_indices):
    # 计算平移部分雅可比矩阵
    J_translation = get_translation_jacobian(robot, joint_indices)
    # 计算旋转部分雅可比矩阵
    J_rotation = get_rotation_jacobian(robot, joint_indices)

    # 上下堆叠平移和旋转部分，形成完整的雅可比矩阵
    J_full = np.vstack((J_translation, J_rotation))
    return J_full


def get_jacobian_pinv(J):
    lamda = 0.01  # 正则化参数
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lamda ** 2 * np.eye(J.shape[0]))
    return J_pinv


def tuck_arm(robot):
    joint_names = ['torso_lift_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_wrist_flex_joint',
                   'r_shoulder_lift_joint', 'r_elbow_flex_joint', 'r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx,
                        (0.24, 1.29023451, -2.32099996, -0.69800004, 1.27843491, -2.32100002, -0.69799996))


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("  'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False)
    camera_distance = 3  # 摄像机与目标的距离
    camera_yaw = 50  # 摄像机的偏航角
    camera_pitch = -35  # 摄像机的俯仰角
    camera_target_position = [-0.75, -0.07551, 0.42]  # 摄像机目标位置
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # 保存每一帧的图像列表
    frames = []

    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_point(robot, (-0.75, -0.07551, 0.02))
    tuck_arm(robot)

    # define active DoFs
    joint_names = ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                   'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    q_arr = np.zeros((1, len(joint_idx)))
    set_joint_positions_np(robot, joint_idx, q_arr)

    targets = [[-0.15070158, 0.47726995, 1.56714123],
               [-0.36535318, 0.11249, 1.08326675],
               [-0.56491217, 0.011443, 1.2922572],
               [-1.07012697, 0.81909669, 0.47344636],
               [-1.11050811, 0.97000718, 1.31087581]]

    for target in targets:
        draw_sphere_marker(target, 0.05, (1, 0, 0, 1))

    joint_limits = {joint_names[i]: (
        get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i
        in
        range(len(joint_idx))}
    q = np.zeros((1, len(joint_names)))
    target = targets[test_idx]

    max_iters = 100
    threshold = 0.01
    alpha = 0.1
    joint_limit = np.array([[value[0], value[1]] for value in joint_limits.values()])
    joint_limit[4] = [-np.pi, np.pi]
    joint_limit[-1] = [-np.pi, np.pi]

    for _ in range(max_iters):
        # 更新机器人关节位置
        set_joint_positions_np(robot, joint_idx, q)

        # 确保场景更新
        p.stepSimulation()

        # 获取当前末端执行器位置
        current = get_ee_transform(robot, joint_idx)[:3, 3]
        draw_sphere_marker(current, 0.05, (0, 0, 1, 1))

        error = target - current
        if np.linalg.norm(error) < threshold:
            print("The configuration is: ", q)
            break

        J = get_translation_jacobian(robot, joint_idx)
        J_pinv = get_jacobian_pinv(J)
        delta_q = J_pinv @ error

        if np.linalg.norm(delta_q) > alpha:
            delta_q = alpha * delta_q / np.linalg.norm(delta_q)

        for i, delta in enumerate(delta_q):
            lower, upper = joint_limit[i]
            q[0, i] = np.clip(q[0, i] + delta, lower, upper)

        # 获取图像前进行渲染更新
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target_position,
                distance=camera_distance,
                yaw=camera_yaw,
                pitch=camera_pitch,
                roll=0,
                upAxisIndex=2
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
        )

        # 转换为 PIL 格式并添加到帧列表中
        img = Image.fromarray(rgb_img)
        frames.append(img)

        time.sleep(0.01)

    save_gif(frames, filename="simulation.gif", duration=100)
    print('The configuration of robot is', q)

    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
