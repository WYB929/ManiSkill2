import argparse
import multiprocessing as mp

import gym
import numpy as np

from mani_skill2 import make_box_space_readable
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode
import zmq
from mani_skill2.examples import VR_TCP_ADDRESS, VR_TOPIC
from mani_skill2.examples.vr_robot_transform import affine_to_robot_pose, get_relative_affine
from mani_skill2.examples.controller_subscriber import vr_subscriber, get_data

# set up logging to a file
import logging
import os
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(
    level=logging.DEBUG,
    datefmt="%m-%d %H:%M",
    filename="logs/manuel_test_recorder.log",
)


MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--control-opt", type=str, default="key")
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    logging.info(f"opts: {opts}")
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    logging.info(f"env_kwargs: {env_kwargs}")
    args.env_kwargs = env_kwargs

    return args

def main():
    make_box_space_readable()
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode, info_on_video=True)
    # env = RecordEpisode(
    #     env,
    #     "./videos", # the directory to save replay videos and trajectories to
    #     render_mode="cameras", # cameras - three camera images + depth images, rgb_array - single camera image
    #     info_on_video=True # when True, will add informative text onto the replay video such as step counter, reward, and other metrics 
    # )


    obs = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render(mode="human")
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            sapien_viewer = env.render(mode="human")
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1
    
    if args.control_opt == "key":
        while True:
            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.enable_sapien_viewer:
                env.render(mode="human")

            render_frame = env.render(mode=args.render_mode)

            if after_reset:
                after_reset = False
                # Re-focus on opencv viewer
                if args.enable_sapien_viewer:
                    opencv_viewer.close()
                    opencv_viewer = OpenCVViewer(exit_on_esc=False)

            # -------------------------------------------------------------------------- #
            # Interaction
            # -------------------------------------------------------------------------- #
            # Input
            key = opencv_viewer.imshow(render_frame)

            if has_base:
                assert args.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
                base_action = np.zeros([4])  # hardcoded
            else:
                base_action = np.zeros([0])

            # Parse end-effector action
            if (
                "pd_ee_delta_pose" in args.control_mode
                or "pd_ee_target_delta_pose" in args.control_mode
            ):
                ee_action = np.zeros([6])
            elif (
                "pd_ee_delta_pos" in args.control_mode
                or "pd_ee_target_delta_pos" in args.control_mode
            ):
                ee_action = np.zeros([3])
            else:
                raise NotImplementedError(args.control_mode)

            # Base
            if has_base:
                if key == "w":  # forward
                    base_action[0] = 1
                elif key == "s":  # backward
                    base_action[0] = -1
                elif key == "a":  # left
                    base_action[1] = 1
                elif key == "d":  # right
                    base_action[1] = -1
                elif key == "q":  # rotate counter
                    base_action[2] = 1
                elif key == "e":  # rotate clockwise
                    base_action[2] = -1
                elif key == "z":  # lift
                    base_action[3] = 1
                elif key == "x":  # lower
                    base_action[3] = -1

            # End-effector
            if num_arms > 0:
                # Position
                if key == "i":  # +x
                    ee_action[0] = EE_ACTION
                elif key == "k":  # -x
                    ee_action[0] = -EE_ACTION
                elif key == "j":  # +y
                    ee_action[1] = EE_ACTION
                elif key == "l":  # -y
                    ee_action[1] = -EE_ACTION
                elif key == "u":  # +z
                    ee_action[2] = EE_ACTION
                elif key == "o":  # -z
                    ee_action[2] = -EE_ACTION

                # Rotation (axis-angle)
                if key == "1":
                    ee_action[3:6] = (1, 0, 0)
                elif key == "2":
                    ee_action[3:6] = (-1, 0, 0)
                elif key == "3":
                    ee_action[3:6] = (0, 1, 0)
                elif key == "4":
                    ee_action[3:6] = (0, -1, 0)
                elif key == "5":
                    ee_action[3:6] = (0, 0, 1)
                elif key == "6":
                    ee_action[3:6] = (0, 0, -1)

            # Gripper
            if has_gripper:
                if key == "f":  # open gripper
                    gripper_action = 1
                elif key == "g":  # close gripper
                    gripper_action = -1

            # Other functions
            if key == "0":  # switch to SAPIEN viewer
                render_wait()
            elif key == "r":  # reset env
                obs = env.reset()
                gripper_action = 1
                after_reset = True
                continue
            elif key == None:  # exit
                break

            # Visualize observation
            if key == "v":
                if "rgbd" in env.obs_mode:
                    from itertools import chain

                    from mani_skill2.utils.visualization.misc import (
                        observations_to_images,
                        tile_images,
                    )

                    images = list(
                        chain(*[observations_to_images(x) for x in obs["image"].values()])
                    )
                    render_frame = tile_images(images)
                    opencv_viewer.imshow(render_frame)
                elif "pointcloud" in env.obs_mode:
                    import trimesh

                    xyzw = obs["pointcloud"]["xyzw"]
                    mask = xyzw[..., 3] > 0
                    rgb = obs["pointcloud"]["rgb"]
                    if "robot_seg" in obs["pointcloud"]:
                        robot_seg = obs["pointcloud"]["robot_seg"]
                        rgb = np.uint8(robot_seg * [11, 61, 127])
                    trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

            # -------------------------------------------------------------------------- #
            # Post-process action
            # -------------------------------------------------------------------------- #
            if args.env_id in MS1_ENV_IDS:
                action_dict = dict(
                    base=base_action,
                    right_arm=ee_action,
                    right_gripper=gripper_action,
                    left_arm=np.zeros_like(ee_action),
                    left_gripper=np.zeros_like(gripper_action),
                )
                action = env.agent.controller.from_action_dict(action_dict)
            else:
                action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
                action = env.agent.controller.from_action_dict(action_dict)
            
            logging.info(f"action {action}")
            obs, reward, done, info = env.step(action)
            logging.info(f"reward {reward}")
            logging.info(f"done {done}")
            logging.info(f"info {info}")
        # env.flush_video()
        env.close()

    elif args.control_opt == "vr":
        
        # create queues for multiprocess message transfer
        shared_queue = mp.Queue()

        lock = mp.Lock()

        subscriber_process = mp.Process(target=vr_subscriber, args=(VR_TCP_ADDRESS, VR_TOPIC, shared_queue, lock))
        subscriber_process.start()

        start_left, start_right = False, False
        # Calibration frames
        init_left_affine, init_right_affine = None, None
        prev_affine = None

        left_x_pressed, right_a_pressed = 0, 0
        robot_gripper = env.agent.robot.get_links()[9]
        gripper_init_affine = robot_gripper.pose.to_transformation_matrix()
        gripper_init_affine[:3, 3] = [0, 0, 0]
        gripper_current_affine = gripper_init_affine
        while True:
            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.enable_sapien_viewer:
                env.render(mode="human")
            render_frame = env.render(mode=args.render_mode)

            if after_reset:
                after_reset = False
                # Re-focus on opencv viewer
                if args.enable_sapien_viewer:
                    opencv_viewer.close()
                    opencv_viewer = OpenCVViewer(exit_on_esc=False)

            opencv_viewer.imshow(render_frame, non_blocking=True, delay=1)

            # -------------------------------------------------------------------------- #
            # Interaction
            # -------------------------------------------------------------------------- #
            # Input


            if has_base:
                assert args.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
                base_action = np.zeros([4])  # hardcoded
            else:
                base_action = np.zeros([0])
            
            # Parse end-effector action
            if (
                "pd_ee_delta_pose" in args.control_mode
                or "pd_ee_target_delta_pose" in args.control_mode
            ):
                ee_action = np.zeros([6])
            elif (
                "pd_ee_delta_pos" in args.control_mode
                or "pd_ee_target_delta_pos" in args.control_mode
            ):
                ee_action = np.zeros([3])
            else:
                raise NotImplementedError(args.control_mode)
            
            parsed_data = get_data(lock, shared_queue)
            logging.info(f"Data {parsed_data}")
            # avoid empty data
            if parsed_data is None:
                continue

            # adjust hand position
            if parsed_data.left_thumbstick or parsed_data.right_thumbstick:
                # prev_affine = prev_affine
                if start_left:
                    init_left_affine = parsed_data.left_affine
                elif start_right:
                    init_right_affine = parsed_data.right_affine
                continue
            # Base
            if has_base:
                if start_left:
                    base_action[0] = parsed_data.right_thumbstick_axes[1] # forward and backward
                    base_action[1] = parsed_data.right_thumbstick_axes[0] # left and right
                    base_action[2] = parsed_data.right_thumbstick_axes[0] # rotation
                    base_action[3] = parsed_data.right_thumbstick_axes[1] # lift and lower
                elif start_right:
                    base_action[0] = parsed_data.left_thumbstick_axes[1] # forward and backward
                    base_action[1] = parsed_data.left_thumbstick_axes[0] # left and right
                    base_action[2] = parsed_data.left_thumbstick_axes[0] # rotation
                    base_action[3] = parsed_data.left_thumbstick_axes[1] # lift and lower

            # End-effector
            if num_arms > 0:
                # control with left teleop or right teleop
                # left
                if parsed_data.left_x and (left_x_pressed == 0):
                    print("Left Telelop Seleted")
                    start_left = True
                    left_x_pressed = 1
                    init_left_affine = parsed_data.left_affine
                    prev_affine = init_left_affine
                # right
                elif parsed_data.right_a and (right_a_pressed == 0):
                    print("Right Telelop Seleted")
                    start_right = True
                    right_a_pressed = 1
                    init_right_affine = parsed_data.right_affine
                    prev_affine = init_right_affine

                # Tracking Position
                # get the affine of current gripper
                robot_gripper = env.agent.robot.get_links()[9]
                gripper_current_affine = robot_gripper.pose.to_transformation_matrix()
                gripper_current_affine[:3, 3] = [0, 0, 0]
                # get the relative affine of current gripper
                relative_rot = get_relative_affine(gripper_init_affine, gripper_current_affine)[:3, :3]

                if start_left:
                    left_relative_affine = get_relative_affine(prev_affine, parsed_data.left_affine)
                    ee_action[0:3] = np.linalg.pinv(relative_rot) @ affine_to_robot_pose(left_relative_affine)[0:3]
                    prev_affine = parsed_data.left_affine
                elif start_right:
                    right_relative_affine = get_relative_affine(prev_affine, parsed_data.right_affine)
                    ee_action[0:3] = np.linalg.pinv(relative_rot) @ affine_to_robot_pose(right_relative_affine)[0:3]
                    prev_affine = parsed_data.right_affine
                
                # Rotation
                if start_left:
                    ee_action[3] = parsed_data.left_thumbstick_axes[0]
                    ee_action[4] = parsed_data.left_thumbstick_axes[1]
                    if parsed_data.left_x and (left_x_pressed==1): # rotate gripper
                        ee_action[5] = 1
                    elif parsed_data.left_y:
                        ee_action[5] = -1
                    else:
                        ee_action[5] = 0

                elif start_right:
                    ee_action[3] = parsed_data.right_thumbstick_axes[0]
                    ee_action[4] = parsed_data.right_thumbstick_axes[1]
                    if parsed_data.right_a and (right_a_pressed==1): # rotate gripper
                        ee_action[5] = 1
                    elif parsed_data.right_b:
                        ee_action[5] = -1
                    else:
                        ee_action[5] = 0


            # Gripper
            if has_gripper:
                if start_left:
                    # open gripper with index trigger
                    if parsed_data.left_index_trigger > 0:
                        gripper_action = 1
                    # close gripper with hand trigger
                    elif parsed_data.left_hand_trigger > 0:
                        gripper_action = -1
                elif start_right:
                    # open gripper with index trigger
                    if parsed_data.right_index_trigger > 0:
                        gripper_action = 1
                    # close gripper with hand trigger
                    elif parsed_data.right_hand_trigger > 0:
                        gripper_action = -1

            # Other functions
            if parsed_data.left_y and parsed_data.right_b:  # reset env
                obs = env.reset()
                gripper_action = 1
                # reset calibration frames and choice of left or right teleop
                start_left, start_right = False, False
                init_left_affine, init_right_affine = None, None
                left_x_pressed, right_a_pressed = 0, 0
                after_reset = True
                continue

            if parsed_data.left_x and parsed_data.right_a:
                logging.info("Exiting program")
                break # exit
        
            # -------------------------------------------------------------------------- #
            # Post-process action
            # -------------------------------------------------------------------------- #
            if args.env_id in MS1_ENV_IDS:
                action_dict = dict(
                    base=base_action,
                    right_arm=ee_action,
                    right_gripper=gripper_action,
                    left_arm=np.zeros_like(ee_action),
                    left_gripper=np.zeros_like(gripper_action),
                )
                action = env.agent.controller.from_action_dict(action_dict)
            else:
                action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
                # logging.info(action_dict)
                action = env.agent.controller.from_action_dict(action_dict)

            logging.info(f"action {action}")
            obs, reward, done, info = env.step(action)
            logging.info(f"reward {reward}")
            logging.info(f"done {done}")
            logging.info(f"info {info}")
        env.close()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()