from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from sapien.core import Pose

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import check_actor_static, vectorize_pose, look_at

from .base_env import StationaryManipulationEnv

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.agents.robots.xmate3 import Xmate3Robotiq
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig

class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self.fixtures = []

    def sample(self, radius, max_trials, append=True, verbose=False):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.
        """
        if len(self.fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self.fixtures])
            fixture_radius = np.array([x[1] for x in self.fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                if np.all(dist > fixture_radius + radius):
                    if verbose:
                        print(f"Found a valid sample at {i}-th trial")
                    break
            else:
                if verbose:
                    print("Fail to find a valid sample!")
        if append:
            self.fixtures.append((pos, radius))
        return pos


@register_env("StackCube-v1", max_episode_steps=200)
class StackCubeEnv(StationaryManipulationEnv):
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.enable_pcm = True
        return scene_config

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.table_height = 0.5
        self.table_thickness = 0.01
        self.table = self._build_table(size=0.5, height=self.table_height, thickness=self.table_thickness)

        self.box_half_size = np.float32([0.02] * 3)
        self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )
        self.cubeC = self._build_cube(self.box_half_size, color=(0, 0, 1), name="cubeC")

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.box_half_size[:2]) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
        cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeC_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))

        z = self.table_height + self.box_half_size[2]
        table_pose = sapien.Pose([-0.1, -0.1, self.table_height])
        cubeA_pose = sapien.Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = sapien.Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)
        cubeC_pose = sapien.Pose([cubeC_xy[0], cubeC_xy[1], z], cubeC_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)
        self.cubeC.set_pose(cubeC_pose)
        self.table.set_pose(table_pose)
    
    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, self.table_height]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_agent_v1(self):
        """Higher EE pos."""
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, self.table_height]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)
        
    def _register_cameras(self):
        pose_1 = look_at([0.3, 0, 0.8], [-0.2, 0, 0.1])
        pose_2 = look_at([0.25, 0.15, 0.7], [-0.1, 0, 0.1])
        pose_3 = look_at([0.3, 0, 1.0], [-0.1, 0, 0.1])
        return [CameraConfig("base_camera_1", pose_1.p, pose_2.q, 128, 128, np.pi / 2, 0.01, 10), 
                CameraConfig("base_camera_2", pose_2.p, pose_2.q, 128, 128, np.pi / 2, 0.01, 10),
                CameraConfig("base_camera_3", pose_3.p, pose_3.q, 128, 128, np.pi / 2, 0.01, 10)]
    
    def _register_render_cameras(self):
        if self.robot_uid == "panda":
            pose_1 = look_at([0.4, 0.4, 1.0], [0.0, 0.0, 0.4])
            pose_2 = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose_1 = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
            pose_2 = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        return [CameraConfig("render_camera", pose_1.p, pose_1.q, 512, 512, 1, 0.01, 10)]
    
    def _setup_viewer(self):
        super()._setup_viewer()
        # self._viewer.set_camera_xyz(0.8, 0, 5.0)
        # self._viewer.set_camera_rpy(0, -1.0, 5.0)

    def _get_obs_extra(self):
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cubeA_pose=vectorize_pose(self.cubeA.pose),
                cubeB_pose=vectorize_pose(self.cubeB.pose),
                cubeC_pose=vectorize_pose(self.cubeC.pose),
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
            )
        return obs

    def _check_cubeA_on_cubeB(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(self.box_half_size[:2]) + 0.005
        )
        z_flag = np.abs(offset[2] - self.box_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)

    def _check_cubeB_on_cubeC(self):
        pos_B = self.cubeB.pose.p
        pos_C= self.cubeC.pose.p
        offset = pos_B - pos_C
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(self.box_half_size[:2]) + 0.005
        )
        z_flag = np.abs(offset[2] - self.box_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)

    def evaluate(self, **kwargs):
        is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
        is_cubeB_on_cubeC = self._check_cubeB_on_cubeC()
        is_cubeA_static = check_actor_static(self.cubeA)
        is_cubeB_static = check_actor_static(self.cubeB)
        is_cubeC_static = check_actor_static(self.cubeC)
        is_cubaA_grasped = self.agent.check_grasp(self.cubeA)
        is_cubeB_grasped = self.agent.check_grasp(self.cubeB)
        is_cubeC_grasped = self.agent.check_grasp(self.cubeC)
        success = is_cubeA_on_cubeB and is_cubeB_on_cubeC and is_cubeA_static and is_cubeB_static and is_cubeC_static and (not is_cubaA_grasped) and (not is_cubeB_grasped) and (not is_cubeC_grasped)

        return {
            "is_cubaA_grasped": is_cubaA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeB_on_cubeC": is_cubeB_on_cubeC,
            "is_cubeB_static": is_cubeB_static,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_cubeC_static": is_cubeC_static,
            # "cubeA_pos": self.cubeA.pose.p,
            # "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
            # "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
            "success": success,
        }
    
    def update_rewards_with_cubes(self, cubeA, cubeB, reward):
        cubeA_pos = cubeA.pose.p
        cubeB_pos = cubeB.pose.p
        goal_xyz = np.hstack(
            [cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2]
        )
        cubeA_on_cubeB = (
            np.linalg.norm(goal_xyz[:2] - cubeA_pos[:2])
            < self.box_half_size[0] * 0.8
        )
        cubeA_on_cubeB = cubeA_on_cubeB and (
            np.abs(goal_xyz[2] - cubeA_pos[2]) <= 0.005
        )
        if cubeA_on_cubeB:
            reward = 10.0
            # ungrasp reward
            is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
            if not is_cubeA_grasped:
                reward += 2.0
            else:
                reward = (
                    reward
                    + 2.0 * np.sum(self.agent.robot.get_qpos()[-2:]) / self.gripper_width
                )
        else:
            # grasping reward
            is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
            if is_cubeA_grasped:
                reward += 1.0

            # reaching goal reward, ensuring that cubeA has appropriate height during this process
            if is_cubeA_grasped:
                cubeA_to_goal = goal_xyz - cubeA_pos
                # cubeA_to_goal_xy_dist = np.linalg.norm(cubeA_to_goal[:2])
                cubeA_to_goal_dist = np.linalg.norm(cubeA_to_goal)
                appropriate_height_penalty = np.maximum(
                    np.maximum(2 * cubeA_to_goal[2], 0.0),
                    np.maximum(2 * (-0.02 - cubeA_to_goal[2]), 0.0),
                )
                reaching_reward2 = 2 * (
                    1 - np.tanh(5.0 * appropriate_height_penalty)
                )
                # qvel_penalty = np.sum(np.abs(self.agent.robot.get_qvel())) # prevent the robot arm from moving too fast
                # reaching_reward2 -= 0.0003 * qvel_penalty
                # if appropriate_height_penalty < 0.01:
                reaching_reward2 += 4 * (1 - np.tanh(5.0 * cubeA_to_goal_dist))
                reward += np.maximum(reaching_reward2, 0.0)
        return reward
        

    def compute_dense_reward(self, info, **kwargs):
        self.gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        reward = 0.0

        if info["success"]:
            reward = 15.0
        else:
            # grasp pose rotation reward
            grasp_rot_loss_fxn = lambda A: np.tanh(
                1 / 8 * np.trace(A.T @ A)
            )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
            tcp_pose_wrt_cubeA = self.cubeA.pose.inv() * self.tcp.pose
            tcp_rot_wrt_cubeA = tcp_pose_wrt_cubeA.to_transformation_matrix()[:3, :3]
            tcp_pose_wrt_cubeB = self.cubeB.pose.inv() * self.tcp.pose
            tcp_rot_wrt_cubeB = tcp_pose_wrt_cubeB.to_transformation_matrix()[:3, :3]
            tcp_pose_wrt_cubeC = self.cubeC.pose.inv() * self.tcp.pose
            tcp_rot_wrt_cubeC = tcp_pose_wrt_cubeC.to_transformation_matrix()[:3, :3]

            gt_rots = [
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            ]
            grasp_rot_loss = min(
                [grasp_rot_loss_fxn(x - tcp_rot_wrt_cubeA - tcp_rot_wrt_cubeB - tcp_rot_wrt_cubeC) for x in gt_rots]
            )
            reward += 1 - grasp_rot_loss

            cubeB_vel_penalty = np.linalg.norm(self.cubeB.velocity) + np.linalg.norm(
                self.cubeB.angular_velocity
            )
            cubeC_vel_penalty = np.linalg.norm(self.cubeC.velocity) + np.linalg.norm(
                self.cubeC.angular_velocity
            )
            reward -= (cubeB_vel_penalty + cubeC_vel_penalty)

            # reaching object reward
            # tcp_pose = self.tcp.pose.p
            # cubeA_pos = self.cubeA.pose.p
            # cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
            # reaching_reward = 1 - np.tanh(3.0 * cubeA_to_tcp_dist)
            # reward += reaching_reward

            # check if cubeA is on cubeB and cubeB is on cubeC
            reward += self.update_rewards_with_cubes(self.cubeA, self.cubeB, reward)
            reward += self.update_rewards_with_cubes(self.cubeB, self.cubeC, reward)

        return reward
print("done")