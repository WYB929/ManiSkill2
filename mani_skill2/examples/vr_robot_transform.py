import numpy as np
from scipy.spatial.transform import Rotation as R
from mani_skill2.examples import SCALE_FACTOR

def robot_pose_aa_to_affine(pose_aa:np.ndarray) -> np.ndarray:
    """ Returns a 4x4 affine matrix from the controller's position and rotation.
    Args:
        controller_position: 3D position of the controller.
        controller_rotation: (ax, ay, az) angle-axis rotation of the controller.
    """
    pose_aa = np.array(pose_aa)
    pos = pose_aa[:3] / SCALE_FACTOR
    rot = pose_aa[3:]
    rot = R.from_rotvec(rot).as_matrix()
    affine = np.eye(4)
    affine[:3, :3] = rot
    affine[:3, 3] = pos
    return affine

def affine_to_robot_pose_aa(affine:np.ndarray) -> np.ndarray:
    """ Returns 
        controller_position: 3D position of the controller.
        controller_rotation: (ax, ay, az) angle-axis rotation of the controller.
    Args:
        a 4x4 affine matrix from the controller's position and rotation.
    """
    affine = np.array(affine)
    pos = affine[:3, 3] * SCALE_FACTOR
    rot = affine[:3, :3]
    rot = R.from_matrix(rot).as_rotvec()
    pose_aa = np.zeros(6)
    pose_aa[:3] = pos
    pose_aa[3:] = rot
    return pose_aa
    