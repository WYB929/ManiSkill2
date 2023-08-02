import numpy as np
from scipy.spatial.transform import Rotation as R
from mani_skill2.examples import SCALE_FACTOR, ROT_FLIP, AXIS_FLIP

def robot_pose_to_affine(pose:np.ndarray) -> np.ndarray:
    """ Returns a 4x4 affine matrix from the controller's position and rotation.
    Args:
        controller_position: 3D position of the controller.
        controller_rotation: (ax, ay, az) angle-axis rotation of the controller.
    """
    pose = np.array(pose)
    pos = pose[:3] / SCALE_FACTOR
    rot = pose[3:]
    rot = R.from_rotvec(rot).as_matrix()
    affine = np.eye(4)
    affine[:3, :3] = rot
    affine[:3, 3] = pos
    return affine

def affine_to_robot_pose(affine:np.ndarray) -> np.ndarray:
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
    pose = np.zeros(6)
    pose[:3] = pos
    pose[3:] = rot
    return pose
    
def get_relative_affine(init_affine, current_affine):
    rel_affine_vr = np.linalg.pinv(init_affine) @ current_affine
    rel_affine_trans = (AXIS_FLIP @ rel_affine_vr @ AXIS_FLIP) [:3, 3]
    rel_affine_rot = (ROT_FLIP @ rel_affine_vr @ ROT_FLIP) [:3, :3]

    rel_affine = np.eye(4)
    rel_affine[:3, :3] = rel_affine_rot
    rel_affine[:3, 3] = rel_affine_trans
    return rel_affine
