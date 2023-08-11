import numpy as np

VR_TCP_ADDRESS = "tcp://10.19.224.217:5555"
VR_TOPIC = "oculus_controller"

SCALE_FACTOR = 50.0

AXIS_FLIP = (np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
]) @ np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
))

ROT_FLIP = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
)
