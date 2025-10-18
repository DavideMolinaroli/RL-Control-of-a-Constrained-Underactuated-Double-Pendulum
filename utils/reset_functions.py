import numpy as np


# used for evaluation, consider creating a file for this
def zero_reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0] # all pointing down
    # observation = [0.0, -1.0, -1.0, 0.0] # all pointing up
    # observation = [0.0, 1.0, 0.0, 0.0] # all pointing up
    # observation = [0.0, 0.0, 0.0, 0.0] # q1 up, q2 down
    # observation = [-1.0, 0.0, 0.0, 0.0] # q1 down, q2 up
    # so, q1 is 0 when first link is up. q2 is +-pi when it is aligned with the first link, looking on the outside
    return observation


# used for training in phase 1
def noisy_reset_func():
    scale = 0.05
    rand = np.random.rand(4) * scale
    rand[2:] -= scale / 2
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation

# used for training in phase 2
def disturb_reset_func():
    sign = np.random.choice([-1,1])
    if sign == 1:
        observation = [-1.0, -1.0, 0.0, 0.0]
    else:
        observation = [0.0, -1.0, -1.0, 0.0]

    return [-1.0, -1.0, -1.0, -1.0]
    return observation

def stabilisation_reset_func():
    observation = [0.0, -1.0, 0.0, 0.0]
    return observation


def noisy_stabilisation_reset_func():
    scale = 0.05
    rand = np.random.rand(4) * scale
    rand[2:] -= scale / 2  # velocity noise
    rand[0] -= scale / 2  # theta1 noise
    observation = [0.0, -1.0, 0.0, 0.0] + rand
    return observation
