import numpy as np

def set_system(system):

    return system


def check_and_set_time_points_input(tp):

    if np.atleast_2d(tp).shape[0] == 1:

        tp = np.squeeze(np.asarray(tp))

    elif np.atleast_2d(tp).shape[1] == 1:

        tp = np.squeeze(np.atleast_2d(tp).T)

    else:

        raise ValueError("Invalid dimension for tp.")

    return tp   
    