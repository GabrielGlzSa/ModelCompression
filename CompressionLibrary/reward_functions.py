import tensorflow as tf


def reward_funcv1(before, after):
    """
    Function that calculates the reward as 1 - (Wt/Wt-1) + At.
    """
    return 1 - (after['weights_after'] / before['weights_before']) + after['acc_after']
