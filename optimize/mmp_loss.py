import numpy as np
from numpy.linalg import norm
from constants import *

def action_variation_loss(Q_expert, Q_irl, state_histogram, q=2):
    '''
    Parameter
        q(int) : norm
    Returns
        sum over state_histogram(s) * ||Q_irl(s) - Q_expert(s)||_q

    '''
    # normalize Q scale since relative freq across action columns only matters
    Q_expert = Q_expert / Q_expert.sum(axis=1, keepdims=True)
    Q_irl = Q_irl / Q_irl.sum(axis=1, keepdims=True)
    Q_diffs = norm(Q_expert - Q_irl, ord=q, axis=1)
    return state_histogram.dot(Q_diffs)






