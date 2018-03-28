import numpy as np
import numba as nb
import pandas as pd
import itertools
import multiprocessing as mp

from constants import *
from utils.utils import is_terminal_state, compute_terminal_state_reward


def make_initial_state_sampler(df, has_priority=True):
    '''
    we only care about empirically observed initial states.
    '''
    if has_priority:
        probs = df[df['bloc'] == 1]['state'].value_counts(normalize=True).tolist()
        f = lambda : np.random.choice(np.arange(len(probs)), p=probs)
    else:
        initial_states = df[df['bloc'] == 1]['state'].unique().tolist()
        f = lambda : np.random.choice(initial_states)
    return f


def get_initial_state_distribution(df):
    '''
    we only care about empirically observed initial states.
    '''
    probs = df[df['bloc'] == 1]['state'].value_counts(normalize=True).tolist()
    return probs


def get_state_action_histogram(df):
    '''returns (s, a) histogram
    '''
    sa_histogram = np.zeros((NUM_PURE_STATES, NUM_ACTIONS))
    df_sa = df.groupby(['state'])['action'].value_counts().astype(np.int)
    for s in range(NUM_PURE_STATES):
        for a in range(NUM_ACTIONS):
            if (s, a) in df_sa:
                sa_histogram[s, a] = df_sa[s, a]
    return sa_histogram


def make_state_centroid_finder(df, columns=None):
    if columns is not None:
        df = df[columns]
    def f(state):
        return df.iloc[state]
    return f


def sample_state(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)

def estimate_feature_expectation(transition_matrix,
                                 initial_state_probs,
                                 phi,
                                 pi,
                                 sample_size,
                                 gamma,
                                 max_iter):
    '''
    estimate mu_pi and v_pi with monte carlo simulation
    '''

    s = sample_state(initial_state_probs)
    mu = np.zeros((phi.shape[1]))
    v_sum = 0.0

    mus = []
    vs = []
    total_traj_length = 0
    num_max_iters = 0
    episodes = [[] for _ in range(sample_size)]
    for i in range(sample_size):
        s = sample_state(initial_state_probs)
        for t in itertools.count():
            if t > max_iter:
                num_max_iters += 1
                break
            # accumulate phi(s) over trajectories
            mu += gamma**t * phi[s, :]
            chosen_a = pi.choose_action(s)

            probs = np.copy(transition_matrix[s, chosen_a, :])

            # need to renomralize so sum(probs) < 1
            # todo: hack
            if np.sum(probs) == 0.0:
                new_s = s
            else:
                probs /= np.sum(probs)
                new_s = np.random.choice(np.arange(len(probs)), p=probs)

            exp = (s, chosen_a, new_s, t)
            episodes[i].append(exp)

            #print('s={} a={} next_s={}'.format(s, chosen_a, new_s))
            if is_terminal_state(new_s):
                # there's no phi(terminal_state)
                # in practice, non-zero rewars for terminal states
                num_features = mu.shape[0]
                v_sum += gamma** t * compute_terminal_state_reward(new_s, num_features)
                break
            s = new_s
        #print('traj ended with', t)
        total_traj_length += t
    max_iter_rate = num_max_iters / sample_size
    #print('max iter rate {:.2f} %'.format(max_iter_rate * 100))
    mu = mu / sample_size
    # let's not use v estimated here
    # let's use evaluate_policy_monte_carlo
    v =  v_sum / sample_size
    avg_traj_length = total_traj_length / sample_size
    history = (avg_traj_length, episodes, max_iter_rate)

    return mu, v, history


def make_phi(df_centroids):
    '''
    don't use this
    '''
    # median values for each centroid
    stats = df_centroids.describe()
    #take median
    median_state = stats.loc['50%']
    def phi(state):
        '''
        state: centroid values whose dimension is {num_features}
        phi: must apply decision rule (=indicator function)

        returs: binary matrix of R^{num_features}
        '''
        # TODO: implement this
        phi_s = np.array((state > median_state).astype(np.int))
        return phi_s
    return phi


def phi(df_centroids):
    # median values for each centroid
    stats = df_centroids.describe()
    #take median
    quartile_1 = stats.loc['25%']
    quartile_2 = stats.loc['50%']
    quartile_3 = stats.loc['75%']
    def phi(state):
        '''
        state: centroid values whose dimension is {num_features}
        phi: must apply decision rule (=indicator function)

        returs: binary matrix of R^{num_features}
        '''
        # TODO: implement this
        phi_s = np.array((state > median_state).astype(np.int))
        return phi_s
    return phi

def make_reward_computer(W, phi):
    def compute_reward(state):
        if is_terminal_state(state):
            # special case of terminal states
            # either 1 or -1
            num_features = W.shape[0]
            return compute_terminal_state_reward(state, num_features)
        return np.dot(W, phi[state, :])
    return compute_reward

def estimate_v_pi_tilda(W, mu, initial_state_probs, sample_size=100):
    # this does not work. don't use this for now.
    v_pi_tilda = np.dot(W, mu)
    # remove two terminal_states
    v_pi_tilda = v_pi_tilda[:v_pi_tilda.shape[0] - NUM_TERMINAL_STATES]
    v_pi_tilda_est = 0.0
    # TODO: vectorize this
    for _ in range(sample_size):
        s_0 = sample_state(initial_state_probs)
        v_pi_tilda_est += v_pi_tilda[s_0]
    return v_pi_tilda_est / sample_size
