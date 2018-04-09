import matplotlib.pyplot as plt
import numpy as np
import gym
import sys
import os

from algo.lstd import LSTDQ, LSTDMu, LSPI
from algo.policy import RandomPolicy2, LinearQ2
from env.simulator import *
from util.plotting import *
from util.basis import *
from algo.fa import LinearQ3

env_id = "MountainCar-v0"
env = gym.envs.make(env_id)
state_dim = env.observation_space.shape[0]
# discrete action
action_dim = 1
n_action = env.action_space.n
sim = Simulator(env, state_dim=state_dim, action_dim=action_dim)

# linear basis func
p_linear = 3
q_linear = 3
phi_linear = simple_phi
psi_linear = phi_linear
p_rbf = 100
q_rbf = 100
phi_rbf = get_rbf_basis(env, n_component=50)
psi_rbf = phi_rbf
p = p_linear
q = q_linear
phi = phi_linear
psi = psi_linear
precision = 0.1
eps = 0.001
gamma = 0.99
action_list = range(env.action_space.n)


# one reason: basis function includes action (remove... but how?)
# swapping with the original code copy paste
# estimator update?
mdp_solver = LinearQ3(env=env,
                      phi=phi_rbf,
                      action_list=action_list,
                      n_episode=60,
                      epsilon=0.0,
                      epsilon_decay=1.0,
                      gamma=gamma)

pi_expert, stats = mdp_solver.solve()
#plotting.plot_cost_to_go_mountain_car(env, pi_expert._estimator)
plotting.plot_episode_stats(stats, smoothing_window=10)

import pdb;pdb.set_trace()
pi_expert.coef
