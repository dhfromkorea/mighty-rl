import os
import json
import logging
import logging.config
import sklearn.preprocessing
from sklearn import manifold, datasets
from sklearn.utils import check_random_state
import gym
import itertools
import pickle
from time import time


from lstd import LSTDQ, LSTDMu, LSPI
from envs.simulator import Simulator
from policy import *
from utils import *
from irl.apprenticeship_learning import ApprenticeshipLearning as AL
from fa import LinearQ3
import plotting


class NearExpertPolicy():
    """
    hard-coded near-optimal expert policy
    for mountaincar-v0
    """
    def choose_action(self, s):
        pos, v = s
        return 0 if v <=0 else 2


def setup_logging(
    default_path='logging.json',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_behavior_policies(only_expert=False):
    pi_list = []
    if not only_expert:
        pi1 = RandomPolicy2(choices=[0]) # left
        pi_list.append(pi1)
        pi2 = RandomPolicy2(choices=[2]) # right
        pi_list.append(pi2)
        pi3 = RandomPolicy2(choices=[0, 2]) # left, right
        pi_list.append(pi3)

    pi_exp = NearExpertPolicy()
    pi_list.append(pi_exp)
    return pi_list


def get_random_policy():
    return RandomPolicy2(choices=[0, 1, 2]) # left, stay, right


def get_training_data(env, pi_list, sample_size, mix_ratio):
    state_dim = env.observation_space.shape[0]
    # discrete action
    action_dim = 1
    n_action = env.action_space.n
    sim = Simulator(env, state_dim=state_dim, action_dim=action_dim)
    traj_list = []
    for pi, r in zip(pi_list, mix_ratio):
        trajs = sim.simulate(pi, n_trial=1, n_episode=int(r * sample_size))
        traj_list += trajs
    return traj_list

def estimate_mu_mc(env, pi, phi, gamma, n_episode):
    mus = []
    ss_init = []
    for epi_i in range(n_episode):

        # this is not fixed
        s = env.reset()
        ss_init.append(s)
        mu = 0.0
        for t in itertools.count():
            a = pi.choose_action(s)
            s_next, r, done, _ = env.step(a)
            # todo figure out whether it's phi(s,a) or phi(s)
            mu += gamma ** t * phi(s, a)
            s = s_next
            if done:
                break
        mus.append(mu)
    return np.array(mus)

def get_basis_function(env_id):
    env = gym.envs.make(env_id)
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    states = np.array([env.observation_space.sample() for x in range(10000)])
    actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
    xs = np.hstack((states, actions))

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(xs)

    phi_rbf = get_phi(scaler, scaler.transform(xs))
    return phi_rbf


def main():
    logging.info("define environment and basis function")
    env_id = "MountainCar-v0"
    env = gym.envs.make(env_id)
    logging.info("env_id: {}".format(env_id))
    action_list = range(env.action_space.n)

    # linear basis func
    p_linear = 3
    q_linear = 3
    phi_linear = simple_phi
    psi_linear = phi_linear

    # radial basis (gaussian) fn
    p_rbf = 200
    q_rbf = 200
    phi_rbf = get_basis_function(env_id)
    psi_rbf = phi_rbf


    # this is specific to mountaincar-v0
    init_s_sampler = lambda : [np.random.uniform(-0.4, -0.6), 0.0]

    # 2. define hyperparams
    gamma= 0.95
    n_trial = 2
    n_iteration = 10
    # @note: hard-coded
    # this's gotta be sufficiently large to avoid mc variance issue
    sample_size_mc = 10**2
    p = p_linear
    q = q_linear
    phi = phi_linear
    psi = psi_linear
    #p = p_rbf
    #q = q_rbf
    #phi = phi_rbf
    #psi = psi_rbf
    precision = 1e-4
    use_slack = False
    # @note: reward may have to be scaled to work with slack penalty
    slack_penalty = 1e-3
    eps = 0.0001
    #eps = 0
    # this should be large to account for varying init sate
    mu_sample_size = 10**2

    logging.info("collect a batch of data (D) from pi_expert (and some noise)")
    pi_exp = NearExpertPolicy()
    pi_random = get_random_policy()
    pi_behavior_list = [pi_exp]
    #mix_ratio = [0.8, 0.2]
    mix_ratio = [1.0]

    D = np.empty((0, 5))
    # number of episodes
    D_sample_size = 50
    for traj in get_training_data(env,
                                  pi_list=pi_behavior_list,
                                  sample_size=D_sample_size,
                                  mix_ratio=mix_ratio):
        D = np.vstack((D, np.array(traj)))

    # preprocessing D in numpy array for k
    logging.info("apprenticeship learning starts")

	mu_exp = AL.estimate_mu(env=env,
							pi_eval=pi_exp,
							mu_sample_size=sample_size_mc,
							phi=self._phi,
							gamma=self._gamma,
							return_epi_len=False)
    #mu_mc_list = estimate_mu_mc(env, pi_exp, phi_linear, gamma, sample_size_mc)
    #mu_mc_list = estimate_mu_mc(env, pi_exp, phi_rbf, gamma, sample_size_mc)
    #mu_exp = np.mean(mu_mc_list, axis=0)

    pi_init = pi_random

    mdp_solver = LinearQ3(env=env,
                          estimator=Estimator(),
                          episode_length=100,
                          epsilon=0.0,
                          gamma=0.99,
                          plotting=plotting)

    al = AL(env=env,
          pi_init=pi_init,
          action_list=action_list,
          p=p,
          q=q,
          phi=phi,
          psi=psi,
          gamma=gamma,
          eps=eps,
          mu_exp=mu_exp,
          init_s_sampler=init_s_sampler,
          mu_sample_size=mu_sample_size,
          precision=precision,
          mdp_solver=mdp_solver,
          use_slack=use_slack,
          slack_penalty=slack_penalty)

    results = al.run(n_trial=n_trial, n_iteration=n_iteration)

    # 5. post-process results (plotting)
    with open("data/res_{}".format(time()), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    setup_logging(default_level=logging.INFO)
    main()
