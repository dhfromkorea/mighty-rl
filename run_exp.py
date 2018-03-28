import sklearn.preprocessing
from sklearn import manifold, datasets
from sklearn.utils import check_random_state
import gym

from irl.apprenticeship_learning import apprenticeship_learning
from irl.maximum_margin_planning import maximum_margin_planning
from utils import *
from lstd import LSTDQ, LSTDMu, LSPI
from simulator import Simulator
from policy import *

def get_behavior_policies(is_expert=False):
    if not is_expert:
        pi_list = []
        pi1 = RandomPolicy2(choices=[0]) # left
        pi_list.append(pi1)
        pi2 = RandomPolicy2(choices=[2]) # right
        pi_list.append(pi2)
        pi3 = RandomPolicy2(choices=[0, 2]) # left, right
        pi_list.append(pi3)

    class ManualPolicy():
        def choose_action(self, s):
            pos, v = s
            return 0 if v <=0 else 2
    pi4 = ManualPolicy()
    pi_list.append(pi4)
    return pi_list


def get_evaluation_policy():
    pi5 = RandomPolicy2(choices=[0, 1, 2]) # left, stay, right
    return pi5


def get_training_data(pi_list):
    state_dim = env.observation_space.shape[0]
    # discrete action
    action_dim = 1
    n_action = env.action_space.n
    sim = Simulator(env, state_dim=state_dim, action_dim=action_dim)
    traj_list = []
    for pi in pi_list:
        trajs = sim.simulate(pi1, n_trial=1, n_episode=50)
        traj_list.append(trajs)
    return traj_list

def get_basis_function(env_id):
    env = gym.envs.make(env_id)
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    states = np.array([env.observation_space.sample() for x in range(10000)])
    actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
    xs = np.hstack((states, actions))

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(xs)

    phi_rbf = get_phi(scaler)
    return phi_rbf, scaler



def main():
    # hypyerparams
    gamma= 0.75
    s0 = [-0.5, -0.05]
    env_id = "MountainCar-v0"
    env = gym.envs.make(env_id)

    pi_behavior_list = get_behavior_policies()

    trajs = []
    for traj in get_training_data(pi_behavior_list):
        trajs += traj

    phi_rbf, scaler = get_basis_function(env_id)
    psi_rbf = phi_rbf
    phi_linear = simple_phi
    psi_linear = phi_linear

    pi_eval = get_evaluation_policy()

    # get lstdmu_linear
    eps = 0.001
    p_linear = 3
    q_linear = 3
    lm_linear = LSTDMu(p=p_linear, q=q_linear, phi=phi_linear, \
                       psi=psi_linear, gamma=gamma, eps=eps)
    lm_linear.fit(D=trajs, pi=pi_eval)

    # get lstdmu_rbf
    p_rbf = 400
    q_rbf = 400
    lm_rbf = LSTDMu(p=p_rbf, q=q_rbf, phi=phi_rbf, \
                    psi=psi_rbf, gamma=gamma, eps=eps)
    lm_linear.fit(D=trajs, pi=pi_eval)



if __name__ == "__main__":
    main()
