import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


class Estimator():
    """
    Q approximator.
    """

    def __init__(self, env, phi, action_list):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self._phi = phi
        self._action_list = action_list
        self.models = []
        for _ in self._action_list:
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([phi(env.reset(), 0).flatten()], [0])
            self.models.append(model)


    def predict(self, s, a=None):
        """
        Q
        """
        if a is None:
            Q = []
            for m, a in zip(self.models, self._action_list):
                features = self._phi(s, a).flatten()
                Q.append(m.predict([features])[0])
            return np.array(Q)
        else:
            features = self._phi(s, a).flatten()
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        features = self._phi(s, a).flatten()
        self.models[a].partial_fit([features], [y])



def q_learning(env, estimator, num_episodes, gamma=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        gamma: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        state = env.reset()

        next_action = None

        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            q_values_next = estimator.predict(next_state)

            td_target = reward + gamma * np.max(q_values_next)

            estimator.update(state, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

            if done:
                break

            state = next_state

    return stats


# @todo: fix this
class EpsilonGreedyPolicy(object):
    def __init__(self, estimator, eps, nA):
        """TODO: to be defined1. """
        self._estimator = estimator
        self._eps = eps
        self._nA = nA

    def _action_probs(self, s):
        probs = np.ones(self._nA, dtype=float) * self._eps / self._nA
        q_values = self._estimator.predict(s)
        best_action = np.argmax(q_values)
        probs[best_action] += (1.0 - self._eps)
        return probs

    def choose_action(self, s):
        p = self._action_probs(s)
        a = np.random.choice(np.arange(len(p)), p=p)
        return a


# @todo: make this included
def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


class LinearQ3(object):
    """Docstring for LinearQ3. """

    def __init__(self, env, phi, action_list, n_episode, epsilon, gamma):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        phi : array
        action_list : list
        n_episode : TODO
        epsilon : TODO
        plotting : TODO
        gamma: float
            discount factor

        """
        self._env = env
        self._estimator = Estimator(env=env, phi=phi, action_list=action_list)
        self._n_episode = n_episode
        self._epsilon = epsilon
        self._epsilon_decay = 1.0
        self._gamma = gamma


    def solve(self, reward_fn):
        """TODO: Docstring for solve.

        Parameters
        ----------
        reward_fn : function
            reward function for IRL
        epi_i_irl : int
            episode index of IRL

        Returns
        -------
        TODO

        """
        # @todo: hacky for copmutational reasons
        #n_episode = self._n_episode

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self._n_episode),
            episode_rewards=np.zeros(self._n_episode))

        for i_episode in range(self._n_episode):

            pi = EpsilonGreedyPolicy(estimator=self._estimator,
                                     eps=self._epsilon * self._epsilon_decay**i_episode,
                                     nA=self._env.action_space.n)
            last_reward = stats.episode_rewards[i_episode - 1]
            sys.stdout.flush()

            state = self._env.reset()

            next_action = None

            for t in itertools.count():

                if next_action is None:
                    action = pi.choose_action(state)
                else:
                    action = next_action

                next_state, reward, done, _ = self._env.step(action)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                q_values_next = self._estimator.predict(next_state)

                # td update
                # td_target = reward + gamma * np.max(q_values_next)
                reward_irl = np.asscalar(reward_fn(state, action))
                td_target = reward_irl + self._gamma * np.max(q_values_next)

                self._estimator.update(state, action, td_target)

                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, self._n_episode, last_reward), end="")

                if done:
                    break

                state = next_state


        return pi, stats


if __name__ == "__main__":
    estimator = Estimator()

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    stats = q_learning(env, estimator, 100, epsilon=0.0)

    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stats, smoothing_window=25)

